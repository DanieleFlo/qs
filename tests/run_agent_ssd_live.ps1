param(
    [string]$Model = "gguf/Qwen3.6-27B-Q4_K_S.gguf",
    [string]$MtpModel = "",
    [ValidateSet("all", "cold", "warm")]
    [string]$MtpPhase = "all",
    [int]$Context = 30000,
    [int]$Port = 18082,
    [ValidateSet("all", "system", "hds", "skill")]
    [string]$Group = "all"
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Repo "agent/.venv/Scripts/python.exe"
$Test = Join-Path $Repo "tests/test_agent_ssd_live.py"
$Fixture = Join-Path $Repo "tests/agent_ssd_live/ssd-canary.txt"
$Workroot = Join-Path $Repo "tests/agent-ssd-live-work"
$Cache = Join-Path $Repo "tests/agent-ssd-live-kv"
$PidFile = Join-Path $Repo "tests/agent-ssd-live-server.pid"
$WslRepo = (wsl.exe -e wslpath -a $Repo).Trim()
$WslCache = (wsl.exe -e wslpath -a $Cache).Trim()
$WslPid = (wsl.exe -e wslpath -a $PidFile).Trim()
$MtpArgs = ""
$ModelId = [System.IO.Path]::GetFileNameWithoutExtension($Model)
if ($MtpModel) {
    $MtpPath = if ([System.IO.Path]::IsPathRooted($MtpModel)) {
        $MtpModel
    } else {
        Join-Path $Repo $MtpModel
    }
    if (-not (Test-Path -LiteralPath $MtpPath)) {
        throw "MTP model not found: $MtpPath"
    }
    $WslMtp = (wsl.exe -e wslpath -a $MtpPath).Trim()
    $MtpArgs = " --mtp-model '$WslMtp' --mtp-draft 2"
}

if (Test-Path $Cache) {
    $resolved = (Resolve-Path $Cache).Path
    if (-not $resolved.StartsWith($Repo, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove cache outside repository: $resolved"
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
New-Item -ItemType Directory -Path $Cache | Out-Null
if (Test-Path $Workroot) {
    $resolved = (Resolve-Path $Workroot).Path
    if (-not $resolved.StartsWith($Repo, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove workroot outside repository: $resolved"
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
New-Item -ItemType Directory -Path $Workroot | Out-Null
Copy-Item -LiteralPath $Fixture -Destination $Workroot

function Wait-Server {
    $deadline = [DateTime]::UtcNow.AddMinutes(3)
    while ([DateTime]::UtcNow -lt $deadline) {
        try {
            Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 `
                "http://127.0.0.1:$Port/v1/models" | Out-Null
            return
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    throw "ds4-server did not become ready"
}

function Stop-Server {
    if (Test-Path $PidFile) {
        $serverPidText = (Get-Content -Raw $PidFile).Trim()
        $serverPid = 0
        if ([int]::TryParse($serverPidText, [ref]$serverPid) -and $serverPid -gt 1) {
            wsl.exe -e sh -lc "kill -TERM $serverPid 2>/dev/null || true"
            $deadline = [DateTime]::UtcNow.AddSeconds(30)
            while ([DateTime]::UtcNow -lt $deadline) {
                wsl.exe -e sh -lc "kill -0 $serverPid 2>/dev/null"
                if ($LASTEXITCODE -ne 0) { break }
                Start-Sleep -Milliseconds 500
            }
            wsl.exe -e sh -lc "kill -0 $serverPid 2>/dev/null"
            if ($LASTEXITCODE -eq 0) {
                wsl.exe -e sh -lc "kill -KILL $serverPid 2>/dev/null || true"
            }
        }
        Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
    }
}

try {
    wsl.exe -e sh -lc "cd '$WslRepo' && make -j2 ds4-server"
    if ($LASTEXITCODE -ne 0) { throw "ds4-server build failed" }

    $RunGroups = if ($Group -eq "all") { @("system", "hds", "skill") } else { @($Group) }
    foreach ($runGroup in $RunGroups) {
        foreach ($phase in @("cold", "warm")) {
            $Log = Join-Path $Repo "tests/agent-ssd-live-$runGroup-$phase.log"
            if (Test-Path $Log) { Remove-Item -LiteralPath $Log -Force }
            $WslLog = (wsl.exe -e wslpath -a $Log).Trim()
            $PhaseMtpArgs = if ($MtpModel -and
                ($MtpPhase -eq "all" -or $MtpPhase -eq $phase)) { $MtpArgs } else { "" }
            $command = "cd '$WslRepo' && echo `$`$ > '$WslPid' && exec ./ds4-server --cuda -m '$Model'$PhaseMtpArgs --ctx $Context --host 0.0.0.0 --port $Port --kv-disk-dir '$WslCache' --kv-disk-space-mb 8192 > '$WslLog' 2>&1"
            $quotedCommand = '"' + $command.Replace('"', '\"') + '"'
            $process = Start-Process -FilePath "wsl.exe" `
                -ArgumentList @("-e", "sh", "-lc", $quotedCommand) `
                -WindowStyle Hidden -PassThru
            try {
                Wait-Server
                $WarmCacheArgs = if ($phase -eq "warm" -and $MtpModel -and
                    $MtpPhase -ne "all") { @("--warm-cache", "refresh") } else { @() }
                & $Python $Test --phase $phase `
                    --base-url "http://127.0.0.1:$Port/v1" --server-log $Log `
                    --workroot $Workroot --group $runGroup --model-id $ModelId `
                    @WarmCacheArgs
                if ($LASTEXITCODE -ne 0) {
                    throw "Agent SSD live phase failed: $runGroup/$phase"
                }
            } finally {
                Stop-Server
                if (-not $process.HasExited) { $process.WaitForExit(5000) | Out-Null }
            }
        }
    }
} finally {
    Stop-Server
    if (Test-Path $Workroot) {
        $resolved = (Resolve-Path $Workroot).Path
        if ($resolved.StartsWith($Repo, [System.StringComparison]::OrdinalIgnoreCase)) {
            Remove-Item -LiteralPath $resolved -Recurse -Force
        }
    }
}

$Mode = if ($MtpModel -and $MtpPhase -eq "warm") {
    " with a target-only to MTP restart"
} elseif ($MtpModel -and $MtpPhase -eq "cold") {
    " with an MTP to target-only restart"
} elseif ($MtpModel) {
    " with MTP"
} else {
    ""
}
Write-Host "Agent SSD live cold/warm suite$Mode passed."
