param(
    [string]$Model = "gguf/Qwen3.6-27B-Q4_K_S.gguf",
    [int]$Port = 18082,
    [ValidateSet("all", "system", "hds", "skill")]
    [string]$Group = "all"
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Repo "agent/.venv/Scripts/python.exe"
$Test = Join-Path $Repo "tests/test_agent_ssd_live.py"
$Cache = Join-Path $Repo "tests/agent-ssd-live-kv"
$PidFile = Join-Path $Repo "tests/agent-ssd-live-server.pid"
$WslRepo = (wsl.exe -e wslpath -a $Repo).Trim()
$WslCache = (wsl.exe -e wslpath -a $Cache).Trim()
$WslPid = (wsl.exe -e wslpath -a $PidFile).Trim()

if (Test-Path $Cache) {
    $resolved = (Resolve-Path $Cache).Path
    if (-not $resolved.StartsWith($Repo, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove cache outside repository: $resolved"
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
New-Item -ItemType Directory -Path $Cache | Out-Null

wsl.exe -e sh -lc "cd '$WslRepo' && make -j2 ds4-server"
if ($LASTEXITCODE -ne 0) { throw "ds4-server build failed" }

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
        }
        Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 2
}

$RunGroups = if ($Group -eq "all") { @("system", "hds", "skill") } else { @($Group) }
foreach ($runGroup in $RunGroups) {
    foreach ($phase in @("cold", "warm")) {
        $Log = Join-Path $Repo "tests/agent-ssd-live-$runGroup-$phase.log"
        if (Test-Path $Log) { Remove-Item -LiteralPath $Log -Force }
        $WslLog = (wsl.exe -e wslpath -a $Log).Trim()
        $command = "cd '$WslRepo' && echo `$`$ > '$WslPid' && exec ./ds4-server --cuda -m '$Model' --ctx 32768 --host 0.0.0.0 --port $Port --kv-disk-dir '$WslCache' --kv-disk-space-mb 8192 > '$WslLog' 2>&1"
        $quotedCommand = '"' + $command.Replace('"', '\"') + '"'
        $process = Start-Process -FilePath "wsl.exe" `
            -ArgumentList @("-e", "sh", "-lc", $quotedCommand) `
            -WindowStyle Hidden -PassThru
        try {
            Wait-Server
            & $Python $Test --phase $phase `
                --base-url "http://127.0.0.1:$Port/v1" --server-log $Log --group $runGroup
            if ($LASTEXITCODE -ne 0) {
                throw "Agent SSD live phase failed: $runGroup/$phase"
            }
        } finally {
            Stop-Server
            if (-not $process.HasExited) { $process.WaitForExit(5000) | Out-Null }
        }
    }
}

Write-Host "Agent SSD live cold/warm suite passed."
