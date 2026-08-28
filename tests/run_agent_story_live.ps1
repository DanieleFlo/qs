param(
    [ValidateSet("target", "mtp", "both")]
    [string]$Mode = "both",
    [string]$Model = "gguf/Qwen3.8-27B-UD-Q4_K_S.gguf",
    [string]$MtpModel = "gguf/mtp-Qwen3.8-27B-Q4_0.gguf",
    [string]$Binary = "ds4-server",
    [string]$CudaArch = "sm_86",
    [int]$Context = 22593,
    [int]$Port = 18084,
    [int]$Warmup = 1,
    [int]$Repetitions = 2,
    [int]$MinimumContext = 10000,
    [int]$MinimumOutputTokens = 400,
    [int]$MinimumWords = 250,
    [string]$RunId = "",
    [string]$ResultsRoot = "performance-results",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Repo "agent/.venv/Scripts/python.exe"
$Profiler = Join-Path $Repo "tools/profile_agent_dsml_story.py"

function Resolve-RepoPath([string]$Path) {
    $candidate = if ([System.IO.Path]::IsPathRooted($Path)) {
        $Path
    } else {
        Join-Path $Repo $Path
    }
    return (Resolve-Path -LiteralPath $candidate).Path
}

function Convert-ToWslPath([string]$Path) {
    $converted = (wsl.exe -e wslpath -a $Path).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $converted) {
        throw "Unable to convert path for WSL: $Path"
    }
    if ($converted.Contains("'")) {
        throw "Paths containing a single quote are not supported: $converted"
    }
    return $converted
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Agent Wiki Python environment not found: $Python"
}
$ModelPath = Resolve-RepoPath $Model
$WslRepo = Convert-ToWslPath $Repo
$WslModel = Convert-ToWslPath $ModelPath
$ModelId = [System.IO.Path]::GetFileNameWithoutExtension($ModelPath)
$WslMtpModel = ""
if ($Mode -ne "target") {
    $MtpModelPath = Resolve-RepoPath $MtpModel
    $WslMtpModel = Convert-ToWslPath $MtpModelPath
}

if (-not $RunId) {
    $RunId = "q38-agent-plain-story-" +
        [DateTime]::UtcNow.ToString("yyyyMMdd-HHmmss")
}
$ResultsPath = if ([System.IO.Path]::IsPathRooted($ResultsRoot)) {
    $ResultsRoot
} else {
    Join-Path $Repo $ResultsRoot
}
New-Item -ItemType Directory -Force -Path $ResultsPath | Out-Null

function Wait-Server {
    $deadline = [DateTime]::UtcNow.AddMinutes(4)
    while ([DateTime]::UtcNow -lt $deadline) {
        try {
            Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 `
                "http://127.0.0.1:$Port/v1/models" | Out-Null
            return
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    throw "ds4-server did not become ready on port $Port"
}

function Stop-Server([string]$PidFile) {
    if (-not (Test-Path -LiteralPath $PidFile)) { return }
    $serverPidText = (Get-Content -Raw -LiteralPath $PidFile).Trim()
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

if (-not $SkipBuild) {
    wsl.exe -e sh -lc `
        "cd '$WslRepo' && make -j2 ds4-server CUDA_ARCH='$CudaArch'"
    if ($LASTEXITCODE -ne 0) { throw "ds4-server build failed" }
}
$BinaryPath = Resolve-RepoPath $Binary
$WslBinary = Convert-ToWslPath $BinaryPath

$Variants = if ($Mode -eq "both") { @("target", "mtp") } else { @($Mode) }
$Outputs = @()
foreach ($Variant in $Variants) {
    $ExperimentId = "$RunId-$Variant"
    $Artifact = Join-Path $ResultsPath $ExperimentId
    if (Test-Path -LiteralPath $Artifact) {
        throw "Refusing to overwrite live story artifact: $Artifact"
    }
    $Workroot = Join-Path $Artifact "workroot"
    $Kv = Join-Path $Artifact "kv"
    $Log = Join-Path $Artifact "server.log"
    $PidFile = Join-Path $Artifact "server.pid"
    $Output = Join-Path $Artifact "experiment.json"
    New-Item -ItemType Directory -Path $Artifact,$Workroot,$Kv | Out-Null

    $WslKv = Convert-ToWslPath $Kv
    $WslLog = Convert-ToWslPath $Log
    $WslPid = Convert-ToWslPath $PidFile
    $MtpArgs = if ($Variant -eq "mtp") {
        " --mtp-model '$WslMtpModel' --mtp-draft 2"
    } else {
        ""
    }
    $command = "cd '$WslRepo' && echo `$`$ > '$WslPid' && " +
        "exec env DS4_SERVER_PHASE_PROFILE=1 DS4_MTP_STATS=1 " +
        "'$WslBinary' --cuda -m '$WslModel'$MtpArgs --ctx $Context " +
        "--host 127.0.0.1 --port $Port --kv-disk-dir '$WslKv' " +
        "--kv-disk-space-mb 8192 > '$WslLog' 2>&1"
    $quotedCommand = '"' + $command.Replace('"', '\"') + '"'
    $process = Start-Process -FilePath "wsl.exe" `
        -ArgumentList @("-e", "sh", "-lc", $quotedCommand) `
        -WindowStyle Hidden -PassThru
    try {
        Wait-Server
        $ProfileArgs = @(
            $Profiler,
            "--base-url", "http://127.0.0.1:$Port/v1",
            "--server-log", $Log,
            "--output", $Output,
            "--workroot", $Workroot,
            "--model", $ModelId,
            "--scenario", "plain-story",
            "--experiment-id", $ExperimentId,
            "--warmup", $Warmup,
            "--repetitions", $Repetitions,
            "--minimum-context", $MinimumContext,
            "--minimum-output-tokens", $MinimumOutputTokens,
            "--minimum-words", $MinimumWords
        )
        & $Python @ProfileArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Agent plain-story live test failed: $Variant"
        }
        $Outputs += $Output
    } finally {
        Stop-Server $PidFile
        if (-not $process.HasExited) { $process.WaitForExit(5000) | Out-Null }
    }
}

$Outputs | ForEach-Object { Write-Output $_ }
