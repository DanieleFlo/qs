param(
    [ValidateSet("target", "mtp", "both")]
    [string]$Mode = "both",
    [string]$Model = "gguf/Qwen3.8-27B-UD-Q4_K_S.gguf",
    [string]$MtpModel = "gguf/mtp-Qwen3.8-27B-Q4_0.gguf",
    [string]$Binary = "ds4-server",
    [string]$CudaArch = "sm_86",
    [int]$Context = 22593,
    [int]$Port = 18085,
    [string]$RunId = "",
    [string]$ResultsRoot = "performance-results",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Repo "agent/.venv/Scripts/python.exe"
$TestModule = "tests.test_server_sampling_matrix.SamplingMatrixLiveTests"

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
    $RunId = "q38-server-sampling-matrix-" +
        [DateTime]::UtcNow.ToString("yyyyMMdd-HHmmss")
}
$ResultsPath = if ([System.IO.Path]::IsPathRooted($ResultsRoot)) {
    $ResultsRoot
} else {
    Join-Path $Repo $ResultsRoot
}
$Artifact = Join-Path $ResultsPath $RunId
if (Test-Path -LiteralPath $Artifact) {
    throw "Refusing to overwrite sampling-matrix artifact: $Artifact"
}
New-Item -ItemType Directory -Path $Artifact | Out-Null

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
$Records = @{}
foreach ($Variant in $Variants) {
    $VariantRoot = Join-Path $Artifact $Variant
    $Kv = Join-Path $VariantRoot "kv"
    $Log = Join-Path $VariantRoot "server.log"
    $PidFile = Join-Path $VariantRoot "server.pid"
    $Output = Join-Path $VariantRoot "matrix.json"
    New-Item -ItemType Directory -Path $VariantRoot,$Kv | Out-Null
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
        "DS4_MTP_CYCLE_TRACE=1 '$WslBinary' --cuda -m '$WslModel'" +
        "$MtpArgs --ctx $Context --host 127.0.0.1 --port $Port " +
        "--kv-disk-dir '$WslKv' --kv-disk-space-mb 8192 " +
        "> '$WslLog' 2>&1"
    $quotedCommand = '"' + $command.Replace('"', '\"') + '"'
    $process = Start-Process -FilePath "wsl.exe" `
        -ArgumentList @("-e", "sh", "-lc", $quotedCommand) `
        -WindowStyle Hidden -PassThru
    try {
        Wait-Server
        $env:DS4_SAMPLING_MATRIX_BASE_URL = "http://127.0.0.1:$Port"
        $env:DS4_SAMPLING_MATRIX_MODEL = $ModelId
        $env:DS4_SAMPLING_MATRIX_VARIANT = $Variant
        $env:DS4_SAMPLING_MATRIX_SERVER_LOG = $Log
        $env:DS4_SAMPLING_MATRIX_OUTPUT = $Output
        Push-Location $Repo
        try {
            & $Python -m unittest $TestModule -v
        } finally {
            Pop-Location
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Server sampling matrix failed: $Variant"
        }
        $Records[$Variant] = Get-Content -Raw -LiteralPath $Output |
            ConvertFrom-Json
    } finally {
        Remove-Item Env:DS4_SAMPLING_MATRIX_BASE_URL -ErrorAction SilentlyContinue
        Remove-Item Env:DS4_SAMPLING_MATRIX_MODEL -ErrorAction SilentlyContinue
        Remove-Item Env:DS4_SAMPLING_MATRIX_VARIANT -ErrorAction SilentlyContinue
        Remove-Item Env:DS4_SAMPLING_MATRIX_SERVER_LOG -ErrorAction SilentlyContinue
        Remove-Item Env:DS4_SAMPLING_MATRIX_OUTPUT -ErrorAction SilentlyContinue
        Stop-Server $PidFile
        if (-not $process.HasExited) { $process.WaitForExit(5000) | Out-Null }
    }
}

$Comparison = @()
if ($Mode -eq "both") {
    $TargetCases = @{}
    foreach ($Case in $Records["target"].cases) { $TargetCases[$Case.id] = $Case }
    foreach ($Case in $Records["mtp"].cases) {
        if (-not $TargetCases.ContainsKey($Case.id)) {
            throw "MTP matrix contains an unmatched case: $($Case.id)"
        }
        $TargetCase = $TargetCases[$Case.id]
        # Chat cases have a fixed seed and must be byte-for-byte semantic
        # matches. Responses exposes no seed in DS4. Its thinking phase also
        # runs through the batched verifier, whose documented near-argmax ties
        # can change hidden reasoning while preserving the public contract.
        # Greedy no-thinking Responses cases remain exact target-only checks.
        $Exact = $Case.api -eq "chat_completions" -or (
            [double]$Case.sampling.temperature -eq 0.0 -and
            -not [bool]$Case.sampling.thinking
        )
        $ComparisonKind = if ($Exact) { "semantic_exact" } else { "contract" }
        $TargetHash = if ($Exact) {
            $TargetCase.semantic_sha256
        } else {
            $TargetCase.contract_sha256
        }
        $MtpHash = if ($Exact) {
            $Case.semantic_sha256
        } else {
            $Case.contract_sha256
        }
        $Equal = $TargetHash -eq $MtpHash
        $Comparison += [ordered]@{
            id = $Case.id
            comparison = $ComparisonKind
            target_sha256 = $TargetHash
            mtp_sha256 = $MtpHash
            equal = $Equal
        }
        if (-not $Equal) {
            throw "Target/MTP $ComparisonKind mismatch in sampling case: $($Case.id)"
        }
    }
    if ($Comparison.Count -ne $TargetCases.Count) {
        throw "Target/MTP sampling matrices have different case counts"
    }
}

$Summary = [ordered]@{
    schema_version = 1
    experiment_id = $RunId
    created_at = [DateTime]::UtcNow.ToString("o")
    status = "PASS"
    model = $ModelId
    context = $Context
    variants = $Variants
    matrix_dimensions = [ordered]@{
        temperatures = @(0.0, 0.6, 0.7)
        temperature_classes = @("zero", "nonzero")
        thinking = @($false, $true)
        mtp = @($false, $true)
        scenarios = @("plain", "optional_text", "required_tool")
    }
    target_mtp_comparison = $Comparison
}
$SummaryPath = Join-Path $Artifact "experiment.json"
[System.IO.File]::WriteAllText(
    $SummaryPath,
    ($Summary | ConvertTo-Json -Depth 8) + [Environment]::NewLine,
    [System.Text.UTF8Encoding]::new($false)
)
Write-Output $SummaryPath
