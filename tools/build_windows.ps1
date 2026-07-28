param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Create .venv before building: py -3.12 -m venv .venv"
}

Push-Location $ProjectRoot
try {
    if (-not $SkipInstall) {
        & $Python -m pip install "pyinstaller>=6.11,<7"
        if ($LASTEXITCODE -ne 0) {
            throw "Build dependency installation failed with exit code $LASTEXITCODE"
        }
    }

    & $Python -m PyInstaller --noconfirm --clean LyreHelper.spec
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }

    $Executable = Join-Path $ProjectRoot "dist\LyreHelper\LyreHelper.exe"
    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
        throw "Build completed without producing $Executable"
    }
    $SmokeTest = Start-Process `
        -FilePath $Executable `
        -ArgumentList "--build-smoke-test" `
        -WorkingDirectory (Split-Path $Executable) `
        -Wait `
        -PassThru
    if ($SmokeTest.ExitCode -ne 0) {
        throw "Packaged ONNX runtime smoke test failed with exit code $($SmokeTest.ExitCode)"
    }
    Write-Output $Executable
}
finally {
    Pop-Location
}
