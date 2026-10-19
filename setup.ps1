$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$pythonVersion = (Get-Content -Raw -LiteralPath (Join-Path $projectRoot '.python-version')).Trim()
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if ($uvCommand) {
    $uvExecutable = $uvCommand.Source
} else {
    $uvExecutable = Join-Path $env:USERPROFILE '.local\bin\uv.exe'
}
if (-not (Test-Path -LiteralPath $uvExecutable)) {
    throw 'uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/ and run setup.ps1 again.'
}
$previousInstallDirectory = [Environment]::GetEnvironmentVariable('UV_PYTHON_INSTALL_DIR', 'Process')
$previousProjectEnvironment = [Environment]::GetEnvironmentVariable('UV_PROJECT_ENVIRONMENT', 'Process')
Push-Location -LiteralPath $projectRoot
try {
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $projectRoot '.python-runtime'
    $env:UV_PROJECT_ENVIRONMENT = Join-Path $projectRoot '.venv'
    & $uvExecutable python install $pythonVersion --install-dir $env:UV_PYTHON_INSTALL_DIR --no-bin --no-registry --no-progress
    if ($LASTEXITCODE -ne 0) { throw 'Standalone Python installation failed.' }
    $runtimePython = & $uvExecutable python find $pythonVersion --managed-python --no-python-downloads
    if ($LASTEXITCODE -ne 0) { throw 'Standalone Python was not found.' }
    $runtimePython = ($runtimePython | Select-Object -Last 1).Trim()
    & $uvExecutable sync --locked --project $projectRoot --python $runtimePython --no-python-downloads --link-mode copy --no-progress
    if ($LASTEXITCODE -ne 0) { throw 'Project dependency installation failed.' }
    $projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
    & $uvExecutable pip check --python $projectPython
    if ($LASTEXITCODE -ne 0) { throw 'Dependency validation failed.' }
    & $projectPython (Join-Path $projectRoot 'main.py') demo
    if ($LASTEXITCODE -ne 0) { throw 'Offline startup validation failed.' }
    Write-Output "Ready. PyCharm interpreter: $projectPython"
} finally {
    Pop-Location
    [Environment]::SetEnvironmentVariable('UV_PYTHON_INSTALL_DIR', $previousInstallDirectory, 'Process')
    [Environment]::SetEnvironmentVariable('UV_PROJECT_ENVIRONMENT', $previousProjectEnvironment, 'Process')
}
