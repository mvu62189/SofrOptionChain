# setup_bloom_env_v3.ps1

$envName = ".env/"
$whlPath = "blpapi-3.25.3-py3-none-win_amd64.whl"
$dllName = "blpapi3_64.dll"

# 1. Create virtual environment if it doesn't exist
if (-Not (Test-Path $envName)) {
    Write-Host "`n Creating virtual environment '$envName'..."
    python -m venv $envName
} else {
    Write-Host "`n  Virtual environment '$envName' already exists."
}

# 2. Activate the environment
Write-Host "`n Activating environment..."
& ".\$envName\Scripts\Activate.ps1"

# 3. Install the wheel
if (Test-Path $whlPath) {
    Write-Host "`n Installing blpapi from wheel: $whlPath"
    pip install $whlPath
} else {
    Write-Host "`n Wheel file not found: $whlPath"
    exit 1
}

# 4. Copy DLL to blpapi install path
$blpapiPath = ".\$envName\Lib\site-packages\blpapi"
$dllSource = Join-Path $PSScriptRoot $dllName
$dllDest = Join-Path $blpapiPath $dllName

if (Test-Path $blpapiPath) {
    Write-Host "`n Copying DLL to: $dllDest"
    Copy-Item -Path $dllSource -Destination $dllDest -Force
} else {
    Write-Host "`n blpapi site-packages path not found: $blpapiPath"
    exit 1
}

# 5. Test import
Write-Host "`n Verifying blpapi installation:"
python -c "import blpapi; print('blpapi loaded. Version:', blpapi.__version__)"