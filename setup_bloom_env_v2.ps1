# Setup script for Bloomberg Python API

$envName = "bloom_env"
$whlPath = "blpapi-3.25.3-py3-none-win_amd64.whl"
$dllName = "blpapi3_64.dll"

# 1. Create virtual environment if it doesn't exist
if (-Not (Test-Path $envName)) {
    Write-Host "Creating virtual environment '$envName'..."
    python -m venv $envName
}

# 2. Activate the environment
& ".\$envName\Scripts\Activate.ps1"

# 3. Install the wheel
if (Test-Path $whlPath) {
    pip install $whlPath
} else {
    Write-Host "Wheel file not found: $whlPath"
    exit 1
}
# 4. Copy DLL into correct site-packages directory
$blpapiPath = ".\$envName\Lib\site-packages\blpapi"
$dllSource = Join-Path $PSScriptRoot $dllName
$dllDest = Join-Path $blpapiPath $dllName

if (Test-Path $blpapiPath) {
    Write-Host "Copying DLL to $dllDest"
    Copy-Item -Path $dllSource -Destination $dllDest -Force
} else {
    Write-Host "[ERROR] blpapi install path not found: $blpapiPath"
    exit 1
}


# 5. Test
python -c "import blpapi; print('blpapi loaded. Version:', blpapi.__version__)"
