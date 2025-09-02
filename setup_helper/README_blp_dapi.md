# Bloomberg Python API (blpapi) Setup Guide

This README provides detailed steps to set up the Bloomberg Python API (`blpapi`) in a fresh Windows environment using a prebuilt `.whl` package and supporting DLL.

## Requirements

### 1. Python Environment

* Python version **3.10.0** (64-bit only)
* Python **must be added to PATH** during installation

### 2. IDE/Terminal (recommended)

* **PowerShell** (used to execute the setup script)
* Optional: **Visual Studio Code (VS Code)** for development

### 3. Files (must be in the same directory)

* `setup_bloom_env.ps1`: PowerShell script that automates the setup
* `blpapi-3.25.3-py3-none-win_amd64.whl`: Precompiled Bloomberg Python API package
* `blpapi3_64.dll`: Bloomberg runtime DLL required by the API

Directory structure example:

```
C:\blp\API\api_wheel\
    |-- setup_bloom_env.ps1
    |-- blpapi-3.25.3-py3-none-win_amd64.whl
    |-- blpapi3_64.dll
```

## One-Time Setup

### Step 1: Extract and Open PowerShell

* Unzip all files into a folder, e.g., `C:\blp\API\api_wheel\`
* Open PowerShell and navigate to the directory:

  ```powershell
  cd C:\blp\API\api_wheel
  ```

### Step 2: Run the Script

* Execute the setup script:

  ```powershell
  .\setup_bloom_env.ps1
  ```
* The script will:

  1. Create a virtual environment `bloom_env/`
  2. Install the `.whl` package inside the venv
  3. Copy the `.dll` file to the correct `site-packages/blpapi/` location
  4. Run a test import to verify the setup

## Common Issues

### Issue: Python not found

* Ensure Python 3.10 is installed and `python` is accessible in PowerShell

### Issue: Script not executing due to policy

* Temporarily allow PowerShell scripts:

  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  ```

### Issue: blpapi3\_64.dll not found

* Make sure the `blpapi3_64.dll` file is in the same folder as the script before running

## After Setup

### Activate Environment Manually (if needed):

```powershell
.\bloom_env\Scripts\Activate.ps1
```

### Test API Access:

```python
from xbbg import blp
print(blp.bdp(tickers='SPY US Equity', flds='last_price'))
```

## Notes

* The virtual environment is portable as long as paths and DLLs remain aligned
* No need to reinstall the SDK or build tools every time — reuse this package on any Bloomberg terminal station

---

For support or enhancements, refer to the documentation or contact your IT/Bloomberg administrator.
