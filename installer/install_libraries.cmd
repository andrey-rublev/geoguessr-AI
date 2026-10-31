@echo off
rem Python and the libraries the bot needs, all in the folder it's installed in, whatever Python
rem this computer has or hasn't got. Setup runs this once its files are in place.
cd /d "%~dp0.."
set "UV_PYTHON_INSTALL_DIR=%CD%\python"
set "UV_CACHE_DIR=%CD%\tools\cache"
set UV_PYTHON_PREFERENCE=only-managed
set UV_NO_PROGRESS=1
"%~dp0uv.exe" sync --frozen --no-dev --extra browser --python 3.13
set code=%errorlevel%
rmdir /s /q "%UV_CACHE_DIR%" 2>nul
exit /b %code%
