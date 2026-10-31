@echo off
rem The libraries the bot needs, in the folder it's installed in, for the Python that came with it.
rem Setup runs this once its files are in place.
cd /d "%~dp0.."
set "UV_CACHE_DIR=%CD%\tools\cache"
set UV_PYTHON_DOWNLOADS=never
set UV_NO_PROGRESS=1
"%~dp0uv.exe" sync --frozen --no-dev --extra browser --python "%CD%\python\python.exe"
set code=%errorlevel%
rmdir /s /q "%UV_CACHE_DIR%" 2>nul
exit /b %code%
