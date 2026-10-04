@echo off
REM ── Steel Billet Inspection System — Windows Launcher ──
echo ============================================
echo  Steel Billet Inspection ^& Traceability
echo  CodeUtsava X.0 - Team INIT
echo ============================================

REM Check if virtual environment already exists
if exist "venv\Scripts\activate.bat" (
    echo Activating existing virtual environment...
    call venv\Scripts\activate.bat
    goto :venv_ready
)

REM Find best compatible Python (prefer 3.11, 3.12, 3.10, or system python)
set PYCMD=
py -3.11 --version >nul 2>&1 && (set PYCMD=py -3.11 & goto :found_python)
py -3.12 --version >nul 2>&1 && (set PYCMD=py -3.12 & goto :found_python)
py -3.10 --version >nul 2>&1 && (set PYCMD=py -3.10 & goto :found_python)
py --version >nul 2>&1 && (set PYCMD=py & goto :found_python)
python --version >nul 2>&1 && (set PYCMD=python & goto :found_python)
if exist "D:\MiniConda\envs\Main\python.exe" (set PYCMD=D:\MiniConda\envs\Main\python.exe & goto :found_python)
if exist "D:\MiniConda\python.exe" (set PYCMD=D:\MiniConda\python.exe & goto :found_python)

echo ERROR: Compatible Python not found. Please install Python 3.8-3.12 or Anaconda/Miniconda.
pause
exit /b 1

:found_python
echo Found Python environment:
%PYCMD% --version

echo Creating virtual environment (venv) ...
%PYCMD% -m venv venv
call venv\Scripts\activate.bat

:venv_ready
REM Upgrade pip and install deps
echo.
echo Checking dependencies ...
python -m pip install -r requirements.txt --quiet

REM Create .env if missing
if not exist .env (
    if exist .env.example copy .env.example .env >nul 2>&1
)

REM Launch Streamlit app
echo.
echo ============================================
echo  Starting application at http://localhost:8501
echo ============================================
echo.
python -m streamlit run app.py --server.headless true
pause
