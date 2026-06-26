@echo off
setlocal
set PROJECT_ROOT=%~1
if "%PROJECT_ROOT%"=="" (
  echo Usage: run_pipeline.bat ^<project-root^>
  exit /b 1
)
python scripts\run_all.py --project-root "%PROJECT_ROOT%"
