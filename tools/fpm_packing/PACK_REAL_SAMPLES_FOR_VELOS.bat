@echo off
rem Packs the central 512 x 512 pixels of every FPM frame in "Real Samples" into "_velos_packed".
rem Originals are only read. If Python is not found, run pack_fpm_for_velos.m in MATLAB instead.
cd /d "%~dp0"
set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY if exist "%USERPROFILE%\anaconda3\python.exe" set "PY=%USERPROFILE%\anaconda3\python.exe"
if not defined PY if exist "C:\ProgramData\anaconda3\python.exe" set "PY=C:\ProgramData\anaconda3\python.exe"
if not defined PY if exist "%USERPROFILE%\miniconda3\python.exe" set "PY=%USERPROFILE%\miniconda3\python.exe"
if not defined PY if exist "C:\ProgramData\miniconda3\python.exe" set "PY=C:\ProgramData\miniconda3\python.exe"
if not defined PY (
  py -3 --version >nul 2>nul && set "PY=py -3"
)
if not defined PY (
  echo Python was not found. Please open MATLAB in this folder and run:  pack_fpm_for_velos
  pause
  exit /b 1
)
echo Using %PY%
%PY% pack_fpm_for_velos.py 512
pause
