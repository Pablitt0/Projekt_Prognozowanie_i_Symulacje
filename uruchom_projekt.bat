@echo off
cd /d "%~dp0"
echo ============================================================
echo PROJEKT: Prognoza Zuzycia Energii Elektrycznej w Polsce
echo ============================================================
echo.
echo Uruchamianie analizy (Zadania 2-5)...
py analiza.py
echo.
echo ============================================================
echo Gotowe! Wszystkie pliki PNG zapisane w folderze projektu.
echo ============================================================
pause
