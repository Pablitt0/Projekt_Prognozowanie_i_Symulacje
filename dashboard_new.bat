@echo off
cd /d "%~dp0"
echo Uruchamianie dashboardu Streamlit (nowy projekt)...
echo Otworz przegladarke: http://localhost:8501
echo (aby zatrzymac: Ctrl+C)
echo.
py -m streamlit run dashboard_new.py
pause
