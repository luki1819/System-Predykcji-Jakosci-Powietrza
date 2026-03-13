@echo off
echo ========================================
echo   AQPS Dashboard - Uruchamianie
echo ========================================
echo.
echo Sprawdzanie srodowiska...

REM Aktywuj venv jesli istnieje
if exist "..\..\venv\Scripts\activate.bat" (
    echo Aktywacja srodowiska wirtualnego...
    call ..\..\venv\Scripts\activate.bat
) else if exist "..\..\.venv\Scripts\activate.bat" (
    echo Aktywacja srodowiska wirtualnego...
    call ..\..\.venv\Scripts\activate.bat
) else (
    echo Nie znaleziono srodowiska wirtualnego
)

echo.
echo start
echo interfejs otworzy sie w przegladarce pod adresem: http://localhost:8501
echo.
echo żeby wyłączyć to ctrl+c i potwierdzić albo zamknąć okno terminala
echo
echo.

streamlit run dashboard.py

pause