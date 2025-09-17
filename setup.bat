@echo off
echo ========================================
echo  Dashboard Media Sosial - Setup Script
echo ========================================

echo.
echo [1/3] Mengaktifkan Virtual Environment...
call VisualEnv\Scripts\activate.bat

echo.
echo [2/3] Menginstall Dependencies...
pip install -r requirements.txt

echo.
echo [3/3] Setup selesai!
echo.
echo Untuk menjalankan dashboard:
echo streamlit run dashboard.py
echo.
echo Dashboard akan terbuka di: http://localhost:8501
echo.
pause 