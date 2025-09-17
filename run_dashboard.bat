@echo off
echo ========================================
echo    Menjalankan Dashboard Media Sosial
echo ========================================

echo.
echo Mengaktifkan Virtual Environment...
call VisualEnv\Scripts\activate.bat

echo.
echo Menjalankan Dashboard...
echo Dashboard akan terbuka di: http://localhost:8501
echo.
echo Tekan Ctrl+C untuk menghentikan dashboard
echo.

streamlit run dashboard.py 