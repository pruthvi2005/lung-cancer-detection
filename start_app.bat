@echo off
echo Starting AI Lung Cancer Detection App...
echo.
echo Installing dependencies...
pip install flask tensorflow pillow numpy
echo.
echo Starting Flask server...
python app.py
echo.
echo App will be available at: http://127.0.0.1:5000
pause
