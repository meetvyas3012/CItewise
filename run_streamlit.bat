@echo off
REM Multi-Document Research Assistant - Streamlit Deployment Script for Windows

echo 🚀 Multi-Document Research Assistant - Streamlit Deployment
echo ==================================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed. Please install Python 3.9 or higher.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    echo [SUCCESS] Virtual environment created
) else (
    echo [INFO] Using existing virtual environment
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
if exist "requirements_streamlit.txt" (
    echo [INFO] Installing dependencies from requirements_streamlit.txt...
    pip install -r requirements_streamlit.txt
    echo [SUCCESS] Dependencies installed
) else (
    echo [WARNING] requirements_streamlit.txt not found, using main requirements.txt
    if exist "requirements.txt" (
        pip install -r requirements.txt
        echo [SUCCESS] Dependencies installed from requirements.txt
    ) else (
        echo [ERROR] No requirements file found!
        pause
        exit /b 1
    )
)

REM Create necessary directories
echo [INFO] Creating data directories...
if not exist "data" mkdir data
if not exist "data\documents" mkdir data\documents
if not exist "data\embeddings" mkdir data\embeddings
if not exist "data\models" mkdir data\models
if not exist "data\temp" mkdir data\temp
if not exist "logs" mkdir logs
echo [SUCCESS] Data directories created

REM Create Streamlit config if it doesn't exist
if not exist ".streamlit\config.toml" (
    echo [INFO] Creating Streamlit configuration...
    if not exist ".streamlit" mkdir .streamlit
    echo [global] > .streamlit\config.toml
    echo developmentMode = false >> .streamlit\config.toml
    echo. >> .streamlit\config.toml
    echo [server] >> .streamlit\config.toml
    echo port = 8501 >> .streamlit\config.toml
    echo address = "0.0.0.0" >> .streamlit\config.toml
    echo headless = true >> .streamlit\config.toml
    echo enableCORS = true >> .streamlit\config.toml
    echo maxUploadSize = 200 >> .streamlit\config.toml
    echo. >> .streamlit\config.toml
    echo [browser] >> .streamlit\config.toml
    echo gatherUsageStats = false >> .streamlit\config.toml
    echo. >> .streamlit\config.toml
    echo [logger] >> .streamlit\config.toml
    echo level = "info" >> .streamlit\config.toml
    echo [SUCCESS] Streamlit configuration created
)

REM Start the application
echo [INFO] Starting application...
echo.
echo 🌐 Your app will be available at: http://localhost:8501
echo 📚 Upload some documents and start asking questions!
echo ⏹️  Press Ctrl+C to stop the application
echo.

REM Run the application
if exist "deploy_streamlit.py" (
    streamlit run deploy_streamlit.py
) else (
    streamlit run multi_doc_rag\ui\streamlit_app.py
)

pause
