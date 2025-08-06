# 🚀 Quick Deployment Guide

## Fastest Start (One Command)

### Linux/Mac:
```bash
./run_streamlit.sh
```

### Windows:
```batch
run_streamlit.bat
```

### Manual:
```bash
pip install -r requirements_streamlit.txt
streamlit run deploy_streamlit.py
```

## Docker Deployment:
```bash
docker-compose -f docker-compose.streamlit.yml up --build
```

## Streamlit Cloud:
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Set main file: `deploy_streamlit.py`
5. Deploy!

## Access:
- Local: http://localhost:8501
- Cloud: Your assigned Streamlit Cloud URL

## Usage:
1. Upload documents (PDF, DOCX, TXT, HTML, MD)
2. Wait for processing
3. Ask questions
4. Get AI responses with citations

For detailed instructions, see `DEPLOYMENT_GUIDE.md`
