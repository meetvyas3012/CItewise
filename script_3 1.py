# Create a comprehensive deployment guide for Streamlit
deployment_guide = '''# 🚀 Streamlit Deployment Guide for Multi-Document Research Assistant

## Quick Start (Local Development)

### 1. **Setup Environment**
```bash
# Clone or navigate to project directory
cd multi-doc-rag

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\\Scripts\\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements_streamlit.txt
```

### 2. **Run Streamlit App**
```bash
# Simple deployment
streamlit run deploy_streamlit.py

# Or run the full UI
streamlit run multi_doc_rag/ui/streamlit_app.py
```

The app will be available at `http://localhost:8501`

## 🌐 Production Deployment Options

### Option 1: Streamlit Cloud (Easiest)

1. **Prepare Repository**
   ```bash
   # Make sure all files are committed
   git add .
   git commit -m "Prepare for Streamlit Cloud deployment"
   git push origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Set main file: `deploy_streamlit.py`
   - Set requirements: `requirements_streamlit.txt`
   - Deploy!

3. **Configure Secrets** (optional)
   - In Streamlit Cloud dashboard, go to "Secrets"
   - Add your configuration:
   ```toml
   [database]
   url = "sqlite:///./data/rag_database.db"
   
   [openai]
   api_key = "your-api-key"
   ```

### Option 2: Docker Deployment

1. **Create Streamlit Dockerfile**
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   # Install system dependencies
   RUN apt-get update && apt-get install -y \\
       tesseract-ocr \\
       tesseract-ocr-eng \\
       poppler-utils \\
       && rm -rf /var/lib/apt/lists/*
   
   # Copy requirements and install Python dependencies
   COPY requirements_streamlit.txt .
   RUN pip install --no-cache-dir -r requirements_streamlit.txt
   
   # Copy application
   COPY . .
   
   # Create directories
   RUN mkdir -p data/documents data/embeddings data/models
   
   # Expose port
   EXPOSE 8501
   
   # Health check
   HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
   
   # Run app
   CMD ["streamlit", "run", "deploy_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Build and Run**
   ```bash
   # Build image
   docker build -t multi-doc-rag:latest .
   
   # Run container
   docker run -p 8501:8501 -v $(pwd)/data:/app/data multi-doc-rag:latest
   ```

### Option 3: VPS/Server Deployment

1. **Setup Server**
   ```bash
   # On your server (Ubuntu/Debian)
   sudo apt update
   sudo apt install python3-pip python3-venv git tesseract-ocr
   
   # Clone repository
   git clone <your-repo-url>
   cd multi-doc-rag
   
   # Setup virtual environment
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements_streamlit.txt
   ```

2. **Run with Process Manager**
   ```bash
   # Install PM2 (Node.js process manager)
   curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
   sudo apt-get install -y nodejs
   sudo npm install -g pm2
   
   # Create ecosystem file
   cat > ecosystem.config.js << EOF
   module.exports = {
     apps: [{
       name: 'multi-doc-rag',
       script: 'streamlit',
       args: 'run deploy_streamlit.py --server.port=8501 --server.address=0.0.0.0',
       interpreter: './venv/bin/python',
       cwd: '/path/to/your/app',
       instances: 1,
       autorestart: true,
       watch: false,
       max_memory_restart: '1G',
       env: {
         NODE_ENV: 'production'
       }
     }]
   }
   EOF
   
   # Start application
   pm2 start ecosystem.config.js
   pm2 save
   pm2 startup
   ```

3. **Setup Reverse Proxy (Nginx)**
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://127.0.0.1:8501;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
           proxy_read_timeout 86400;
       }
   }
   ```

### Option 4: Heroku Deployment

1. **Prepare Heroku Files**
   ```bash
   # Create Procfile
   echo "web: streamlit run deploy_streamlit.py --server.port=$PORT --server.address=0.0.0.0" > Procfile
   
   # Create runtime.txt
   echo "python-3.9.18" > runtime.txt
   
   # Create setup.sh for system dependencies
   cat > setup.sh << EOF
   mkdir -p ~/.streamlit/
   echo "\\
   [server]\\n\\
   headless = true\\n\\
   port = $PORT\\n\\
   enableCORS = false\\n\\
   \\n\\
   [browser]\\n\\
   gatherUsageStats = false\\n\\
   " > ~/.streamlit/config.toml
   EOF
   ```

2. **Deploy to Heroku**
   ```bash
   # Install Heroku CLI and login
   heroku login
   
   # Create app
   heroku create your-app-name
   
   # Set buildpacks
   heroku buildpacks:add --index 1 heroku/python
   heroku buildpacks:add --index 2 https://github.com/heroku/heroku-buildpack-apt
   
   # Create Aptfile for system dependencies
   echo "tesseract-ocr\\ntesseract-ocr-eng\\npoppler-utils" > Aptfile
   
   # Deploy
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

## 🔧 Configuration and Optimization

### Performance Optimization

1. **Memory Management**
   ```python
   # Add to your app
   import streamlit as st
   
   # Enable caching
   @st.cache_resource
   def load_model():
       # Your heavy loading code
       pass
   
   # Limit file upload size
   st.set_page_config(
       page_title="RAG App",
       layout="wide",
       initial_sidebar_state="expanded"
   )
   ```

2. **Environment Variables**
   ```bash
   # Set in your deployment environment
   export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
   export STREAMLIT_SERVER_ENABLE_CORS=false
   export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
   ```

### Security Considerations

1. **Authentication** (Optional)
   ```python
   import streamlit_authenticator as stauth
   
   # Add to your app
   authenticator = stauth.Authenticate(
       credentials,
       'some_cookie_name',
       'some_signature_key',
       cookie_expiry_days=30
   )
   
   name, authentication_status, username = authenticator.login('Login', 'main')
   
   if authentication_status:
       # Your app content
       pass
   ```

2. **HTTPS Setup**
   - Use Let's Encrypt for free SSL certificates
   - Configure nginx or cloudflare for SSL termination

### Monitoring and Logging

1. **Application Monitoring**
   ```python
   import logging
   import streamlit as st
   
   # Setup logging
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
   )
   
   # Log user interactions
   @st.cache_data
   def log_query(query):
       logging.info(f"User query: {query}")
   ```

2. **Health Checks**
   ```python
   # Add health check endpoint
   def health_check():
       return {"status": "healthy", "timestamp": datetime.now()}
   ```

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Make sure all dependencies are installed
   pip install -r requirements_streamlit.txt
   
   # Check Python path
   export PYTHONPATH="${PYTHONPATH}:/path/to/your/app"
   ```

2. **Memory Issues**
   ```python
   # Reduce model size or use CPU-only versions
   # In requirements_streamlit.txt, use:
   # torch>=2.0.0+cpu
   # faiss-cpu>=1.7.4
   ```

3. **File Upload Issues**
   ```bash
   # Increase upload limits in .streamlit/config.toml
   [server]
   maxUploadSize = 200
   ```

4. **Database Connection Issues**
   ```python
   # Ensure data directory exists
   import os
   os.makedirs('data', exist_ok=True)
   ```

### Debug Mode

```bash
# Run in debug mode
streamlit run deploy_streamlit.py --logger.level=debug
```

## 📊 Deployment Checklist

- [ ] ✅ Dependencies installed from `requirements_streamlit.txt`
- [ ] ✅ Data directory created (`data/documents`, `data/embeddings`, etc.)
- [ ] ✅ Streamlit config file configured (`.streamlit/config.toml`)
- [ ] ✅ Environment variables set (if needed)
- [ ] ✅ System dependencies installed (tesseract, poppler-utils)
- [ ] ✅ Port 8501 accessible (or configured port)
- [ ] ✅ SSL certificate configured (for production)
- [ ] ✅ Domain name configured (for production)
- [ ] ✅ Monitoring and logging set up
- [ ] ✅ Backup strategy for data directory
- [ ] ✅ Update mechanism planned

## 🎯 Quick Commands Reference

```bash
# Local development
streamlit run deploy_streamlit.py

# With specific port
streamlit run deploy_streamlit.py --server.port 8502

# Production mode
streamlit run deploy_streamlit.py --server.headless true

# Check Streamlit version
streamlit version

# Clear cache
streamlit cache clear
```

Your Multi-Document Research Assistant is now ready for deployment! 🚀

Choose the deployment option that best fits your needs:
- **Streamlit Cloud**: Easiest for demos and small projects
- **Docker**: Best for consistent environments
- **VPS/Server**: Best for full control and customization
- **Heroku**: Good balance of ease and functionality

Happy deploying! 📚🤖
'''

with open('DEPLOYMENT_GUIDE.md', 'w') as f:
    f.write(deployment_guide)

print("✅ Comprehensive deployment guide created")