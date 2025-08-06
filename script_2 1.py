# Create Streamlit configuration files
streamlit_config = '''[global]
developmentMode = false
dataFrameSerialization = "legacy"

[server]
port = 8501
address = "0.0.0.0"
headless = true
enableCORS = true
enableXsrfProtection = false
maxUploadSize = 200

[browser]
serverAddress = "localhost"
gatherUsageStats = false

[logger]
level = "info"

[client]
caching = true
displayEnabled = true
'''

# Create .streamlit directory
import os
os.makedirs('.streamlit', exist_ok=True)

with open('.streamlit/config.toml', 'w') as f:
    f.write(streamlit_config)

# Create secrets template
secrets_template = '''# Streamlit Secrets Configuration
# Copy this to .streamlit/secrets.toml and fill in your values

[database]
url = "sqlite:///./data/rag_database.db"

[openai]
api_key = "your-openai-api-key-here"  # Optional

[auth]
# Optional authentication
enabled = false
username = "admin"
password = "password123"

[app]
title = "Multi-Document Research Assistant"
debug = false
'''

with open('.streamlit/secrets.toml.template', 'w') as f:
    f.write(secrets_template)

print("✅ Streamlit configuration files created")