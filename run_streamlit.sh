#!/bin/bash

# Multi-Document Research Assistant - Streamlit Deployment Script

echo "🚀 Multi-Document Research Assistant - Streamlit Deployment"
echo "=================================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_status "Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_status "Using existing virtual environment"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install requirements
if [ -f "requirements_streamlit.txt" ]; then
    print_status "Installing dependencies from requirements_streamlit.txt..."
    pip install -r requirements_streamlit.txt
    print_success "Dependencies installed"
else
    print_warning "requirements_streamlit.txt not found, using main requirements.txt"
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Dependencies installed from requirements.txt"
    else
        print_error "No requirements file found!"
        exit 1
    fi
fi

# Create necessary directories
print_status "Creating data directories..."
mkdir -p data/documents data/embeddings data/models data/temp logs
print_success "Data directories created"

# Check if Streamlit config exists
if [ ! -f ".streamlit/config.toml" ]; then
    print_status "Creating Streamlit configuration..."
    mkdir -p .streamlit
    cat > .streamlit/config.toml << EOF
[global]
developmentMode = false

[server]
port = 8501
address = "0.0.0.0"
headless = true
enableCORS = true
maxUploadSize = 200

[browser]
gatherUsageStats = false

[logger]
level = "info"
EOF
    print_success "Streamlit configuration created"
fi

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        return 1
    else
        return 0
    fi
}

# Find available port
PORT=8501
while ! check_port $PORT; do
    print_warning "Port $PORT is already in use, trying $((PORT+1))..."
    PORT=$((PORT+1))
done

print_success "Using port $PORT"

# Check deployment mode
DEPLOYMENT_MODE=${1:-"local"}

case $DEPLOYMENT_MODE in
    "local"|"dev")
        print_status "Starting in local development mode..."
        echo ""
        echo "🌐 Your app will be available at: http://localhost:$PORT"
        echo "📚 Upload some documents and start asking questions!"
        echo "⏹️  Press Ctrl+C to stop the application"
        echo ""

        # Run the simplified deployment script
        if [ -f "deploy_streamlit.py" ]; then
            streamlit run deploy_streamlit.py --server.port=$PORT
        else
            # Fallback to main UI
            streamlit run multi_doc_rag/ui/streamlit_app.py --server.port=$PORT
        fi
        ;;

    "docker")
        print_status "Starting with Docker..."

        if ! command -v docker &> /dev/null; then
            print_error "Docker is not installed. Please install Docker first."
            exit 1
        fi

        if [ -f "docker-compose.streamlit.yml" ]; then
            print_status "Using docker-compose for deployment..."
            docker-compose -f docker-compose.streamlit.yml up --build
        else
            print_status "Building and running Docker container..."
            docker build -f Dockerfile.streamlit -t multi-doc-rag:streamlit .
            docker run -p 8501:8501 -v $(pwd)/data:/app/data multi-doc-rag:streamlit
        fi
        ;;

    "production")
        print_status "Starting in production mode..."
        print_warning "Make sure you have configured:"
        print_warning "- Reverse proxy (nginx/apache)"
        print_warning "- SSL certificates"
        print_warning "- Process manager (pm2/systemd)"

        echo ""
        echo "🌐 Production URL: http://your-domain.com"
        echo "🔒 Make sure to configure HTTPS!"
        echo ""

        streamlit run deploy_streamlit.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
        ;;

    *)
        print_error "Unknown deployment mode: $DEPLOYMENT_MODE"
        echo "Usage: $0 [local|docker|production]"
        echo ""
        echo "Deployment modes:"
        echo "  local      - Local development (default)"
        echo "  docker     - Docker containerized deployment"
        echo "  production - Production server deployment"
        exit 1
        ;;
esac
