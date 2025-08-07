# Multi-Document Research Assistant 📚

A comprehensive RAG (Retrieval-Augmented Generation) system that enables users to pose complex questions across a corpus of documents and receive synthesized answers with inline citations. Built entirely with open-source tools for privacy-preserving, cost-effective document analysis.

## 🚀 Features

- **📄 Multi-Format Document Support**: PDF, DOCX, HTML, TXT, MD with OCR for scanned documents
- **🧠 Advanced Text Processing**: Intelligent chunking, structure preservation, metadata extraction
- **🔍 Hybrid Search**: Combines semantic vector search with traditional BM25 keyword search
- **🤖 Local LLM Integration**: Privacy-preserving responses using GPT4All, Llama 2, or Vicuna
- **📖 Citation System**: Traceable sources with inline citations for transparency
- **🌐 Interactive Web Interface**: User-friendly Streamlit application
- **⚡ Production Ready**: Docker deployment with PostgreSQL and Redis support
- **📊 Analytics Dashboard**: System performance monitoring and document statistics

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Document      │    │    Embedding     │    │   Retrieval     │
│   Processing    │───▶│   & Indexing     │───▶│   System        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Text          │    │   FAISS Vector   │    │   Hybrid        │
│   Cleaning      │    │   Database       │    │   BM25 + Vector │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                  │                        │
                                  ▼                        ▼
                        ┌──────────────────┐    ┌─────────────────┐
                        │   Metadata       │    │   Response      │
                        │   Storage        │    │   Generation    │
                        └──────────────────┘    └─────────────────┘
```

## 🛠️ Installation

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/multi-doc-rag/multi-doc-rag.git
cd multi-doc-rag

# Start with Docker Compose
docker-compose up --build
```

The application will be available at `http://localhost:8501`

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/multi-doc-rag/multi-doc-rag.git
cd multi-doc-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Tesseract OCR (for PDF processing)
# Ubuntu/Debian: sudo apt-get install tesseract-ocr
# macOS: brew install tesseract
# Windows: Download from https://github.com/tesseract-ocr/tesseract

# Run the application
streamlit run multi_doc_rag/ui/streamlit_app.py
```

### Option 3: Package Installation

```bash
pip install multi-doc-rag

# Run the application
rag-server
```

## 🚦 Quick Start

1. **Upload Documents**: Use the web interface to upload PDF, DOCX, HTML, TXT, or MD files
2. **Wait for Processing**: Documents are automatically processed, chunked, and embedded
3. **Ask Questions**: Use natural language to query across all your documents
4. **Get Cited Responses**: Receive comprehensive answers with source citations

### Python API Usage

```python
from multi_doc_rag import add_document, ask_question

# Add a document to the system
document_id = add_document("/path/to/your/document.pdf")

# Ask a question
response = ask_question("What are the main findings about climate change?")

print(response.text)
print(f"Sources used: {len(response.sources)}")
```

## 📁 Project Structure

```
multi-doc-rag/
├── multi_doc_rag/
│   ├── core/                 # Core system components
│   │   ├── document_processor.py
│   │   ├── embeddings.py
│   │   ├── retrieval.py
│   │   ├── generation.py
│   │   └── database.py
│   ├── utils/                # Utility modules
│   │   ├── text_processing.py
│   │   └── chunking.py
│   ├── config/               # Configuration
│   │   └── settings.py
│   └── ui/                   # User interfaces
│       ├── streamlit_app.py
│       └── fastapi_app.py
├── data/                     # Data storage
│   ├── documents/
│   ├── embeddings/
│   └── models/
├── tests/                    # Test suites
├── docker-compose.yml        # Docker deployment
├── Dockerfile
└── requirements.txt
```

## ⚙️ Configuration

The system can be configured through environment variables or the `settings.py` file:

```python
# Key configuration options
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence transformer model
LLM_PROVIDER = "gpt4all"              # Local LLM provider
CHUNK_SIZE = 1000                     # Document chunk size
VECTOR_SEARCH_K = 5                   # Number of results to retrieve
DATABASE_URL = "sqlite:///rag.db"     # Database connection
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=multi_doc_rag tests/

# Run specific test category
pytest tests/test_document_processor.py -v
```

## 📊 Performance

The system is optimized for both accuracy and performance:

- **Document Processing**: 10-50 documents per minute (depending on size)
- **Query Response Time**: 1-5 seconds for typical queries
- **Memory Usage**: ~2GB for 1000 documents with embeddings
- **Storage**: ~100MB per 1000 document pages

## 🔧 Advanced Features

### Custom Chunking Strategies

```python
from multi_doc_rag.utils.chunking import ChunkingManager

chunker = ChunkingManager()
chunks = chunker.chunk_document(
    text="Your document text...",
    document_id="doc_123",
    strategy="semantic",  # or "fixed", "recursive"
    chunk_size=1000,
    overlap=200
)
```

### Hybrid Search Configuration

```python
from multi_doc_rag.core.retrieval import HybridRetriever

retriever = HybridRetriever()
retriever.vector_weight = 0.7  # Adjust vector vs BM25 balance
retriever.bm25_weight = 0.3

results = retriever.search("your query", k=10)
```

### Custom Response Generation

```python
from multi_doc_rag.core.generation import ResponseGenerator

generator = ResponseGenerator()
response = generator.generate_response(
    query="What are the key findings?",
    retrieval_results=results,
    template_name="comparative_analysis",
    temperature=0.7
)
```

## 🚀 Deployment

### Production Deployment

1. **Environment Setup**:
```bash
export DATABASE_URL="postgresql://user:pass@localhost/ragdb"
export REDIS_URL="redis://localhost:6379"
export OPENAI_API_KEY="your-key-here"  # Optional for better LLM
```

2. **Scale with Docker Swarm**:
```bash
docker swarm init
docker stack deploy -c docker-compose.prod.yml rag-stack
```

3. **Kubernetes Deployment**:
```bash
kubectl apply -f k8s/
```

### Performance Tuning

- **GPU Acceleration**: Set `CUDA_AVAILABLE=true` for GPU-accelerated embeddings
- **Database Optimization**: Use PostgreSQL for production workloads
- **Caching**: Enable Redis for query result caching
- **Load Balancing**: Deploy multiple instances behind a load balancer

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Sentence Transformers** for excellent embedding models
- **FAISS** for high-performance vector search
- **GPT4All** for local LLM capabilities
- **Streamlit** for rapid web app development
- **LangChain** for inspiration on RAG architectures

## 📞 Support

- 📖 **Documentation**: [https://multi-doc-rag.readthedocs.io/](https://multi-doc-rag.readthedocs.io/)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/multi-doc-rag/multi-doc-rag/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/multi-doc-rag/multi-doc-rag/discussions)
- 📧 **Email**: support@multi-doc-rag.com

## 🔮 Roadmap

- [ ] **Multi-language Support**: Add support for non-English documents
- [ ] **Advanced Analytics**: More detailed performance and usage analytics
- [ ] **API Gateway**: RESTful API for programmatic access
- [ ] **Plugin System**: Extensible plugin architecture
- [ ] **Cloud Deployment**: One-click cloud deployment options
- [ ] **Mobile Interface**: Mobile-optimized web interface

---

**Built with ❤️ by the Multi-Document RAG Team**
