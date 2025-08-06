# Create the configuration settings
settings_content = '''"""
Configuration settings for the Multi-Document Research Assistant
"""
import os
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, List
import json

class DatabaseConfig(BaseModel):
    """Database configuration"""
    url: str = "sqlite:///./data/rag_database.db"
    echo: bool = False

class EmbeddingConfig(BaseModel):
    """Embedding model configuration"""
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32
    device: str = "cpu"  # or "cuda" if GPU available

class VectorDBConfig(BaseModel):
    """Vector database configuration"""
    index_type: str = "flat"  # flat, ivf, or hnsw
    metric: str = "l2"  # l2, ip (inner product), or cosine
    nlist: int = 100  # for IVF index
    nprobe: int = 10  # for IVF search

class LLMConfig(BaseModel):
    """Large Language Model configuration"""
    provider: str = "gpt4all"  # gpt4all, llama-cpp, or openai
    model_path: str = "./data/models/ggml-gpt4all-j-v1.3-groovy.bin"
    max_tokens: int = 512
    temperature: float = 0.7
    context_length: int = 2048

class ChunkingConfig(BaseModel):
    """Document chunking configuration"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    strategy: str = "recursive"  # fixed, recursive, or semantic

class RetrievalConfig(BaseModel):
    """Retrieval configuration"""
    top_k: int = 5
    hybrid_search: bool = True
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    rerank: bool = True

class UIConfig(BaseModel):
    """User Interface configuration"""
    title: str = "Multi-Document Research Assistant"
    theme: str = "light"
    max_file_size: int = 50  # MB
    allowed_extensions: List[str] = [".pdf", ".txt", ".docx", ".html", ".md"]

class Settings(BaseModel):
    """Main settings class"""
    # Paths
    data_dir: Path = Path("./data")
    documents_dir: Path = Path("./data/documents")
    embeddings_dir: Path = Path("./data/embeddings")
    models_dir: Path = Path("./data/models")
    logs_dir: Path = Path("./logs")
    
    # Component configurations
    database: DatabaseConfig = DatabaseConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    vector_db: VectorDBConfig = VectorDBConfig()
    llm: LLMConfig = LLMConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    ui: UIConfig = UIConfig()
    
    # Environment variables
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        for directory in [
            self.data_dir,
            self.documents_dir, 
            self.embeddings_dir,
            self.models_dir,
            self.logs_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def save_to_file(self, file_path: str):
        """Save settings to JSON file"""
        with open(file_path, 'w') as f:
            json.dump(self.dict(), f, indent=2, default=str)
    
    @classmethod
    def load_from_file(cls, file_path: str):
        """Load settings from JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls(**data)

# Global settings instance
settings = Settings()

# Environment-specific overrides
if os.getenv("ENVIRONMENT") == "production":
    settings.database.url = os.getenv("DATABASE_URL", settings.database.url)
    settings.llm.provider = "openai"  # Use OpenAI in production
    settings.embedding.device = "cuda" if os.getenv("CUDA_AVAILABLE") else "cpu"
'''

with open('multi_doc_rag/config/settings.py', 'w') as f:
    f.write(settings_content)

print("✅ Configuration settings created")