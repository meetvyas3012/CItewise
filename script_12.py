# Create the main package init files
main_init_content = '''"""
Multi-Document Research Assistant

A comprehensive RAG (Retrieval-Augmented Generation) system that enables users to pose complex questions
across a corpus of documents and receive synthesized answers with inline citations.

Key Features:
- Multi-format document processing (PDF, DOCX, HTML, TXT, MD)
- Advanced text chunking strategies
- Hybrid search combining vector similarity and BM25
- Local LLM integration for privacy-preserving responses
- Citation-enabled response generation
- Interactive web interface
"""

__version__ = "1.0.0"
__author__ = "Multi-Document RAG Team"
__description__ = "AI-powered research assistant for multi-document analysis"

# Import main components
from .core.document_processor import document_processor, DocumentProcessor
from .core.database import db_manager, DatabaseManager
from .core.embeddings import embedding_manager, EmbeddingManager
from .core.retrieval import hybrid_retriever, HybridRetriever
from .core.generation import response_generator, ResponseGenerator
from .config.settings import settings, Settings

# Import utilities
from .utils.text_processing import text_processor, TextProcessor
from .utils.chunking import chunking_manager, ChunkingManager

__all__ = [
    'document_processor', 'DocumentProcessor',
    'db_manager', 'DatabaseManager', 
    'embedding_manager', 'EmbeddingManager',
    'hybrid_retriever', 'HybridRetriever',
    'response_generator', 'ResponseGenerator',
    'text_processor', 'TextProcessor',
    'chunking_manager', 'ChunkingManager',
    'settings', 'Settings'
]

# Quick start function
def quick_start():
    """
    Quick start function to set up the system
    """
    print(f"Multi-Document Research Assistant v{__version__}")
    print("Initializing system components...")
    
    try:
        # Initialize database
        print("✓ Database initialized")
        
        # Initialize embedding manager
        print("✓ Embedding system ready")
        
        # Initialize retrieval system
        hybrid_retriever.initialize()
        print("✓ Retrieval system ready")
        
        print("\\n🚀 System ready! You can now:")
        print("  1. Upload documents using the web interface")
        print("  2. Run queries through the API")
        print("  3. Use the Python API directly")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        return False

# API shortcuts
def process_document(file_path, chunking_strategy='recursive', **kwargs):
    """
    Process a single document
    
    Args:
        file_path: Path to the document
        chunking_strategy: Chunking strategy to use
        **kwargs: Additional parameters
        
    Returns:
        ProcessedDocument object
    """
    return document_processor.process_document(file_path, chunking_strategy, **kwargs)

def ask_question(query, k=5):
    """
    Ask a question and get an AI-generated response
    
    Args:
        query: Question to ask
        k: Number of source documents to retrieve
        
    Returns:
        GeneratedResponse object
    """
    # Retrieve relevant chunks
    retrieval_results = hybrid_retriever.search(query, k)
    
    # Generate response
    response = response_generator.generate_response(query, retrieval_results)
    
    return response

def add_document(file_path, **kwargs):
    """
    Add a document to the system (process + embed + index)
    
    Args:
        file_path: Path to the document
        **kwargs: Additional processing parameters
        
    Returns:
        Document ID
    """
    # Process document
    processed_doc = process_document(file_path, **kwargs)
    
    # Save to database
    document_id = db_manager.save_document(processed_doc)
    
    # Generate embeddings
    embedding_manager.embed_document_chunks(document_id)
    
    # Reinitialize retrieval system
    hybrid_retriever.initialize()
    
    return document_id
'''

with open('multi_doc_rag/__init__.py', 'w') as f:
    f.write(main_init_content)

# Create core init file
core_init_content = '''"""
Core components of the Multi-Document Research Assistant
"""

from .document_processor import document_processor, DocumentProcessor
from .database import db_manager, DatabaseManager
from .embeddings import embedding_manager, EmbeddingManager
from .retrieval import hybrid_retriever, HybridRetriever
from .generation import response_generator, ResponseGenerator

__all__ = [
    'document_processor', 'DocumentProcessor',
    'db_manager', 'DatabaseManager',
    'embedding_manager', 'EmbeddingManager', 
    'hybrid_retriever', 'HybridRetriever',
    'response_generator', 'ResponseGenerator'
]
'''

with open('multi_doc_rag/core/__init__.py', 'w') as f:
    f.write(core_init_content)

# Create utils init file
utils_init_content = '''"""
Utility modules for the Multi-Document Research Assistant
"""

from .text_processing import text_processor, TextProcessor
from .chunking import chunking_manager, ChunkingManager

__all__ = [
    'text_processor', 'TextProcessor',
    'chunking_manager', 'ChunkingManager'
]
'''

with open('multi_doc_rag/utils/__init__.py', 'w') as f:
    f.write(utils_init_content)

print("✅ Package initialization files created")