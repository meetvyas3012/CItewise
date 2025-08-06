# Create sample test files
test_document_processor_content = '''"""
Tests for document processor module
"""
import pytest
import tempfile
from pathlib import Path
from multi_doc_rag.core.document_processor import document_processor, DocumentProcessor

class TestDocumentProcessor:
    """Test cases for DocumentProcessor"""
    
    def test_process_text_file(self):
        """Test processing a simple text file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document.\\nIt has multiple lines.\\nFor testing purposes.")
            temp_path = f.name
        
        try:
            processed_doc = document_processor.process_document(temp_path)
            
            assert processed_doc.content is not None
            assert len(processed_doc.chunks) > 0
            assert processed_doc.metadata.filename.endswith('.txt')
            assert processed_doc.metadata.file_type == '.txt'
            
        finally:
            Path(temp_path).unlink()
    
    def test_supported_formats(self):
        """Test that processor reports correct supported formats"""
        processor = DocumentProcessor()
        supported = processor.supported_formats
        
        assert '.pdf' in supported
        assert '.txt' in supported
        assert '.docx' in supported
        assert '.html' in supported
    
    def test_metadata_extraction(self):
        """Test metadata extraction"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            temp_path = f.name
        
        try:
            processor = DocumentProcessor()
            metadata = processor._extract_metadata(Path(temp_path))
            
            assert metadata.filename is not None
            assert metadata.file_size > 0
            assert metadata.hash_md5 is not None
            assert len(metadata.hash_md5) == 32  # MD5 hash length
            
        finally:
            Path(temp_path).unlink()
    
    def test_batch_processing(self):
        """Test batch processing multiple files"""
        temp_dir = tempfile.mkdtemp()
        temp_dir_path = Path(temp_dir)
        
        # Create test files
        for i in range(3):
            with open(temp_dir_path / f"test_{i}.txt", 'w') as f:
                f.write(f"Test document {i} content.")
        
        try:
            processed_docs = document_processor.batch_process(temp_dir_path)
            
            assert len(processed_docs) == 3
            for doc in processed_docs:
                assert doc.content is not None
                assert len(doc.chunks) > 0
                
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    pytest.main([__file__])
'''

with open('tests/test_document_processor.py', 'w') as f:
    f.write(test_document_processor_content)

# Create CLI module
cli_content = '''"""
Command-line interface for Multi-Document Research Assistant
"""
import click
import sys
from pathlib import Path
from multi_doc_rag import quick_start, add_document, ask_question, settings

@click.group()
@click.version_option(version="1.0.0")
def main():
    """Multi-Document Research Assistant CLI"""
    pass

@main.command()
def init():
    """Initialize the RAG system"""
    click.echo("Initializing Multi-Document Research Assistant...")
    
    if quick_start():
        click.echo("✅ System initialized successfully!")
    else:
        click.echo("❌ System initialization failed!")
        sys.exit(1)

@main.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--strategy', default='recursive', help='Chunking strategy')
@click.option('--chunk-size', default=1000, help='Chunk size')
def add(file_path, strategy, chunk_size):
    """Add a document to the system"""
    try:
        click.echo(f"Processing document: {file_path}")
        document_id = add_document(
            file_path, 
            chunking_strategy=strategy,
            chunk_size=chunk_size
        )
        click.echo(f"✅ Document added successfully! ID: {document_id}")
    except Exception as e:
        click.echo(f"❌ Error adding document: {e}")
        sys.exit(1)

@main.command()
@click.argument('query')
@click.option('--k', default=5, help='Number of sources to retrieve')
def ask(query, k):
    """Ask a question"""
    try:
        click.echo(f"Query: {query}")
        click.echo("🔍 Searching for relevant information...")
        
        response = ask_question(query, k)
        
        click.echo("\\n💡 Response:")
        click.echo("-" * 50)
        click.echo(response.text)
        
        if response.sources:
            click.echo(f"\\n📚 Sources ({len(response.sources)}):")
            for i, source in enumerate(response.sources, 1):
                click.echo(f"[{i}] {source['text_preview'][:100]}...")
        
        click.echo(f"\\n⏱️  Generation time: {response.generation_time:.2f}s")
        if response.confidence_score:
            click.echo(f"🎯 Confidence: {response.confidence_score:.1%}")
            
    except Exception as e:
        click.echo(f"❌ Error processing query: {e}")
        sys.exit(1)

@main.command()
def server():
    """Start the web server"""
    try:
        import streamlit.web.cli as stcli
        import sys
        
        # Get the path to streamlit app
        app_path = Path(__file__).parent / "ui" / "streamlit_app.py"
        
        # Set up streamlit arguments
        sys.argv = [
            "streamlit",
            "run",
            str(app_path),
            "--server.port=8501",
            "--server.address=0.0.0.0"
        ]
        
        # Run streamlit
        stcli.main()
        
    except ImportError:
        click.echo("❌ Streamlit not installed. Install with: pip install streamlit")
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error starting server: {e}")
        sys.exit(1)

@main.command()
def status():
    """Show system status"""
    try:
        from multi_doc_rag.core.database import db_manager
        from multi_doc_rag.core.embeddings import embedding_manager
        
        click.echo("📊 System Status")
        click.echo("=" * 30)
        
        # Database stats
        db_stats = db_manager.get_database_stats()
        click.echo(f"Documents: {db_stats.get('document_count', 0)}")
        click.echo(f"Chunks: {db_stats.get('chunk_count', 0)}")
        click.echo(f"Embeddings: {db_stats.get('embedding_count', 0)}")
        
        # Embedding stats  
        emb_stats = embedding_manager.get_embedding_stats()
        click.echo(f"Model: {emb_stats.get('model_name', 'Unknown')}")
        click.echo(f"Device: {emb_stats.get('device', 'Unknown')}")
        
        # Configuration
        click.echo(f"\\nConfiguration:")
        click.echo(f"Chunk size: {settings.chunking.chunk_size}")
        click.echo(f"LLM provider: {settings.llm.provider}")
        
    except Exception as e:
        click.echo(f"❌ Error getting status: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

# Create CLI directory in multi_doc_rag
with open('multi_doc_rag/cli.py', 'w') as f:
    f.write(cli_content)

print("✅ CLI and test files created")