"""
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

        click.echo("\n💡 Response:")
        click.echo("-" * 50)
        click.echo(response.text)

        if response.sources:
            click.echo(f"\n📚 Sources ({len(response.sources)}):")
            for i, source in enumerate(response.sources, 1):
                click.echo(f"[{i}] {source['text_preview'][:100]}...")

        click.echo(f"\n⏱️  Generation time: {response.generation_time:.2f}s")
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
        click.echo(f"\nConfiguration:")
        click.echo(f"Chunk size: {settings.chunking.chunk_size}")
        click.echo(f"LLM provider: {settings.llm.provider}")

    except Exception as e:
        click.echo(f"❌ Error getting status: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
