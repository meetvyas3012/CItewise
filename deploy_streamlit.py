"""
Streamlined Streamlit deployment for Multi-Document Research Assistant
"""
import streamlit as st
import os
import sys
import logging
from pathlib import Path

# Configure Streamlit page
st.set_page_config(
    page_title="Multi-Document Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def initialize_system():
    """Initialize the RAG system with caching"""
    try:
        # Import system components
        from multi_doc_rag.core.document_processor import document_processor
        from multi_doc_rag.core.database import db_manager
        from multi_doc_rag.core.embeddings import embedding_manager
        from multi_doc_rag.core.retrieval import hybrid_retriever
        from multi_doc_rag.core.generation import response_generator
        from multi_doc_rag.config.settings import settings

        # Initialize retrieval system
        hybrid_retriever.initialize()

        return {
            'document_processor': document_processor,
            'db_manager': db_manager,
            'embedding_manager': embedding_manager,
            'hybrid_retriever': hybrid_retriever,
            'response_generator': response_generator,
            'settings': settings
        }
    except Exception as e:
        st.error(f"Error initializing system: {e}")
        st.info("This might be due to missing dependencies. Please install requirements.txt")
        return None

def main():
    """Main Streamlit application"""

    st.title("📚 Multi-Document Research Assistant")
    st.markdown("Ask questions across multiple documents with AI-powered analysis")

    # Initialize system
    system = initialize_system()

    if not system:
        st.error("System initialization failed. Please check the logs.")
        st.stop()

    # Sidebar for system info
    with st.sidebar:
        st.header("System Status")

        try:
            db_stats = system['db_manager'].get_database_stats()
            st.metric("Documents", db_stats.get('document_count', 0))
            st.metric("Chunks", db_stats.get('chunk_count', 0))
            st.metric("Embeddings", db_stats.get('embedding_count', 0))
        except Exception as e:
            st.warning(f"Could not load stats: {e}")

        st.divider()

        if st.button("🔄 Refresh System"):
            st.cache_resource.clear()
            st.rerun()

    # Main tabs
    tab1, tab2 = st.tabs(["🔍 Query Documents", "📤 Upload Documents"])

    with tab1:
        render_query_tab(system)

    with tab2:
        render_upload_tab(system)

def render_query_tab(system):
    """Render the document query interface"""

    st.header("Ask Your Question")

    # Query input
    query = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="What would you like to know from your documents?"
    )

    # Search settings
    col1, col2 = st.columns(2)
    with col1:
        num_sources = st.slider("Number of sources", 1, 10, 5)
    with col2:
        temperature = st.slider("Response creativity", 0.0, 1.0, 0.7, 0.1)

    # Search button
    if st.button("🚀 Search & Answer", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a question!")
            return

        # Check if documents exist
        try:
            documents = system['db_manager'].list_documents()
            if not documents:
                st.warning("No documents found! Please upload some documents first.")
                return
        except Exception as e:
            st.error(f"Error checking documents: {e}")
            return

        # Process query
        with st.spinner("🔍 Searching for relevant information..."):
            try:
                # Get relevant chunks
                retrieval_results = system['hybrid_retriever'].search(query, num_sources)

                if not retrieval_results:
                    st.warning("No relevant information found. Try rephrasing your question.")
                    return

                # Generate response
                response = system['response_generator'].generate_response(
                    query=query,
                    retrieval_results=retrieval_results,
                    temperature=temperature
                )

                # Display results
                display_response(response)

            except Exception as e:
                st.error(f"Error processing query: {e}")
                logger.error(f"Query processing error: {e}", exc_info=True)

def render_upload_tab(system):
    """Render the document upload interface"""

    st.header("Upload Documents")

    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        accept_multiple_files=True,
        type=['pdf', 'txt', 'docx', 'html', 'md'],
        help="Supported formats: PDF, TXT, DOCX, HTML, MD"
    )

    if uploaded_files:
        col1, col2 = st.columns(2)

        with col1:
            chunking_strategy = st.selectbox(
                "Chunking Strategy",
                ["recursive", "fixed", "semantic"],
                index=0,
                help="Recursive: Respects document structure, Fixed: Equal sized chunks, Semantic: Topic-based chunks"
            )

        with col2:
            chunk_size = st.number_input(
                "Chunk Size (characters)",
                min_value=100,
                max_value=2000,
                value=1000,
                step=100
            )

        if st.button("📤 Process & Upload Documents", type="primary"):
            process_uploads(system, uploaded_files, chunking_strategy, chunk_size)

    # Show current documents
    st.divider()
    st.subheader("Current Documents")

    try:
        documents = system['db_manager'].list_documents()
        if documents:
            for doc in documents:
                with st.expander(f"📄 {doc['filename']} ({doc['chunk_count']} chunks)"):
                    st.write(f"**Type:** {doc['file_type']}")
                    st.write(f"**Size:** {doc['file_size'] / 1024:.1f} KB")
                    st.write(f"**Uploaded:** {doc['created_at']}")

                    if st.button(f"🗑️ Delete {doc['filename']}", key=f"delete_{doc['id']}"):
                        try:
                            system['db_manager'].delete_document(doc['id'])
                            st.success(f"Deleted {doc['filename']}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting document: {e}")
        else:
            st.info("No documents uploaded yet. Upload some documents to get started!")

    except Exception as e:
        st.error(f"Error loading documents: {e}")

def process_uploads(system, uploaded_files, chunking_strategy, chunk_size):
    """Process uploaded files"""

    progress_bar = st.progress(0)
    status_text = st.empty()

    processed_count = 0
    total_files = len(uploaded_files)

    for i, uploaded_file in enumerate(uploaded_files):
        try:
            status_text.text(f"Processing {uploaded_file.name}...")

            # Create temp directory
            temp_dir = Path("data/temp")
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Save uploaded file
            temp_path = temp_dir / uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Process document
            processed_doc = system['document_processor'].process_document(
                temp_path,
                chunking_strategy=chunking_strategy,
                chunk_size=chunk_size,
                overlap=200
            )

            # Save to database
            document_id = system['db_manager'].save_document(processed_doc)

            # Generate embeddings
            embedding_count = system['embedding_manager'].embed_document_chunks(document_id)

            # Clean up temp file
            temp_path.unlink()

            processed_count += 1
            progress_bar.progress((i + 1) / total_files)

            st.success(f"✅ Processed {uploaded_file.name} - {embedding_count} chunks embedded")

        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {e}")
            logger.error(f"Upload processing error: {e}", exc_info=True)
            continue

    # Clean up progress indicators
    progress_bar.empty()
    status_text.empty()

    if processed_count > 0:
        # Reinitialize retrieval system
        try:
            system['hybrid_retriever'].initialize()
            system['embedding_manager'].save_index()
        except Exception as e:
            st.warning(f"System reinitialization warning: {e}")

        st.success(f"🎉 Successfully processed {processed_count} out of {total_files} files!")

        # Clear cache and rerun
        st.cache_resource.clear()
        st.rerun()
    else:
        st.error("No files were processed successfully.")

def display_response(response):
    """Display the generated response"""

    st.divider()
    st.header("💡 Response")

    # Main response
    st.markdown(response.text)

    # Metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("⏱️ Generation Time", f"{response.generation_time:.2f}s")

    with col2:
        if response.confidence_score:
            st.metric("🎯 Confidence", f"{response.confidence_score:.0%}")
        else:
            st.metric("🎯 Confidence", "N/A")

    with col3:
        st.metric("📚 Sources", len(response.sources))

    # Sources
    if response.sources:
        st.header("📖 Sources")

        for source in response.sources:
            with st.expander(f"Source [{source['citation_number']}] - Click to view"):
                st.write(f"**Document ID:** {source.get('document_id', 'Unknown')}")
                if source.get('score'):
                    st.write(f"**Relevance Score:** {source['score']:.3f}")
                st.write("**Text Preview:**")
                st.text(source['text_preview'])

if __name__ == "__main__":
    main()
