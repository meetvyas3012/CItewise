# Create the Streamlit UI application
streamlit_app_content = '''"""
Streamlit web application for the Multi-Document Research Assistant
"""
import streamlit as st
import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# Configure page
st.set_page_config(
    page_title="Multi-Document Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import our modules (with error handling for development)
try:
    from ..core.document_processor import document_processor
    from ..core.database import db_manager
    from ..core.embeddings import embedding_manager
    from ..core.retrieval import hybrid_retriever
    from ..core.generation import response_generator
    from ..config.settings import settings
except ImportError:
    st.error("Error importing modules. Make sure you're running from the correct directory.")
    st.stop()

class StreamlitApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        self.setup_logging()
        self.init_session_state()
    
    def setup_logging(self):
        """Setup logging for Streamlit app"""
        logging.basicConfig(
            level=getattr(logging, settings.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def init_session_state(self):
        """Initialize Streamlit session state"""
        if 'documents' not in st.session_state:
            st.session_state.documents = []
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        if 'current_response' not in st.session_state:
            st.session_state.current_response = None
    
    def run(self):
        """Main application entry point"""
        st.title("📚 Multi-Document Research Assistant")
        st.markdown("Ask questions across multiple documents with AI-powered retrieval and generation")
        
        # Sidebar
        self.render_sidebar()
        
        # Main interface
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Query", "📄 Documents", "📊 Analytics", "⚙️ Settings"])
        
        with tab1:
            self.render_query_interface()
        
        with tab2:
            self.render_document_management()
        
        with tab3:
            self.render_analytics()
        
        with tab4:
            self.render_settings()
    
    def render_sidebar(self):
        """Render sidebar with system information"""
        with st.sidebar:
            st.header("📋 System Status")
            
            # Database stats
            try:
                db_stats = db_manager.get_database_stats()
                st.metric("Documents", db_stats.get('document_count', 0))
                st.metric("Chunks", db_stats.get('chunk_count', 0))
                st.metric("Embeddings", db_stats.get('embedding_count', 0))
            except Exception as e:
                st.error(f"Error loading database stats: {e}")
            
            st.divider()
            
            # Quick actions
            st.header("⚡ Quick Actions")
            
            if st.button("🔄 Refresh System", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
            
            if st.button("🗑️ Clear Cache", use_container_width=True):
                st.cache_data.clear()
                st.success("Cache cleared!")
            
            # Model information
            st.divider()
            st.header("🤖 Model Info")
            
            try:
                embedding_stats = embedding_manager.get_embedding_stats()
                st.text(f"Embedding Model: {embedding_stats.get('model_name', 'Unknown')}")
                st.text(f"Dimension: {embedding_stats.get('dimension', 'Unknown')}")
                st.text(f"Device: {embedding_stats.get('device', 'Unknown')}")
                
                generation_stats = response_generator.get_generation_stats()
                st.text(f"LLM Provider: {generation_stats.get('provider', 'Unknown')}")
            except Exception as e:
                st.text(f"Error loading model info: {e}")
    
    def render_query_interface(self):
        """Render the main query interface"""
        st.header("🔍 Ask Your Question")
        
        # Query input
        query = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="What would you like to know from your documents?"
        )
        
        # Search parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_results = st.slider("Number of sources", 1, 20, 5)
        
        with col2:
            search_mode = st.selectbox(
                "Search Mode",
                ["hybrid", "vector_only", "bm25_only"],
                index=0
            )
        
        with col3:
            temperature = st.slider("Response Creativity", 0.0, 1.0, 0.7, 0.1)
        
        # Query button
        if st.button("🚀 Search & Generate", type="primary", use_container_width=True):
            if not query.strip():
                st.warning("Please enter a question!")
                return
            
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Retrieve relevant chunks
                status_text.text("🔍 Searching for relevant information...")
                progress_bar.progress(25)
                
                if search_mode == "vector_only":
                    # Use only vector search
                    retrieval_results = embedding_manager.search(query, num_results)
                elif search_mode == "bm25_only":
                    # Use only BM25 search
                    hybrid_retriever.initialize()
                    retrieval_results = hybrid_retriever.bm25_retriever.search(query, num_results)
                else:
                    # Use hybrid search
                    retrieval_results = hybrid_retriever.search(query, num_results)
                
                progress_bar.progress(50)
                
                if not retrieval_results:
                    st.warning("No relevant documents found. Try rephrasing your question or upload more documents.")
                    return
                
                # Step 2: Generate response
                status_text.text("🤖 Generating response...")
                progress_bar.progress(75)
                
                response = response_generator.generate_response(
                    query=query,
                    retrieval_results=retrieval_results,
                    temperature=temperature
                )
                
                progress_bar.progress(100)
                status_text.text("✅ Complete!")
                
                # Store in session state
                st.session_state.current_response = response
                st.session_state.query_history.append({
                    'query': query,
                    'timestamp': time.time(),
                    'response': response
                })
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                st.error(f"Error processing query: {e}")
                progress_bar.empty()
                status_text.empty()
                return
        
        # Display results
        if st.session_state.current_response:
            self.display_response(st.session_state.current_response)
    
    def display_response(self, response):
        """Display the generated response with sources"""
        st.divider()
        st.header("💡 Response")
        
        # Response text
        st.markdown(response.text)
        
        # Metadata
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Generation Time", f"{response.generation_time:.2f}s")
        
        with col2:
            if response.confidence_score:
                st.metric("Confidence", f"{response.confidence_score:.1%}")
        
        with col3:
            st.metric("Sources Used", len(response.sources))
        
        # Sources
        if response.sources:
            st.header("📚 Sources")
            
            for i, source in enumerate(response.sources):
                with st.expander(f"Source [{source['citation_number']}] - Preview"):
                    st.text(f"Document ID: {source.get('document_id', 'Unknown')}")
                    st.text(f"Chunk ID: {source.get('chunk_id', 'Unknown')}")
                    if source.get('score'):
                        st.text(f"Relevance Score: {source['score']:.3f}")
                    st.markdown(f"**Text Preview:**")
                    st.text(source['text_preview'])
    
    def render_document_management(self):
        """Render document management interface"""
        st.header("📄 Document Management")
        
        # Upload section
        st.subheader("📤 Upload Documents")
        
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
                    index=0
                )
            
            with col2:
                chunk_size = st.number_input(
                    "Chunk Size",
                    min_value=100,
                    max_value=2000,
                    value=1000,
                    step=100
                )
            
            if st.button("📤 Process & Upload", type="primary"):
                self.process_uploaded_files(uploaded_files, chunking_strategy, chunk_size)
        
        st.divider()
        
        # Document list
        st.subheader("📋 Current Documents")
        
        try:
            documents = db_manager.list_documents()
            
            if documents:
                # Create DataFrame for better display
                df = pd.DataFrame(documents)
                df['created_at'] = pd.to_datetime(df['created_at'])
                df['file_size_mb'] = (df['file_size'] / 1024 / 1024).round(2)
                
                # Display table
                st.dataframe(
                    df[['filename', 'file_type', 'file_size_mb', 'chunk_count', 'created_at']],
                    use_container_width=True,
                    column_config={
                        'filename': 'File Name',
                        'file_type': 'Type',
                        'file_size_mb': st.column_config.NumberColumn('Size (MB)', format="%.2f"),
                        'chunk_count': 'Chunks',
                        'created_at': st.column_config.DatetimeColumn('Upload Date')
                    }
                )
                
                # Document actions
                st.subheader("🔧 Document Actions")
                
                doc_to_delete = st.selectbox(
                    "Select document to delete:",
                    [""] + [doc['filename'] for doc in documents],
                    index=0
                )
                
                if doc_to_delete and st.button("🗑️ Delete Document", type="secondary"):
                    doc_id = next(doc['id'] for doc in documents if doc['filename'] == doc_to_delete)
                    try:
                        db_manager.delete_document(doc_id)
                        st.success(f"Deleted {doc_to_delete}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting document: {e}")
                
            else:
                st.info("No documents uploaded yet. Upload some documents to get started!")
                
        except Exception as e:
            st.error(f"Error loading documents: {e}")
    
    def process_uploaded_files(self, uploaded_files, chunking_strategy, chunk_size):
        """Process and save uploaded files"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_files = len(uploaded_files)
        processed_count = 0
        
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                status_text.text(f"Processing {uploaded_file.name}...")
                
                # Save uploaded file temporarily
                temp_path = settings.data_dir / "temp" / uploaded_file.name
                temp_path.parent.mkdir(exist_ok=True)
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process the document
                processed_doc = document_processor.process_document(
                    temp_path,
                    chunking_strategy=chunking_strategy,
                    chunk_size=chunk_size,
                    overlap=200
                )
                
                # Save to database
                document_id = db_manager.save_document(processed_doc)
                
                # Generate embeddings
                embedding_manager.embed_document_chunks(document_id)
                
                # Clean up temp file
                temp_path.unlink()
                
                processed_count += 1
                progress_bar.progress((i + 1) / total_files)
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        if processed_count > 0:
            st.success(f"Successfully processed {processed_count} out of {total_files} files!")
            
            # Reinitialize retrieval system with new documents
            hybrid_retriever.initialize()
            
            # Save embeddings index
            embedding_manager.save_index()
            
            st.rerun()
        else:
            st.error("No files were processed successfully.")
    
    def render_analytics(self):
        """Render analytics and statistics"""
        st.header("📊 System Analytics")
        
        try:
            # Database statistics
            db_stats = db_manager.get_database_stats()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Documents", db_stats.get('document_count', 0))
            
            with col2:
                st.metric("Total Chunks", db_stats.get('chunk_count', 0))
            
            with col3:
                st.metric("Total Embeddings", db_stats.get('embedding_count', 0))
            
            with col4:
                missing_embeddings = db_stats.get('chunks_without_embeddings', 0)
                st.metric("Missing Embeddings", missing_embeddings)
            
            # File type distribution
            if db_stats.get('file_types'):
                st.subheader("📁 File Type Distribution")
                
                file_type_df = pd.DataFrame(
                    list(db_stats['file_types'].items()),
                    columns=['File Type', 'Count']
                )
                
                st.bar_chart(file_type_df.set_index('File Type'))
            
            # Query history
            if st.session_state.query_history:
                st.subheader("🕒 Recent Queries")
                
                recent_queries = st.session_state.query_history[-10:]  # Last 10 queries
                
                for i, query_data in enumerate(reversed(recent_queries)):
                    with st.expander(f"Query {len(recent_queries) - i}: {query_data['query'][:50]}..."):
                        st.text(f"Timestamp: {time.ctime(query_data['timestamp'])}")
                        st.text(f"Generation Time: {query_data['response'].generation_time:.2f}s")
                        st.text(f"Sources Used: {len(query_data['response'].sources)}")
                        if query_data['response'].confidence_score:
                            st.text(f"Confidence: {query_data['response'].confidence_score:.1%}")
            
            # System performance
            st.subheader("⚡ System Performance")
            
            embedding_stats = embedding_manager.get_embedding_stats()
            retrieval_stats = hybrid_retriever.get_retrieval_stats()
            
            perf_col1, perf_col2 = st.columns(2)
            
            with perf_col1:
                st.text("Embedding System:")
                st.text(f"  Model: {embedding_stats.get('model_name', 'Unknown')}")
                st.text(f"  Dimension: {embedding_stats.get('dimension', 'Unknown')}")
                st.text(f"  Device: {embedding_stats.get('device', 'Unknown')}")
                st.text(f"  Index Size: {embedding_stats.get('index_size_mb', 0):.1f} MB")
            
            with perf_col2:
                st.text("Retrieval System:")
                st.text(f"  Hybrid Search: {retrieval_stats.get('hybrid_search_enabled', False)}")
                st.text(f"  Vector Weight: {retrieval_stats.get('vector_weight', 0)}")
                st.text(f"  BM25 Weight: {retrieval_stats.get('bm25_weight', 0)}")
                st.text(f"  Default K: {retrieval_stats.get('default_k', 0)}")
            
        except Exception as e:
            st.error(f"Error loading analytics: {e}")
    
    def render_settings(self):
        """Render system settings"""
        st.header("⚙️ System Settings")
        
        # Model settings
        st.subheader("🤖 Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.text("Embedding Model:")
            st.code(settings.embedding.model_name)
            st.text("LLM Provider:")
            st.code(settings.llm.provider)
        
        with col2:
            st.text("Vector Database:")
            st.code(f"{settings.vector_db.index_type} index")
            st.text("Chunking Strategy:")
            st.code(settings.chunking.strategy)
        
        # Configuration display
        st.subheader("🔧 Current Configuration")
        
        config_data = {
            "Embedding": {
                "Model": settings.embedding.model_name,
                "Dimension": settings.embedding.dimension,
                "Batch Size": settings.embedding.batch_size,
                "Device": settings.embedding.device
            },
            "Retrieval": {
                "Top K": settings.retrieval.top_k,
                "Vector Weight": settings.retrieval.vector_weight,
                "BM25 Weight": settings.retrieval.bm25_weight,
                "Hybrid Search": settings.retrieval.hybrid_search
            },
            "Generation": {
                "Provider": settings.llm.provider,
                "Max Tokens": settings.llm.max_tokens,
                "Temperature": settings.llm.temperature
            }
        }
        
        for category, values in config_data.items():
            with st.expander(f"📋 {category} Settings"):
                for key, value in values.items():
                    st.text(f"{key}: {value}")
        
        # System actions
        st.subheader("🔧 System Actions")
        
        action_col1, action_col2 = st.columns(2)
        
        with action_col1:
            if st.button("🔄 Rebuild Embeddings Index", use_container_width=True):
                try:
                    with st.spinner("Rebuilding embeddings index..."):
                        embedding_manager.rebuild_index()
                    st.success("Embeddings index rebuilt successfully!")
                except Exception as e:
                    st.error(f"Error rebuilding index: {e}")
        
        with action_col2:
            if st.button("💾 Save Index to Disk", use_container_width=True):
                try:
                    with st.spinner("Saving index..."):
                        embedding_manager.save_index()
                    st.success("Index saved successfully!")
                except Exception as e:
                    st.error(f"Error saving index: {e}")

# Main application runner
def main():
    """Run the Streamlit application"""
    app = StreamlitApp()
    app.run()

if __name__ == "__main__":
    main()
'''

with open('multi_doc_rag/ui/streamlit_app.py', 'w') as f:
    f.write(streamlit_app_content)

print("✅ Streamlit application created")