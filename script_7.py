# Create the embeddings module
embeddings_content = '''"""
Text embeddings generation and management using Sentence Transformers
"""
import os
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pickle
import json
from datetime import datetime

# Core ML libraries
try:
    from sentence_transformers import SentenceTransformer
    import torch
    import faiss
except ImportError as e:
    logging.warning(f"ML libraries not available: {e}")

from ..config.settings import settings
from ..core.database import db_manager

class EmbeddingManager:
    """Manages text embeddings and vector operations"""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or settings.embedding.model_name
        self.device = device or settings.embedding.device
        self.batch_size = settings.embedding.batch_size
        
        # Initialize model
        self.model = None
        self.dimension = None
        
        # Vector index
        self.index = None
        self.chunk_id_to_index = {}  # Maps chunk_id to index position
        self.index_to_chunk_id = {}  # Maps index position to chunk_id
        
        # Setup logging
        self.setup_logging()
        
        # Load model and index
        self.load_model()
        self.load_or_create_index()
    
    def setup_logging(self):
        """Setup logging for embeddings"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, settings.log_level))
    
    def load_model(self):
        """Load the sentence transformer model"""
        try:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.logger.info(f"Model loaded successfully. Dimension: {self.dimension}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def encode_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode texts to embeddings
        
        Args:
            texts: List of texts to encode
            show_progress: Show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True  # L2 normalization
            )
            
            self.logger.info(f"Encoded {len(texts)} texts to embeddings")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error encoding texts: {e}")
            raise
    
    def encode_single_text(self, text: str) -> np.ndarray:
        """Encode a single text to embedding"""
        return self.encode_texts([text], show_progress=False)[0]
    
    def load_or_create_index(self):
        """Load existing index or create new one"""
        index_path = settings.embeddings_dir / "faiss_index.bin"
        metadata_path = settings.embeddings_dir / "index_metadata.json"
        
        if index_path.exists() and metadata_path.exists():
            try:
                self.load_index(index_path, metadata_path)
            except Exception as e:
                self.logger.warning(f"Error loading existing index: {e}")
                self.create_new_index()
        else:
            self.create_new_index()
    
    def create_new_index(self):
        """Create a new FAISS index"""
        if self.dimension is None:
            # Initialize model if not loaded
            self.load_model()
        
        # Create index based on configuration
        if settings.vector_db.index_type == "flat":
            if settings.vector_db.metric == "l2":
                self.index = faiss.IndexFlatL2(self.dimension)
            elif settings.vector_db.metric == "ip":
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                raise ValueError(f"Unsupported metric: {settings.vector_db.metric}")
                
        elif settings.vector_db.index_type == "ivf":
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer, 
                self.dimension, 
                settings.vector_db.nlist,
                faiss.METRIC_L2
            )
        else:
            raise ValueError(f"Unsupported index type: {settings.vector_db.index_type}")
        
        self.chunk_id_to_index = {}
        self.index_to_chunk_id = {}
        
        self.logger.info(f"Created new {settings.vector_db.index_type} index")
    
    def add_embeddings(self, embeddings: np.ndarray, chunk_ids: List[str]):
        """
        Add embeddings to the index
        
        Args:
            embeddings: Numpy array of embeddings
            chunk_ids: Corresponding chunk IDs
        """
        if len(embeddings) != len(chunk_ids):
            raise ValueError("Number of embeddings must match number of chunk IDs")
        
        # Ensure embeddings are float32
        embeddings = embeddings.astype('float32')
        
        # Get starting index
        start_idx = self.index.ntotal
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Update mapping
        for i, chunk_id in enumerate(chunk_ids):
            idx = start_idx + i
            self.chunk_id_to_index[chunk_id] = idx
            self.index_to_chunk_id[idx] = chunk_id
        
        self.logger.info(f"Added {len(embeddings)} embeddings to index")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of search results with scores and chunk information
        """
        if self.index.ntotal == 0:
            return []
        
        # Encode query
        query_embedding = self.encode_single_text(query).reshape(1, -1).astype('float32')
        
        # Configure search parameters for IVF index
        if settings.vector_db.index_type == "ivf":
            self.index.nprobe = settings.vector_db.nprobe
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # No more results
                break
                
            chunk_id = self.index_to_chunk_id.get(idx)
            if chunk_id:
                # Get chunk information from database
                chunk_info = db_manager.get_chunk(chunk_id)
                if chunk_info:
                    results.append({
                        'chunk_id': chunk_id,
                        'score': float(score),
                        'text': chunk_info['text'],
                        'document_id': chunk_info['document_id'],
                        'metadata': chunk_info['metadata']
                    })
        
        return results
    
    def search_by_embedding(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search using pre-computed embedding"""
        if self.index.ntotal == 0:
            return []
        
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        if settings.vector_db.index_type == "ivf":
            self.index.nprobe = settings.vector_db.nprobe
        
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                break
                
            chunk_id = self.index_to_chunk_id.get(idx)
            if chunk_id:
                chunk_info = db_manager.get_chunk(chunk_id)
                if chunk_info:
                    results.append({
                        'chunk_id': chunk_id,
                        'score': float(score),
                        'text': chunk_info['text'],
                        'document_id': chunk_info['document_id'],
                        'metadata': chunk_info['metadata']
                    })
        
        return results
    
    def embed_document_chunks(self, document_id: str) -> int:
        """
        Generate embeddings for all chunks of a document
        
        Args:
            document_id: Document ID
            
        Returns:
            Number of chunks processed
        """
        # Get chunks for this document
        chunks = db_manager.get_document_chunks(document_id)
        
        if not chunks:
            return 0
        
        # Filter chunks that don't have embeddings
        chunks_to_embed = [chunk for chunk in chunks if not chunk.get('embedding_id')]
        
        if not chunks_to_embed:
            self.logger.info(f"All chunks for document {document_id} already have embeddings")
            return 0
        
        # Extract texts and IDs
        texts = [chunk['text'] for chunk in chunks_to_embed]
        chunk_ids = [chunk['id'] for chunk in chunks_to_embed]
        
        # Generate embeddings
        embeddings = self.encode_texts(texts)
        
        # Add to index
        self.add_embeddings(embeddings, chunk_ids)
        
        # Save embedding metadata to database
        for i, chunk_id in enumerate(chunk_ids):
            embedding_id = f"emb_{chunk_id}_{int(datetime.now().timestamp())}"
            vector_file_path = str(settings.embeddings_dir / f"{embedding_id}.npy")
            
            # Save individual embedding vector
            np.save(vector_file_path, embeddings[i])
            
            # Save metadata to database
            db_manager.save_embedding_metadata(
                chunk_id=chunk_id,
                embedding_id=embedding_id,
                model_name=self.model_name,
                dimension=self.dimension,
                vector_file_path=vector_file_path
            )
        
        self.logger.info(f"Generated embeddings for {len(chunks_to_embed)} chunks")
        return len(chunks_to_embed)
    
    def embed_all_documents(self) -> Dict[str, int]:
        """
        Generate embeddings for all documents without embeddings
        
        Returns:
            Dictionary with document IDs and chunk counts processed
        """
        documents = db_manager.list_documents()
        results = {}
        
        for doc in documents:
            doc_id = doc['id']
            chunk_count = self.embed_document_chunks(doc_id)
            results[doc_id] = chunk_count
        
        return results
    
    def save_index(self):
        """Save the current index to disk"""
        index_path = settings.embeddings_dir / "faiss_index.bin"
        metadata_path = settings.embeddings_dir / "index_metadata.json"
        
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(index_path))
            
            # Save metadata
            metadata = {
                'model_name': self.model_name,
                'dimension': self.dimension,
                'index_type': settings.vector_db.index_type,
                'total_vectors': self.index.ntotal,
                'chunk_id_to_index': self.chunk_id_to_index,
                'index_to_chunk_id': self.index_to_chunk_id,
                'created_at': datetime.now().isoformat()
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Saved index with {self.index.ntotal} vectors")
            
        except Exception as e:
            self.logger.error(f"Error saving index: {e}")
            raise
    
    def load_index(self, index_path: Path, metadata_path: Path):
        """Load index from disk"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.chunk_id_to_index = metadata['chunk_id_to_index']
            self.index_to_chunk_id = {int(k): v for k, v in metadata['index_to_chunk_id'].items()}
            
            self.logger.info(f"Loaded index with {self.index.ntotal} vectors")
            
        except Exception as e:
            self.logger.error(f"Error loading index: {e}")
            raise
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding statistics"""
        return {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'device': self.device,
            'total_vectors': self.index.ntotal if self.index else 0,
            'index_type': settings.vector_db.index_type,
            'index_size_mb': self.index.ntotal * self.dimension * 4 / (1024 * 1024) if self.index else 0
        }
    
    def remove_chunk_embedding(self, chunk_id: str):
        """
        Remove embedding for a specific chunk
        Note: FAISS doesn't support efficient deletion, so this marks as deleted
        """
        if chunk_id in self.chunk_id_to_index:
            idx = self.chunk_id_to_index[chunk_id]
            # Remove from mappings
            del self.chunk_id_to_index[chunk_id]
            if idx in self.index_to_chunk_id:
                del self.index_to_chunk_id[idx]
            
            self.logger.info(f"Removed embedding mapping for chunk {chunk_id}")
    
    def rebuild_index(self):
        """Rebuild the entire index from database"""
        self.logger.info("Rebuilding embedding index from database")
        
        # Get all chunks with embeddings
        chunks_with_embeddings = []
        documents = db_manager.list_documents()
        
        for doc in documents:
            chunks = db_manager.get_document_chunks(doc['id'])
            for chunk in chunks:
                if chunk.get('embedding_id'):
                    chunks_with_embeddings.append(chunk)
        
        if not chunks_with_embeddings:
            self.logger.info("No chunks with embeddings found")
            return
        
        # Recreate index
        self.create_new_index()
        
        # Load embeddings and add to index
        embeddings_list = []
        chunk_ids = []
        
        for chunk in chunks_with_embeddings:
            try:
                # Load embedding vector from file
                embedding_metadata = db_manager.get_session().query(
                    db_manager.Embedding
                ).filter_by(chunk_id=chunk['id']).first()
                
                if embedding_metadata and os.path.exists(embedding_metadata.vector_file_path):
                    embedding = np.load(embedding_metadata.vector_file_path)
                    embeddings_list.append(embedding)
                    chunk_ids.append(chunk['id'])
            except Exception as e:
                self.logger.warning(f"Error loading embedding for chunk {chunk['id']}: {e}")
                continue
        
        if embeddings_list:
            embeddings_array = np.vstack(embeddings_list)
            self.add_embeddings(embeddings_array, chunk_ids)
            
        self.logger.info(f"Rebuilt index with {len(chunk_ids)} embeddings")

# Global embedding manager instance  
embedding_manager = EmbeddingManager()
'''

with open('multi_doc_rag/core/embeddings.py', 'w') as f:
    f.write(embeddings_content)

print("✅ Embeddings module created")