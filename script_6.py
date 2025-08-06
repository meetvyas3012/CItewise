# Create the database module
database_content = '''"""
Database management for document metadata and embeddings
"""
import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Text, Integer, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import logging

from ..config.settings import settings

Base = declarative_base()

class Document(Base):
    """Document table model"""
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True)  # MD5 hash
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    mime_type = Column(String)
    file_size = Column(Integer)
    content = Column(Text)
    metadata = Column(JSON)
    processing_info = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    indexed_at = Column(DateTime)

class Chunk(Base):
    """Chunk table model"""
    __tablename__ = "chunks"
    
    id = Column(String, primary_key=True)  # chunk_id
    document_id = Column(String, nullable=False)  # Foreign key to documents
    text = Column(Text, nullable=False)
    start_index = Column(Integer)
    end_index = Column(Integer)
    chunk_number = Column(Integer)
    metadata = Column(JSON)
    embedding_id = Column(String)  # Reference to embedding vector
    created_at = Column(DateTime, default=datetime.utcnow)

class Embedding(Base):
    """Embedding table model (metadata only, vectors stored separately)"""
    __tablename__ = "embeddings"
    
    id = Column(String, primary_key=True)  # embedding_id
    chunk_id = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    dimension = Column(Integer, nullable=False)
    vector_file_path = Column(String)  # Path to the actual vector file
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    """Database operations manager"""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or settings.database.url
        self.engine = create_engine(
            self.database_url,
            echo=settings.database.echo
        )
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.setup_logging()
        self.init_database()
    
    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, settings.log_level))
    
    def init_database(self):
        """Initialize database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            self.logger.info("Database tables initialized")
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    def save_document(self, processed_doc) -> str:
        """
        Save processed document to database
        
        Args:
            processed_doc: ProcessedDocument object
            
        Returns:
            Document ID
        """
        with self.get_session() as session:
            try:
                # Check if document already exists
                existing_doc = session.query(Document).filter_by(
                    id=processed_doc.metadata.hash_md5
                ).first()
                
                if existing_doc:
                    self.logger.info(f"Document already exists: {processed_doc.metadata.filename}")
                    # Update existing document
                    existing_doc.content = processed_doc.content
                    existing_doc.metadata = processed_doc.metadata.__dict__
                    existing_doc.processing_info = processed_doc.processing_info
                    existing_doc.updated_at = datetime.utcnow()
                    document_id = existing_doc.id
                else:
                    # Create new document
                    document = Document(
                        id=processed_doc.metadata.hash_md5,
                        filename=processed_doc.metadata.filename,
                        file_path=processed_doc.metadata.file_path,
                        file_type=processed_doc.metadata.file_type,
                        mime_type=processed_doc.metadata.mime_type,
                        file_size=processed_doc.metadata.file_size,
                        content=processed_doc.content,
                        metadata=processed_doc.metadata.__dict__,
                        processing_info=processed_doc.processing_info
                    )
                    session.add(document)
                    document_id = document.id
                
                # Save chunks
                self.save_chunks(session, processed_doc.chunks, document_id)
                
                session.commit()
                self.logger.info(f"Saved document: {processed_doc.metadata.filename}")
                
                return document_id
                
            except Exception as e:
                session.rollback()
                self.logger.error(f"Error saving document: {e}")
                raise
    
    def save_chunks(self, session: Session, chunks: List, document_id: str):
        """Save document chunks to database"""
        # Delete existing chunks for this document
        session.query(Chunk).filter_by(document_id=document_id).delete()
        
        # Save new chunks
        for chunk in chunks:
            chunk_record = Chunk(
                id=chunk.chunk_id,
                document_id=document_id,
                text=chunk.text,
                start_index=chunk.start_index,
                end_index=chunk.end_index,
                chunk_number=chunk.metadata.get('chunk_number', 0),
                metadata=chunk.metadata
            )
            session.add(chunk_record)
    
    def get_document(self, document_id: str) -> Optional[Dict]:
        """Get document by ID"""
        with self.get_session() as session:
            document = session.query(Document).filter_by(id=document_id).first()
            if document:
                return {
                    'id': document.id,
                    'filename': document.filename,
                    'file_path': document.file_path,
                    'content': document.content,
                    'metadata': document.metadata,
                    'processing_info': document.processing_info,
                    'created_at': document.created_at,
                    'updated_at': document.updated_at
                }
            return None
    
    def get_document_chunks(self, document_id: str) -> List[Dict]:
        """Get all chunks for a document"""
        with self.get_session() as session:
            chunks = session.query(Chunk).filter_by(document_id=document_id).all()
            return [
                {
                    'id': chunk.id,
                    'text': chunk.text,
                    'start_index': chunk.start_index,
                    'end_index': chunk.end_index,
                    'metadata': chunk.metadata,
                    'embedding_id': chunk.embedding_id
                }
                for chunk in chunks
            ]
    
    def get_chunk(self, chunk_id: str) -> Optional[Dict]:
        """Get chunk by ID"""
        with self.get_session() as session:
            chunk = session.query(Chunk).filter_by(id=chunk_id).first()
            if chunk:
                return {
                    'id': chunk.id,
                    'document_id': chunk.document_id,
                    'text': chunk.text,
                    'start_index': chunk.start_index,
                    'end_index': chunk.end_index,
                    'metadata': chunk.metadata,
                    'embedding_id': chunk.embedding_id
                }
            return None
    
    def list_documents(self, limit: Optional[int] = None) -> List[Dict]:
        """List all documents"""
        with self.get_session() as session:
            query = session.query(Document).order_by(Document.created_at.desc())
            if limit:
                query = query.limit(limit)
            
            documents = query.all()
            return [
                {
                    'id': doc.id,
                    'filename': doc.filename,
                    'file_type': doc.file_type,
                    'file_size': doc.file_size,
                    'created_at': doc.created_at,
                    'chunk_count': len(self.get_document_chunks(doc.id))
                }
                for doc in documents
            ]
    
    def search_documents(self, query: str) -> List[Dict]:
        """Simple text search in documents"""
        with self.get_session() as session:
            documents = session.query(Document).filter(
                Document.content.contains(query)
            ).all()
            
            return [
                {
                    'id': doc.id,
                    'filename': doc.filename,
                    'file_type': doc.file_type,
                    'relevance_snippet': self._get_snippet(doc.content, query)
                }
                for doc in documents
            ]
    
    def _get_snippet(self, content: str, query: str, snippet_length: int = 200) -> str:
        """Get text snippet around query match"""
        query_pos = content.lower().find(query.lower())
        if query_pos == -1:
            return content[:snippet_length]
        
        start = max(0, query_pos - snippet_length // 2)
        end = min(len(content), query_pos + snippet_length // 2)
        
        snippet = content[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."
            
        return snippet
    
    def save_embedding_metadata(self, chunk_id: str, embedding_id: str, 
                              model_name: str, dimension: int, 
                              vector_file_path: str):
        """Save embedding metadata"""
        with self.get_session() as session:
            try:
                embedding = Embedding(
                    id=embedding_id,
                    chunk_id=chunk_id,
                    model_name=model_name,
                    dimension=dimension,
                    vector_file_path=vector_file_path
                )
                session.add(embedding)
                
                # Update chunk with embedding reference
                chunk = session.query(Chunk).filter_by(id=chunk_id).first()
                if chunk:
                    chunk.embedding_id = embedding_id
                
                session.commit()
                self.logger.info(f"Saved embedding metadata for chunk {chunk_id}")
                
            except Exception as e:
                session.rollback()
                self.logger.error(f"Error saving embedding metadata: {e}")
                raise
    
    def get_chunks_for_embedding(self, batch_size: int = 100) -> List[Tuple[str, str]]:
        """Get chunks that need embedding"""
        with self.get_session() as session:
            chunks = session.query(Chunk).filter(
                Chunk.embedding_id.is_(None)
            ).limit(batch_size).all()
            
            return [(chunk.id, chunk.text) for chunk in chunks]
    
    def delete_document(self, document_id: str):
        """Delete document and its chunks"""
        with self.get_session() as session:
            try:
                # Delete chunks first
                session.query(Chunk).filter_by(document_id=document_id).delete()
                
                # Delete embeddings for this document's chunks
                chunk_ids = [chunk[0] for chunk in session.query(Chunk.id).filter_by(document_id=document_id).all()]
                if chunk_ids:
                    session.query(Embedding).filter(Embedding.chunk_id.in_(chunk_ids)).delete(synchronize_session=False)
                
                # Delete document
                session.query(Document).filter_by(id=document_id).delete()
                
                session.commit()
                self.logger.info(f"Deleted document: {document_id}")
                
            except Exception as e:
                session.rollback()
                self.logger.error(f"Error deleting document: {e}")
                raise
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        with self.get_session() as session:
            doc_count = session.query(Document).count()
            chunk_count = session.query(Chunk).count()
            embedding_count = session.query(Embedding).count()
            
            # Get file type distribution
            file_types = session.query(Document.file_type, func.count()).group_by(Document.file_type).all()
            
            return {
                'document_count': doc_count,
                'chunk_count': chunk_count,
                'embedding_count': embedding_count,
                'file_types': dict(file_types) if file_types else {},
                'chunks_without_embeddings': chunk_count - embedding_count
            }

# Global database manager instance
db_manager = DatabaseManager()
'''

# Add missing import
import_fix = '''from sqlalchemy import func
'''

full_content = database_content.replace('from ..config.settings import settings', 
                                       'from sqlalchemy import func\nfrom ..config.settings import settings')

with open('multi_doc_rag/core/database.py', 'w') as f:
    f.write(full_content)

print("✅ Database module created")