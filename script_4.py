# Create the chunking utilities
chunking_content = '''"""
Advanced document chunking strategies for optimal retrieval
"""
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..utils.text_processing import text_processor

@dataclass
class Chunk:
    """Represents a document chunk with metadata"""
    text: str
    start_index: int
    end_index: int
    chunk_id: str
    document_id: str
    metadata: Dict[str, Any]
    
    def __len__(self):
        return len(self.text)

class ChunkingStrategy:
    """Base class for chunking strategies"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def chunk(self, text: str, document_id: str) -> List[Chunk]:
        """
        Abstract method to be implemented by subclasses
        
        Args:
            text: Document text to chunk
            document_id: Unique identifier for the document
            
        Returns:
            List of chunks
        """
        raise NotImplementedError

class FixedSizeChunking(ChunkingStrategy):
    """Simple fixed-size chunking with overlap"""
    
    def chunk(self, text: str, document_id: str) -> List[Chunk]:
        """
        Split text into fixed-size chunks with overlap
        
        Args:
            text: Document text to chunk  
            document_id: Unique identifier for the document
            
        Returns:
            List of chunks
        """
        chunks = []
        start = 0
        chunk_num = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending within the last 100 characters
                sentence_end = text.rfind('.', max(start, end - 100), end)
                if sentence_end != -1:
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk = Chunk(
                    text=chunk_text,
                    start_index=start,
                    end_index=end,
                    chunk_id=f"{document_id}_chunk_{chunk_num}",
                    document_id=document_id,
                    metadata={
                        'chunk_number': chunk_num,
                        'chunk_type': 'fixed_size',
                        'word_count': len(chunk_text.split()),
                        'char_count': len(chunk_text)
                    }
                )
                chunks.append(chunk)
                chunk_num += 1
            
            # Move start position with overlap
            start = max(start + self.chunk_size - self.overlap, end)
            
        return chunks

class RecursiveChunking(ChunkingStrategy):
    """Recursive chunking that respects document structure"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        super().__init__(chunk_size, overlap)
        self.separators = [
            "\\n\\n",  # Paragraph breaks
            "\\n",     # Line breaks  
            ". ",      # Sentence endings
            " ",       # Word boundaries
            ""         # Character level (last resort)
        ]
    
    def chunk(self, text: str, document_id: str) -> List[Chunk]:
        """
        Recursively split text using hierarchical separators
        
        Args:
            text: Document text to chunk
            document_id: Unique identifier for the document
            
        Returns:
            List of chunks
        """
        return self._recursive_split(text, document_id, 0)
    
    def _recursive_split(self, text: str, document_id: str, 
                        start_offset: int = 0) -> List[Chunk]:
        """Recursive splitting logic"""
        chunks = []
        
        if len(text) <= self.chunk_size:
            # Base case: text fits in one chunk
            if text.strip():
                chunk = Chunk(
                    text=text.strip(),
                    start_index=start_offset,
                    end_index=start_offset + len(text),
                    chunk_id=f"{document_id}_chunk_{len(chunks)}",
                    document_id=document_id,
                    metadata={
                        'chunk_type': 'recursive',
                        'word_count': len(text.split()),
                        'char_count': len(text)
                    }
                )
                chunks.append(chunk)
            return chunks
        
        # Try each separator in order
        for separator in self.separators:
            if separator in text:
                splits = text.split(separator)
                current_chunk = ""
                current_start = start_offset
                
                for i, split in enumerate(splits):
                    potential_chunk = current_chunk + (separator if current_chunk else "") + split
                    
                    if len(potential_chunk) <= self.chunk_size:
                        current_chunk = potential_chunk
                    else:
                        # Current chunk is ready
                        if current_chunk.strip():
                            chunk = Chunk(
                                text=current_chunk.strip(),
                                start_index=current_start,
                                end_index=current_start + len(current_chunk),
                                chunk_id=f"{document_id}_chunk_{len(chunks)}",
                                document_id=document_id,
                                metadata={
                                    'chunk_type': 'recursive',
                                    'separator_used': separator,
                                    'word_count': len(current_chunk.split()),
                                    'char_count': len(current_chunk)
                                }
                            )
                            chunks.append(chunk)
                        
                        # Start new chunk with overlap
                        overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else current_chunk
                        current_chunk = overlap_text + separator + split
                        current_start = current_start + len(current_chunk) - len(overlap_text + separator + split)
                
                # Add final chunk
                if current_chunk.strip():
                    chunk = Chunk(
                        text=current_chunk.strip(),
                        start_index=current_start,
                        end_index=current_start + len(current_chunk),
                        chunk_id=f"{document_id}_chunk_{len(chunks)}",
                        document_id=document_id,
                        metadata={
                            'chunk_type': 'recursive',
                            'separator_used': separator,
                            'word_count': len(current_chunk.split()),
                            'char_count': len(current_chunk)
                        }
                    )
                    chunks.append(chunk)
                
                return chunks
        
        # Fallback: use fixed-size chunking
        return FixedSizeChunking(self.chunk_size, self.overlap).chunk(text, document_id)

class SemanticChunking(ChunkingStrategy):
    """Semantic chunking based on topic coherence"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200, 
                 similarity_threshold: float = 0.5):
        super().__init__(chunk_size, overlap)
        self.similarity_threshold = similarity_threshold
    
    def chunk(self, text: str, document_id: str) -> List[Chunk]:
        """
        Chunk text based on semantic similarity
        Note: This is a simplified version. Full implementation would use embeddings.
        
        Args:
            text: Document text to chunk
            document_id: Unique identifier for the document
            
        Returns:
            List of chunks
        """
        # Extract sentences
        sentences = text_processor.extract_sentences(text)
        
        if len(sentences) <= 1:
            return FixedSizeChunking(self.chunk_size, self.overlap).chunk(text, document_id)
        
        chunks = []
        current_chunk_sentences = []
        current_length = 0
        chunk_num = 0
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            
            # Check if adding this sentence would exceed chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk_sentences:
                # Create chunk from current sentences
                chunk_text = ' '.join(current_chunk_sentences)
                start_pos = text.find(current_chunk_sentences[0])
                end_pos = start_pos + len(chunk_text)
                
                chunk = Chunk(
                    text=chunk_text,
                    start_index=start_pos,
                    end_index=end_pos,
                    chunk_id=f"{document_id}_chunk_{chunk_num}",
                    document_id=document_id,
                    metadata={
                        'chunk_type': 'semantic',
                        'sentence_count': len(current_chunk_sentences),
                        'word_count': len(chunk_text.split()),
                        'char_count': len(chunk_text)
                    }
                )
                chunks.append(chunk)
                chunk_num += 1
                
                # Start new chunk with overlap
                if self.overlap > 0 and len(current_chunk_sentences) > 1:
                    overlap_sentences = current_chunk_sentences[-1:]  # Take last sentence for overlap
                    current_chunk_sentences = overlap_sentences + [sentence]
                    current_length = sum(len(s) for s in current_chunk_sentences)
                else:
                    current_chunk_sentences = [sentence]
                    current_length = sentence_length
            else:
                current_chunk_sentences.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            start_pos = text.find(current_chunk_sentences[0])
            end_pos = start_pos + len(chunk_text)
            
            chunk = Chunk(
                text=chunk_text,
                start_index=start_pos,
                end_index=end_pos,
                chunk_id=f"{document_id}_chunk_{chunk_num}",
                document_id=document_id,
                metadata={
                    'chunk_type': 'semantic',
                    'sentence_count': len(current_chunk_sentences),
                    'word_count': len(chunk_text.split()),
                    'char_count': len(chunk_text)
                }
            )
            chunks.append(chunk)
        
        return chunks

class ChunkingManager:
    """Manager class for different chunking strategies"""
    
    def __init__(self):
        self.strategies = {
            'fixed': FixedSizeChunking,
            'recursive': RecursiveChunking,
            'semantic': SemanticChunking
        }
    
    def get_strategy(self, strategy_name: str, **kwargs) -> ChunkingStrategy:
        """
        Get chunking strategy by name
        
        Args:
            strategy_name: Name of the strategy ('fixed', 'recursive', 'semantic')
            **kwargs: Parameters for the strategy
            
        Returns:
            ChunkingStrategy instance
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown chunking strategy: {strategy_name}")
            
        return self.strategies[strategy_name](**kwargs)
    
    def chunk_document(self, text: str, document_id: str, 
                      strategy: str = 'recursive', **kwargs) -> List[Chunk]:
        """
        Chunk a document using specified strategy
        
        Args:
            text: Document text to chunk
            document_id: Unique identifier for the document
            strategy: Chunking strategy to use
            **kwargs: Parameters for the chunking strategy
            
        Returns:
            List of chunks
        """
        chunking_strategy = self.get_strategy(strategy, **kwargs)
        return chunking_strategy.chunk(text, document_id)

# Global chunking manager instance
chunking_manager = ChunkingManager()
'''

with open('multi_doc_rag/utils/chunking.py', 'w') as f:
    f.write(chunking_content)

print("✅ Chunking utilities created")