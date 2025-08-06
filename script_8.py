# Create the retrieval module
retrieval_content = '''"""
Hybrid retrieval system combining dense vector search and sparse BM25
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import defaultdict
import math
import re

# BM25 implementation
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    logging.warning("rank-bm25 not available, using simple implementation")
    BM25Okapi = None

from ..config.settings import settings
from ..core.embeddings import embedding_manager
from ..core.database import db_manager

class SimpleBM25:
    """Simple BM25 implementation as fallback"""
    
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_lengths = [len(doc.split()) for doc in corpus]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        self.doc_frequencies = self._compute_doc_frequencies()
        
    def _compute_doc_frequencies(self) -> Dict[str, int]:
        """Compute document frequencies for each term"""
        doc_frequencies = defaultdict(int)
        
        for doc in self.corpus:
            words = set(doc.lower().split())
            for word in words:
                doc_frequencies[word] += 1
                
        return dict(doc_frequencies)
    
    def _compute_idf(self, term: str) -> float:
        """Compute IDF for a term"""
        n = len(self.corpus)
        df = self.doc_frequencies.get(term.lower(), 0)
        if df == 0:
            return 0
        return math.log((n - df + 0.5) / (df + 0.5))
    
    def get_scores(self, query: str) -> List[float]:
        """Get BM25 scores for query against all documents"""
        query_terms = query.lower().split()
        scores = []
        
        for i, doc in enumerate(self.corpus):
            score = 0
            doc_terms = doc.lower().split()
            doc_length = self.doc_lengths[i]
            
            for term in query_terms:
                if term in doc_terms:
                    tf = doc_terms.count(term)
                    idf = self._compute_idf(term)
                    
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                    
                    score += idf * (numerator / denominator)
                    
            scores.append(score)
            
        return scores

class BM25Retriever:
    """BM25 sparse retrieval implementation"""
    
    def __init__(self):
        self.bm25_model = None
        self.chunk_texts = []
        self.chunk_metadata = []
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, settings.log_level))
    
    def build_index(self, chunks_data: List[Dict[str, Any]]):
        """
        Build BM25 index from chunks data
        
        Args:
            chunks_data: List of chunk dictionaries with text and metadata
        """
        self.chunk_texts = [chunk['text'] for chunk in chunks_data]
        self.chunk_metadata = chunks_data
        
        # Tokenize texts for BM25
        tokenized_corpus = [doc.lower().split() for doc in self.chunk_texts]
        
        if BM25Okapi:
            self.bm25_model = BM25Okapi(tokenized_corpus)
        else:
            # Use simple implementation
            self.bm25_model = SimpleBM25(self.chunk_texts)
        
        self.logger.info(f"Built BM25 index with {len(self.chunk_texts)} documents")
    
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search using BM25
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results
        """
        if not self.bm25_model or not self.chunk_texts:
            return []
        
        if BM25Okapi and hasattr(self.bm25_model, 'get_scores'):
            # Use rank-bm25 library
            query_tokens = query.lower().split()
            scores = self.bm25_model.get_scores(query_tokens)
        else:
            # Use simple implementation
            scores = self.bm25_model.get_scores(query)
        
        # Get top-k results
        scored_results = [(i, score) for i, score in enumerate(scores)]
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for i, (doc_idx, score) in enumerate(scored_results[:k]):
            if score > 0:  # Only include results with positive scores
                chunk_data = self.chunk_metadata[doc_idx]
                results.append({
                    'chunk_id': chunk_data.get('chunk_id', f'chunk_{doc_idx}'),
                    'score': float(score),
                    'text': chunk_data['text'],
                    'document_id': chunk_data.get('document_id', 'unknown'),
                    'metadata': chunk_data.get('metadata', {}),
                    'retrieval_method': 'bm25'
                })
        
        return results

class HybridRetriever:
    """Hybrid retrieval combining vector search and BM25"""
    
    def __init__(self):
        self.bm25_retriever = BM25Retriever()
        self.vector_weight = settings.retrieval.vector_weight
        self.bm25_weight = settings.retrieval.bm25_weight
        self.use_hybrid = settings.retrieval.hybrid_search
        self.setup_logging()
        self.is_initialized = False
    
    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, settings.log_level))
    
    def initialize(self):
        """Initialize the retrieval system"""
        if self.is_initialized:
            return
        
        try:
            # Get all chunks from database for BM25 indexing
            documents = db_manager.list_documents()
            all_chunks = []
            
            for doc in documents:
                chunks = db_manager.get_document_chunks(doc['id'])
                for chunk in chunks:
                    chunk['chunk_id'] = chunk['id']  # Ensure consistent naming
                    all_chunks.append(chunk)
            
            if all_chunks:
                # Build BM25 index
                self.bm25_retriever.build_index(all_chunks)
                self.logger.info(f"Initialized hybrid retrieval with {len(all_chunks)} chunks")
            else:
                self.logger.warning("No chunks found in database for indexing")
                
            self.is_initialized = True
            
        except Exception as e:
            self.logger.error(f"Error initializing retrieval system: {e}")
            raise
    
    def search(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector and BM25 retrieval
        
        Args:
            query: Search query
            k: Number of results to return (default from settings)
            
        Returns:
            List of search results with combined scores
        """
        if not self.is_initialized:
            self.initialize()
        
        k = k or settings.retrieval.top_k
        
        if not self.use_hybrid:
            # Use only vector search
            return embedding_manager.search(query, k)
        
        # Get results from both retrievers
        vector_results = embedding_manager.search(query, k * 2)  # Get more for fusion
        bm25_results = self.bm25_retriever.search(query, k * 2)
        
        # Combine results using reciprocal rank fusion
        fused_results = self._reciprocal_rank_fusion(
            vector_results, 
            bm25_results,
            k
        )
        
        return fused_results[:k]
    
    def _reciprocal_rank_fusion(self, vector_results: List[Dict], 
                              bm25_results: List[Dict], 
                              k: int = 10) -> List[Dict[str, Any]]:
        """
        Combine results using Reciprocal Rank Fusion (RRF)
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            k: Number of final results
            
        Returns:
            Fused and ranked results
        """
        # Create a mapping of chunk_id to combined score
        chunk_scores = defaultdict(float)
        chunk_data = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results):
            chunk_id = result['chunk_id']
            # RRF formula: 1 / (rank + k) where k is usually 60
            rrf_score = self.vector_weight / (rank + 60)
            chunk_scores[chunk_id] += rrf_score
            
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = result.copy()
                chunk_data[chunk_id]['vector_score'] = result['score']
                chunk_data[chunk_id]['vector_rank'] = rank + 1
        
        # Process BM25 results
        for rank, result in enumerate(bm25_results):
            chunk_id = result['chunk_id']
            rrf_score = self.bm25_weight / (rank + 60)
            chunk_scores[chunk_id] += rrf_score
            
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = result.copy()
            
            chunk_data[chunk_id]['bm25_score'] = result['score']
            chunk_data[chunk_id]['bm25_rank'] = rank + 1
        
        # Sort by combined score
        sorted_chunks = sorted(
            chunk_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Prepare final results
        fused_results = []
        for chunk_id, combined_score in sorted_chunks:
            if chunk_id in chunk_data:
                result = chunk_data[chunk_id]
                result['combined_score'] = combined_score
                result['retrieval_method'] = 'hybrid'
                
                # Add individual scores if missing
                if 'vector_score' not in result:
                    result['vector_score'] = 0.0
                    result['vector_rank'] = None
                if 'bm25_score' not in result:
                    result['bm25_score'] = 0.0
                    result['bm25_rank'] = None
                
                fused_results.append(result)
        
        return fused_results
    
    def search_by_document(self, query: str, document_id: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search within a specific document
        
        Args:
            query: Search query
            document_id: Document to search within
            k: Number of results
            
        Returns:
            Search results from the specified document
        """
        # Get all results and filter by document
        all_results = self.search(query, k * 3)  # Get more to ensure we have enough after filtering
        
        document_results = [
            result for result in all_results 
            if result.get('document_id') == document_id
        ]
        
        return document_results[:k]
    
    def get_similar_chunks(self, chunk_id: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Find chunks similar to a given chunk
        
        Args:
            chunk_id: Reference chunk ID
            k: Number of similar chunks to return
            
        Returns:
            List of similar chunks
        """
        # Get the reference chunk
        chunk_info = db_manager.get_chunk(chunk_id)
        if not chunk_info:
            return []
        
        # Search using the chunk text as query
        results = self.search(chunk_info['text'], k + 1)
        
        # Remove the original chunk from results
        similar_chunks = [
            result for result in results 
            if result['chunk_id'] != chunk_id
        ]
        
        return similar_chunks[:k]
    
    def rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank results using advanced scoring (placeholder for future enhancement)
        
        Args:
            query: Original query
            results: Search results to rerank
            
        Returns:
            Reranked results
        """
        if not settings.retrieval.rerank:
            return results
        
        # Simple reranking based on query term overlap
        query_terms = set(query.lower().split())
        
        for result in results:
            text_terms = set(result['text'].lower().split())
            overlap = len(query_terms.intersection(text_terms))
            term_coverage = overlap / len(query_terms) if query_terms else 0
            
            # Boost score based on term coverage
            if 'combined_score' in result:
                result['combined_score'] *= (1 + term_coverage * 0.1)
            else:
                result['score'] *= (1 + term_coverage * 0.1)
        
        # Re-sort by updated scores
        if results and 'combined_score' in results[0]:
            results.sort(key=lambda x: x['combined_score'], reverse=True)
        else:
            results.sort(key=lambda x: x['score'], reverse=True)
        
        return results
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval system statistics"""
        return {
            'hybrid_search_enabled': self.use_hybrid,
            'vector_weight': self.vector_weight,
            'bm25_weight': self.bm25_weight,
            'default_k': settings.retrieval.top_k,
            'reranking_enabled': settings.retrieval.rerank,
            'total_chunks_indexed': len(self.bm25_retriever.chunk_texts),
            'embedding_stats': embedding_manager.get_embedding_stats()
        }

# Global retrieval system instance
hybrid_retriever = HybridRetriever()
'''

with open('multi_doc_rag/core/retrieval.py', 'w') as f:
    f.write(retrieval_content)

print("✅ Retrieval module created")