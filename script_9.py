# Create the generation module
generation_content = '''"""
Response generation using local LLMs with citation support
"""
import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# LLM integration libraries
try:
    from gpt4all import GPT4All
except ImportError:
    logging.warning("GPT4All not available")
    GPT4All = None

try:
    import openai
except ImportError:
    logging.warning("OpenAI library not available")
    openai = None

from ..config.settings import settings

@dataclass
class GeneratedResponse:
    """Container for generated response with metadata"""
    text: str
    sources: List[Dict[str, Any]]
    query: str
    model_used: str
    generation_time: float
    token_count: Optional[int] = None
    confidence_score: Optional[float] = None

class PromptTemplate:
    """Manages prompt templates for different tasks"""
    
    def __init__(self):
        self.templates = {
            'qa_with_citations': '''Based on the provided context from multiple documents, answer the following question. 
Include inline citations using [1], [2], etc. to reference the specific sources.

Context:
{context}

Question: {question}

Instructions:
- Provide a comprehensive answer based on the context
- Use inline citations [1], [2], etc. to reference specific sources
- If information is not available in the context, state this clearly
- Synthesize information from multiple sources when possible
- Be accurate and do not make up information

Answer:''',

            'summarization': '''Summarize the following documents, highlighting key points and themes.

Documents:
{context}

Task: Create a comprehensive summary that:
- Captures the main ideas from all documents
- Identifies common themes and patterns
- Highlights any contradictions or different perspectives
- Uses citations [1], [2], etc. to reference sources

Summary:''',

            'comparative_analysis': '''Compare and analyze the following documents on the topic: {topic}

Documents:
{context}

Task: Provide a comparative analysis that:
- Identifies similarities and differences between the documents
- Analyzes different perspectives or approaches
- Highlights key insights from the comparison
- Uses citations [1], [2], etc. to reference sources

Analysis:'''
        }
    
    def get_template(self, template_name: str) -> str:
        """Get prompt template by name"""
        return self.templates.get(template_name, self.templates['qa_with_citations'])
    
    def format_context(self, retrieval_results: List[Dict[str, Any]]) -> str:
        """Format retrieval results into context string"""
        context_parts = []
        
        for i, result in enumerate(retrieval_results, 1):
            # Get document info for better citations
            doc_info = ""
            if result.get('metadata'):
                metadata = result['metadata']
                if isinstance(metadata, dict):
                    doc_info = f" (Document: {metadata.get('filename', 'Unknown')})"
            
            context_part = f"[{i}] {result['text'][:1000]}{'...' if len(result['text']) > 1000 else ''}{doc_info}"
            context_parts.append(context_part)
        
        return "\\n\\n".join(context_parts)

class LocalLLMProvider:
    """Local LLM provider using GPT4All"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.llm.model_path
        self.model = None
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, settings.log_level))
    
    def load_model(self):
        """Load the local LLM model"""
        if self.model is not None:
            return
        
        if not GPT4All:
            raise ImportError("GPT4All not available. Install with: pip install gpt4all")
        
        try:
            self.logger.info(f"Loading local LLM: {self.model_path}")
            self.model = GPT4All(self.model_path, device='gpu' if settings.embedding.device == 'cuda' else 'cpu')
            self.logger.info("Local LLM loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading local LLM: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: int = None, temperature: float = None) -> str:
        """Generate response using local LLM"""
        if self.model is None:
            self.load_model()
        
        max_tokens = max_tokens or settings.llm.max_tokens
        temperature = temperature or settings.llm.temperature
        
        try:
            response = self.model.generate(
                prompt,
                max_tokens=max_tokens,
                temp=temperature,
                streaming=False
            )
            return response
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            raise

class OpenAIProvider:
    """OpenAI API provider"""
    
    def __init__(self):
        self.client = None
        self.setup_logging()
        
        if settings.openai_api_key:
            self.setup_client()
    
    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, settings.log_level))
    
    def setup_client(self):
        """Setup OpenAI client"""
        if not openai:
            raise ImportError("OpenAI library not available")
        
        openai.api_key = settings.openai_api_key
        self.client = openai
    
    def generate(self, prompt: str, max_tokens: int = None, temperature: float = None) -> str:
        """Generate response using OpenAI API"""
        if not self.client or not settings.openai_api_key:
            raise ValueError("OpenAI API key not configured")
        
        max_tokens = max_tokens or settings.llm.max_tokens
        temperature = temperature or settings.llm.temperature
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful research assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error generating response with OpenAI: {e}")
            raise

class ResponseGenerator:
    """Main response generation orchestrator"""
    
    def __init__(self):
        self.prompt_template = PromptTemplate()
        self.setup_logging()
        
        # Initialize providers based on configuration
        if settings.llm.provider == "openai" and settings.openai_api_key:
            self.provider = OpenAIProvider()
        else:
            self.provider = LocalLLMProvider()
    
    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, settings.log_level))
    
    def generate_response(self, 
                         query: str, 
                         retrieval_results: List[Dict[str, Any]],
                         template_name: str = "qa_with_citations",
                         **generation_kwargs) -> GeneratedResponse:
        """
        Generate response from query and retrieval results
        
        Args:
            query: User query
            retrieval_results: Results from retrieval system
            template_name: Prompt template to use
            **generation_kwargs: Additional generation parameters
            
        Returns:
            GeneratedResponse object
        """
        if not retrieval_results:
            return GeneratedResponse(
                text="I couldn't find any relevant information to answer your question. Please try rephrasing your query or check if documents have been properly indexed.",
                sources=[],
                query=query,
                model_used=settings.llm.provider,
                generation_time=0.0
            )
        
        start_time = datetime.now()
        
        try:
            # Format context from retrieval results
            context = self.prompt_template.format_context(retrieval_results)
            
            # Get and format prompt template
            template = self.prompt_template.get_template(template_name)
            
            if template_name == 'comparative_analysis':
                prompt = template.format(
                    context=context,
                    topic=query  # Use query as topic for comparative analysis
                )
            else:
                prompt = template.format(
                    context=context,
                    question=query
                )
            
            self.logger.info(f"Generating response for query: {query[:100]}...")
            
            # Generate response
            raw_response = self.provider.generate(prompt, **generation_kwargs)
            
            # Process response to extract citations and clean text
            processed_response, citations_used = self._process_response(
                raw_response, retrieval_results
            )
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"Generated response in {generation_time:.2f}s")
            
            return GeneratedResponse(
                text=processed_response,
                sources=citations_used,
                query=query,
                model_used=settings.llm.provider,
                generation_time=generation_time,
                confidence_score=self._calculate_confidence(processed_response, retrieval_results)
            )
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            
            return GeneratedResponse(
                text=f"I encountered an error while generating the response: {str(e)}",
                sources=[],
                query=query,
                model_used=settings.llm.provider,
                generation_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _process_response(self, response: str, retrieval_results: List[Dict[str, Any]]) -> tuple:
        """
        Process response to validate citations and extract sources
        
        Args:
            response: Raw response from LLM
            retrieval_results: Original retrieval results
            
        Returns:
            Tuple of (processed_response, citations_used)
        """
        # Find all citation patterns [1], [2], etc.
        citation_pattern = r'\\[(\\d+)\\]'
        citations_found = re.findall(citation_pattern, response)
        
        citations_used = []
        valid_citations = set()
        
        for citation_num_str in citations_found:
            citation_num = int(citation_num_str)
            if 1 <= citation_num <= len(retrieval_results):
                valid_citations.add(citation_num)
                
                # Add to citations_used if not already present
                result = retrieval_results[citation_num - 1]  # Convert to 0-based index
                citation_info = {
                    'citation_number': citation_num,
                    'chunk_id': result.get('chunk_id'),
                    'document_id': result.get('document_id'),
                    'text_preview': result['text'][:200] + "..." if len(result['text']) > 200 else result['text'],
                    'metadata': result.get('metadata', {}),
                    'score': result.get('score', 0.0)
                }
                
                # Avoid duplicates
                if not any(c['citation_number'] == citation_num for c in citations_used):
                    citations_used.append(citation_info)
        
        # Clean up invalid citations in the response
        def replace_citation(match):
            num = int(match.group(1))
            if num in valid_citations:
                return match.group(0)  # Keep valid citations
            else:
                return ""  # Remove invalid citations
        
        processed_response = re.sub(citation_pattern, replace_citation, response)
        
        # Remove any double spaces left by removed citations
        processed_response = re.sub(r'\\s+', ' ', processed_response).strip()
        
        return processed_response, citations_used
    
    def _calculate_confidence(self, response: str, retrieval_results: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score for the response
        
        Args:
            response: Generated response
            retrieval_results: Retrieval results used
            
        Returns:
            Confidence score between 0 and 1
        """
        if not retrieval_results:
            return 0.0
        
        # Factors for confidence calculation
        factors = []
        
        # 1. Number of sources used
        citations_count = len(re.findall(r'\\[\\d+\\]', response))
        source_coverage = min(citations_count / len(retrieval_results), 1.0)
        factors.append(source_coverage * 0.3)
        
        # 2. Average retrieval score
        avg_score = sum(r.get('score', 0) for r in retrieval_results) / len(retrieval_results)
        # Normalize assuming max score of 1.0 for vector similarity
        normalized_score = min(avg_score, 1.0)
        factors.append(normalized_score * 0.4)
        
        # 3. Response length (longer responses might be more comprehensive)
        response_length_factor = min(len(response) / 1000, 1.0)  # Normalize to 1000 chars
        factors.append(response_length_factor * 0.2)
        
        # 4. Presence of qualifying language (reduces overconfidence)
        qualifying_phrases = ['might', 'could', 'possibly', 'according to', 'suggests']
        has_qualifying = any(phrase in response.lower() for phrase in qualifying_phrases)
        qualification_factor = 0.9 if has_qualifying else 1.0
        factors.append(qualification_factor * 0.1)
        
        confidence = sum(factors)
        return min(max(confidence, 0.0), 1.0)  # Ensure between 0 and 1
    
    def generate_summary(self, retrieval_results: List[Dict[str, Any]]) -> GeneratedResponse:
        """Generate a summary of the retrieved documents"""
        return self.generate_response(
            query="Summarize the key information from these documents",
            retrieval_results=retrieval_results,
            template_name="summarization"
        )
    
    def generate_comparative_analysis(self, topic: str, retrieval_results: List[Dict[str, Any]]) -> GeneratedResponse:
        """Generate a comparative analysis of documents on a specific topic"""
        return self.generate_response(
            query=topic,
            retrieval_results=retrieval_results,
            template_name="comparative_analysis"
        )
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation system statistics"""
        return {
            'provider': settings.llm.provider,
            'model_path': settings.llm.model_path if settings.llm.provider != 'openai' else 'OpenAI API',
            'max_tokens': settings.llm.max_tokens,
            'temperature': settings.llm.temperature,
            'context_length': settings.llm.context_length
        }

# Global response generator instance
response_generator = ResponseGenerator()
'''

with open('multi_doc_rag/core/generation.py', 'w') as f:
    f.write(generation_content)

print("✅ Response generation module created")