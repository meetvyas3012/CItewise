# Create the text processing utilities
text_processing_content = '''"""
Text processing utilities for document preprocessing
"""
import re
import string
from typing import List, Optional
import unicodedata
from pathlib import Path

class TextProcessor:
    """Advanced text processing and cleaning utilities"""
    
    def __init__(self):
        # Common patterns for cleaning
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b')
        self.phone_pattern = re.compile(r'\\b(?:\\d{3}[-.]?)?\\d{3}[-.]?\\d{4}\\b')
        self.excessive_whitespace = re.compile(r'\\s+')
        self.line_breaks = re.compile(r'\\n+')
        
    def clean_text(self, text: str, remove_urls: bool = True, 
                  remove_emails: bool = False, remove_phone: bool = False) -> str:
        """
        Comprehensive text cleaning
        
        Args:
            text: Raw text to clean
            remove_urls: Remove URLs from text
            remove_emails: Remove email addresses
            remove_phone: Remove phone numbers
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
            
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove URLs if specified
        if remove_urls:
            text = self.url_pattern.sub('', text)
            
        # Remove emails if specified  
        if remove_emails:
            text = self.email_pattern.sub('', text)
            
        # Remove phone numbers if specified
        if remove_phone:
            text = self.phone_pattern.sub('', text)
        
        # Fix common encoding issues
        text = text.replace('â€™', "'")
        text = text.replace('â€œ', '"')
        text = text.replace('â€\\x9d', '"')
        text = text.replace('â€"', '—')
        
        # Normalize whitespace
        text = self.excessive_whitespace.sub(' ', text)
        text = self.line_breaks.sub('\\n', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        return text.strip()
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from text using regex patterns
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting pattern
        sentence_pattern = re.compile(r'(?<=[.!?])\\s+(?=[A-Z])')
        sentences = sentence_pattern.split(text)
        
        # Filter out very short sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return sentences
    
    def extract_paragraphs(self, text: str) -> List[str]:
        """
        Extract paragraphs from text
        
        Args:
            text: Input text
            
        Returns:
            List of paragraphs
        """
        paragraphs = text.split('\\n\\n')
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 50]
        return paragraphs
    
    def remove_headers_footers(self, text: str, 
                              header_patterns: Optional[List[str]] = None,
                              footer_patterns: Optional[List[str]] = None) -> str:
        """
        Remove common headers and footers from documents
        
        Args:
            text: Input text
            header_patterns: List of regex patterns for headers
            footer_patterns: List of regex patterns for footers
            
        Returns:
            Text with headers/footers removed
        """
        if header_patterns is None:
            header_patterns = [
                r'^.*Page \\d+ of \\d+.*$',
                r'^.*\\d{1,2}/\\d{1,2}/\\d{4}.*$',  # Date patterns
                r'^.*CONFIDENTIAL.*$',
                r'^.*DRAFT.*$'
            ]
            
        if footer_patterns is None:
            footer_patterns = [
                r'^.*Page \\d+.*$',
                r'^.*Copyright.*$',
                r'^.*All rights reserved.*$',
                r'^.*\\w+\\.com.*$'  # Website patterns
            ]
        
        lines = text.split('\\n')
        filtered_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                filtered_lines.append(line)
                continue
                
            # Check against header patterns
            is_header_footer = False
            for pattern in header_patterns + footer_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    is_header_footer = True
                    break
                    
            if not is_header_footer:
                filtered_lines.append(line)
                
        return '\\n'.join(filtered_lines)
    
    def extract_key_phrases(self, text: str, min_length: int = 3) -> List[str]:
        """
        Extract key phrases from text using simple heuristics
        
        Args:
            text: Input text
            min_length: Minimum phrase length
            
        Returns:
            List of key phrases
        """
        # Remove punctuation and convert to lowercase
        clean_text = text.translate(str.maketrans('', '', string.punctuation))
        words = clean_text.lower().split()
        
        # Extract n-grams (2-4 words)
        phrases = []
        for n in range(2, 5):
            for i in range(len(words) - n + 1):
                phrase = ' '.join(words[i:i+n])
                if len(phrase) >= min_length:
                    phrases.append(phrase)
        
        # Simple frequency-based filtering
        phrase_counts = {}
        for phrase in phrases:
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
            
        # Return phrases that appear more than once
        key_phrases = [phrase for phrase, count in phrase_counts.items() if count > 1]
        return sorted(key_phrases, key=lambda x: phrase_counts[x], reverse=True)
    
    def preserve_structure(self, text: str) -> dict:
        """
        Analyze and preserve document structure
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with structure information
        """
        structure = {
            'titles': [],
            'sections': [],
            'lists': [],
            'tables': []
        }
        
        lines = text.split('\\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Detect titles (all caps, short lines)
            if line.isupper() and len(line) < 100:
                structure['titles'].append({'text': line, 'line': i})
                
            # Detect section headers (starts with number or letter)
            if re.match(r'^[0-9]+\\.\\s|^[A-Z]\\.\\s|^#{1,6}\\s', line):
                structure['sections'].append({'text': line, 'line': i})
                
            # Detect list items
            if re.match(r'^[•\\-\\*]\\s|^\\d+\\.\\s|^[a-z]\\)\\s', line):
                structure['lists'].append({'text': line, 'line': i})
                
            # Detect table-like structures (multiple | characters)
            if line.count('|') >= 2:
                structure['tables'].append({'text': line, 'line': i})
        
        return structure

# Global text processor instance
text_processor = TextProcessor()
'''

with open('multi_doc_rag/utils/text_processing.py', 'w') as f:
    f.write(text_processing_content)

print("✅ Text processing utilities created")