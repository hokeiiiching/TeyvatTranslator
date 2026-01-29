# -*- coding: utf-8 -*-
"""
Context Engine Module

Provides vocabulary matching and lookup functionality.
Searches the curated Genshin Impact term database for
context-aware translations with learning information.
"""

from typing import List, Dict, Any, Optional

from src.data.vocabulary import VOCABULARY


class ContextEngine:
    """
    Context-aware vocabulary matching engine.
    
    Indexes and searches the Genshin vocabulary database
    to find matching terms with their associated context
    information (pinyin, breakdowns, definitions).
    
    Attributes:
        vocabulary: List of vocabulary term dictionaries
        index: Lookup index for fast matching
    """
    
    def __init__(self) -> None:
        """Initialize the context engine with vocabulary data."""
        self.vocabulary = VOCABULARY
        self.index = self._build_index()
        print(f"✓ Context engine loaded: {len(self.vocabulary)} terms")
        
    def _build_index(self) -> Dict[str, Dict[str, Any]]:
        """
        Build a search index for fast vocabulary lookups.
        
        Returns:
            Dictionary mapping Chinese text to term data.
        """
        index = {}
        
        for term in self.vocabulary:
            mandarin = term.get('mandarin', '')
            if mandarin:
                index[mandarin] = term
                
                # Also index substrings for partial matching
                for i in range(len(mandarin)):
                    for j in range(i + 2, len(mandarin) + 1):
                        substr = mandarin[i:j]
                        if substr not in index:
                            index[substr] = term
                            
        return index
        
    def find_matches(self, text: str) -> List[Dict[str, Any]]:
        """
        Find vocabulary matches within the given text.
        
        Args:
            text: Chinese text to search for matches.
            
        Returns:
            List of matching term dictionaries, sorted by relevance.
        """
        matches = []
        seen = set()
        
        # Check for exact match first
        if text in self.index:
            term = self.index[text]
            mandarin = term.get('mandarin', '')
            if mandarin not in seen:
                matches.append(term)
                seen.add(mandarin)
                
        # Find all vocabulary terms that appear in the text
        for term in self.vocabulary:
            mandarin = term.get('mandarin', '')
            if mandarin and mandarin in text and mandarin not in seen:
                matches.append(term)
                seen.add(mandarin)
                
        # Sort by length (longer matches are more specific)
        matches.sort(key=lambda t: len(t.get('mandarin', '')), reverse=True)
        
        return matches[:5]  # Return top 5 matches
        
    def get_term_by_id(self, term_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific term by its ID.
        
        Args:
            term_id: Unique identifier of the term.
            
        Returns:
            Term dictionary or None if not found.
        """
        for term in self.vocabulary:
            if term.get('id') == term_id:
                return term
        return None
        
    def get_terms_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """
        Get all terms with a specific tag.
        
        Args:
            tag: Tag to filter by (e.g., 'combat', 'dialogue').
            
        Returns:
            List of matching terms.
        """
        return [t for t in self.vocabulary if tag in t.get('tags', [])]
        
    def get_terms_by_region(self, region: str) -> List[Dict[str, Any]]:
        """
        Get all terms for a specific region.
        
        Args:
            region: Region name (e.g., 'liyue', 'mondstadt').
            
        Returns:
            List of terms from that region.
        """
        return [t for t in self.vocabulary if t.get('region') == region]
