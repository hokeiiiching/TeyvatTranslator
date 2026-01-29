# -*- coding: utf-8 -*-
"""
RAG Engine Module - Retrieval-Augmented Generation

Provides semantic search and context retrieval for more accurate
Genshin Impact translations. Uses embeddings to find relevant 
vocabulary, quest context, and character information.

Note: This is a stub implementation. Full RAG requires additional
dependencies: sentence-transformers, chromadb (or faiss-cpu).
"""

from typing import List, Dict, Any, Optional
import json
import os

from src.data.vocabulary import VOCABULARY

# =============================================================================
# GLOBAL RAG INSTANCE (for background pre-initialization)
# =============================================================================
_global_rag_engine = None
_rag_init_thread = None
_rag_ready = False

def _init_rag_sync():
    """Initialize RAGEngine synchronously (called in background thread)."""
    global _global_rag_engine, _rag_ready
    
    try:
        print("⏳ Initializing RAG engine in background...")
        _global_rag_engine = RAGEngine(use_embeddings=True)
        _rag_ready = True
    except Exception as e:
        print(f"❌ Background RAG init failed: {e}")


def preload_rag():
    """
    Start RAG engine initialization in a background thread.
    Call this at app startup to reduce wait time when OCR is first needed.
    """
    global _rag_init_thread
    
    if _global_rag_engine is not None or _rag_ready:
        return  # Already initialized
    
    from threading import Thread
    
    _rag_init_thread = Thread(target=_init_rag_sync, daemon=True)
    _rag_init_thread.start()
    print("⏳ RAG background initialization started")


def get_rag_engine():
    """
    Get the shared RAGEngine instance.
    If preload_rag() was called, returns the pre-initialized instance.
    Otherwise, initializes on demand (lazy loading).
    """
    global _global_rag_engine, _rag_init_thread, _rag_ready
    
    # Wait for background init to complete if in progress
    if _rag_init_thread is not None and _rag_init_thread.is_alive():
        print("⏳ Waiting for background RAG init to complete...")
        _rag_init_thread.join()
    
    # Return cached instance if available
    if _global_rag_engine is not None:
        return _global_rag_engine
    
    # Fallback: initialize synchronously
    _init_rag_sync()
    return _global_rag_engine


class RAGEngine:
    """
    Retrieval-Augmented Generation engine for context-aware translation.
    
    Uses semantic similarity to find the most relevant vocabulary terms
    and context for a given Chinese text input. This helps provide more
    accurate translations for Genshin-specific terminology.
    
    Attributes:
        vocabulary: List of vocabulary term dictionaries
        knowledge_base: Additional context data (quests, characters)
        use_embeddings: Whether to use semantic embeddings (requires extra deps)
    """
    
    def __init__(self, use_embeddings: bool = False) -> None:
        """
        Initialize the RAG engine.
        
        Args:
            use_embeddings: If True, use sentence embeddings for semantic search.
                           Requires sentence-transformers and chromadb.
        """
        self.vocabulary = VOCABULARY
        self.knowledge_base = self._load_knowledge_base()
        self.use_embeddings = use_embeddings
        
        # LRU cache for query results (speaker names and phrases repeat frequently)
        from collections import OrderedDict
        self._query_cache: OrderedDict = OrderedDict()
        self._cache_max_size = 100
        
        # Build exact match lookup dict for O(1) speaker name lookup
        self._exact_match_dict = {term.get('mandarin', ''): term for term in self.vocabulary if term.get('mandarin')}
        
        if use_embeddings:
            self._init_embeddings()
        
        print(f"✓ RAG engine initialized: {len(self.vocabulary)} terms")
    
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """
        Load additional knowledge base files (quests, characters).
        
        Returns:
            Dictionary containing loaded knowledge data.
        """
        kb = {
            'quests': [],
            'characters': [],
            'locations': []
        }
        
        data_dir = os.path.dirname(__file__).replace('engine', 'data')
        
        # Load quest data if available
        quest_path = os.path.join(data_dir, 'quests.json')
        if os.path.exists(quest_path):
            with open(quest_path, 'r', encoding='utf-8') as f:
                kb['quests'] = json.load(f)
                
        # Load character data if available
        char_path = os.path.join(data_dir, 'characters.json')
        if os.path.exists(char_path):
            with open(char_path, 'r', encoding='utf-8') as f:
                kb['characters'] = json.load(f)
        
        return kb
    
    def _init_embeddings(self) -> None:
        """
        Initialize sentence embeddings for semantic search.
        
        Requires: sentence-transformers, chromadb
        """
        try:
            from sentence_transformers import SentenceTransformer
            import chromadb
            
            # Load multilingual model for Chinese/English
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
            # Create or get existing vector store collection
            self.chroma_client = chromadb.Client()
            self.collection = self.chroma_client.get_or_create_collection("genshin_vocab")
            
            # Add vocabulary to vector store
            self._index_vocabulary()
            
            print("✓ Embeddings initialized")
            
        except ImportError:
            print("⚠ sentence-transformers or chromadb not installed, using keyword matching")
            self.use_embeddings = False
    
    def _index_vocabulary(self) -> None:
        """Index all vocabulary terms in the vector store."""
        if not self.use_embeddings:
            return
            
        documents = []
        ids = []
        metadatas = []
        
        for i, term in enumerate(self.vocabulary):
            # Combine relevant text for embedding
            text = f"{term.get('mandarin', '')} {term.get('english', '')} {term.get('game_definition', '')}"
            documents.append(text)
            ids.append(f"vocab_{i}")
            metadatas.append({'idx': i})
        
        # Generate embeddings and add to collection
        embeddings = self.model.encode(documents).tolist()
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
    
    def get_context(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant context for a given text.
        
        Args:
            text: Chinese text to find context for.
            top_k: Number of top matches to return.
            
        Returns:
            List of relevant vocabulary terms and context.
        """
        # Check cache first
        cache_key = (text, top_k)
        if cache_key in self._query_cache:
            # Move to end for LRU
            self._query_cache.move_to_end(cache_key)
            return self._query_cache[cache_key]
        
        # Compute result
        if self.use_embeddings:
            result = self._semantic_search(text, top_k)
        else:
            result = self._keyword_search(text, top_k)
        
        # Cache result with LRU eviction
        while len(self._query_cache) >= self._cache_max_size:
            self._query_cache.popitem(last=False)
        self._query_cache[cache_key] = result
        
        return result
    
    def _semantic_search(self, text: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Search using semantic embeddings with exact match priority.
        
        Args:
            text: Text to search for.
            top_k: Number of results.
            
        Returns:
            List of matching terms.
        """
        # FAST PATH: O(1) exact match lookup (most common case for speaker names)
        if text in self._exact_match_dict:
            # Return exact match immediately - skip expensive embedding computation
            return [self._exact_match_dict[text]]
        
        matches = []
        
        # No exact match - use pure embedding search
        query_embedding = self.model.encode([text]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        for metadata in results['metadatas'][0]:
            idx = metadata['idx']
            matches.append(self.vocabulary[idx])
        
        return matches
    
    def _keyword_search(self, text: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Fallback keyword-based search with fuzzy matching.
        
        Args:
            text: Text to search for.
            top_k: Number of results.
            
        Returns:
            List of matching terms.
        """
        matches = []
        seen = set()
        
        # EXACT match first (for speaker name lookup)
        for term in self.vocabulary:
            mandarin = term.get('mandarin', '')
            if mandarin and mandarin == text and mandarin not in seen:
                matches.insert(0, term)  # Insert at front for priority
                seen.add(mandarin)
                break  # Only one exact match needed
        
        # Substring matching (original mandarin in text)
        for term in self.vocabulary:
            mandarin = term.get('mandarin', '')
            if mandarin and mandarin in text and mandarin not in seen:
                matches.append(term)
                seen.add(mandarin)
        
        # Fuzzy matching for OCR errors (if no exact matches)
        if not matches:
            for term in self.vocabulary:
                mandarin = term.get('mandarin', '')
                if mandarin and mandarin not in seen:
                    # Calculate character-level similarity
                    similarity = self._char_similarity(text, mandarin)
                    if similarity >= 0.7:  # 70% threshold
                        term_copy = term.copy()
                        term_copy['_similarity'] = similarity
                        matches.append(term_copy)
                        seen.add(mandarin)
            
            # Sort by similarity score
            matches.sort(key=lambda t: t.get('_similarity', 0), reverse=True)
        else:
            # Sort by match length (longer = more specific), but keep exact match first
            exact_match = matches[0] if matches and matches[0].get('mandarin', '') == text else None
            other_matches = matches[1:] if exact_match else matches
            other_matches.sort(key=lambda t: len(t.get('mandarin', '')), reverse=True)
            if exact_match:
                matches = [exact_match] + other_matches
            else:
                matches = other_matches
        
        return matches[:top_k]
    
    def _char_similarity(self, text1: str, text2: str) -> float:
        """Calculate character-level similarity between two strings."""
        if not text1 or not text2:
            return 0.0
        
        # Count matching characters
        chars1 = set(text1)
        chars2 = set(text2)
        intersection = chars1 & chars2
        union = chars1 | chars2
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def enhance_translation(
        self, 
        text: str, 
        base_translation: str,
        context: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Enhance a translation with retrieved context.
        
        Args:
            text: Original Chinese text.
            base_translation: Initial translation from Google Translate.
            context: Optional pre-retrieved context.
            
        Returns:
            Enhanced translation result with context information.
        """
        if context is None:
            context = self.get_context(text)
        
        result = {
            'original': text,
            'translation': base_translation,
            'context_matches': context,
            'enhanced': False
        }
        
        # If we found matching vocabulary, use its translation
        for match in context:
            if match.get('mandarin') == text:
                result['translation'] = match.get('english', base_translation)
                result['pinyin'] = match.get('pinyin', '')
                result['game_context'] = match.get('game_definition', '')
                result['breakdown'] = match.get('literal_breakdown', '')
                result['enhanced'] = True
                break
        
        return result
