# -*- coding: utf-8 -*-
"""
SQLite Database Module for Genshin Vocabulary

Provides persistent storage for vocabulary, user history, and cached translations.
"""

import sqlite3
import os
import json
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

# Database path
DB_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(DB_DIR, 'genshin.db')


@contextmanager
def get_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable dict-like access
    try:
        yield conn
    finally:
        conn.close()


def init_database():
    """Initialize the database with required tables."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Vocabulary table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vocabulary (
                id TEXT PRIMARY KEY,
                mandarin TEXT NOT NULL,
                pinyin TEXT,
                english TEXT NOT NULL,
                literal_breakdown TEXT,
                game_definition TEXT,
                real_world_definition TEXT,
                tags TEXT,  -- JSON array
                region TEXT,
                hsk INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Translation history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS translation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_text TEXT NOT NULL,
                translated_text TEXT NOT NULL,
                confidence REAL,
                source TEXT,  -- 'ocr', 'manual', etc.
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Translation cache table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS translation_cache (
                text_hash TEXT PRIMARY KEY,
                original_text TEXT NOT NULL,
                translated_text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for faster lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_vocab_mandarin ON vocabulary(mandarin)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_vocab_english ON vocabulary(english)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_cache_hash ON translation_cache(text_hash)')
        
        conn.commit()
        print(f"Database initialized: {DB_PATH}")


def import_vocabulary_from_list(vocab_list: List[Dict[str, Any]]) -> int:
    """
    Import vocabulary from a Python list (like VOCABULARY).
    
    Args:
        vocab_list: List of vocabulary dictionaries
        
    Returns:
        Number of terms imported
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        count = 0
        
        for term in vocab_list:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO vocabulary 
                    (id, mandarin, pinyin, english, literal_breakdown, 
                     game_definition, real_world_definition, tags, region, hsk)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    term.get('id', ''),
                    term.get('mandarin', ''),
                    term.get('pinyin', ''),
                    term.get('english', ''),
                    term.get('literal_breakdown', ''),
                    term.get('game_definition', ''),
                    term.get('real_world_definition', ''),
                    json.dumps(term.get('tags', [])),
                    term.get('region', ''),
                    term.get('hsk')
                ))
                count += 1
            except Exception as e:
                print(f"Failed to import term {term.get('id')}: {e}")
        
        conn.commit()
        return count


def search_vocabulary(text: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search vocabulary for matching terms.
    
    Args:
        text: Chinese text to search for
        limit: Maximum results to return
        
    Returns:
        List of matching vocabulary entries
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Search for exact or substring matches
        cursor.execute('''
            SELECT * FROM vocabulary 
            WHERE mandarin = ? OR ? LIKE '%' || mandarin || '%'
            ORDER BY LENGTH(mandarin) DESC
            LIMIT ?
        ''', (text, text, limit))
        
        results = []
        for row in cursor.fetchall():
            entry = dict(row)
            entry['tags'] = json.loads(entry.get('tags', '[]'))
            results.append(entry)
        
        return results


def get_all_vocabulary() -> List[Dict[str, Any]]:
    """Get all vocabulary entries."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM vocabulary ORDER BY mandarin')
        
        results = []
        for row in cursor.fetchall():
            entry = dict(row)
            entry['tags'] = json.loads(entry.get('tags', '[]'))
            results.append(entry)
        
        return results


def get_vocabulary_count() -> int:
    """Get total vocabulary count."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM vocabulary')
        return cursor.fetchone()[0]


# Initialize database on import
if not os.path.exists(DB_PATH):
    init_database()
