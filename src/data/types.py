# -*- coding: utf-8 -*-
"""
Type Definitions for Genshin Translator

Python dataclasses and type aliases converted from TypeScript interfaces.
Provides type-safe vocabulary term definitions for the RAG knowledge base.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, List

# Type aliases matching TypeScript definitions
TermTag = Literal[
    'combat',      # Combat-related terms
    'ui',          # User interface terms
    'dialogue',    # Story/dialogue terms
    'character',   # Character names and titles
    'location',    # Place names
    'enemy',       # Enemy names
    'item',        # Items, materials, weapons
    'mechanic'     # Game mechanics
]

Region = Literal[
    'mondstadt',   # City of Freedom
    'liyue',       # Harbor of Stone
    'inazuma',     # Nation of Eternity
    'sumeru',      # Nation of Wisdom
    'fontaine',    # Nation of Justice
    'natlan',      # Nation of War
    'snezhnaya',   # Nation of Ice
    'general'      # Universal terms
]

GameContext = Literal['combat', 'story', 'menu']

HSKLevel = Literal[1, 2, 3, 4, 5, 6]  # Chinese proficiency levels

Difficulty = Literal[1, 2, 3, 4, 5]  # Term difficulty (1=basic, 5=advanced)

ArchonQuest = Literal[
    'prologue',    # Mondstadt arc
    'chapter1',    # Liyue arc
    'chapter2',    # Inazuma arc
    'chapter3',    # Sumeru arc
    'chapter4',    # Fontaine arc
    'chapter5',    # Natlan arc
    'interlude'    # Side chapters
]


@dataclass
class Term:
    """
    Core vocabulary term definition.
    
    Matches the PRD data model for Genshin-specific vocabulary.
    Each term includes Chinese, Pinyin, English, and contextual definitions.
    """
    id: str
    mandarin: str
    pinyin: str
    english: str
    literal_breakdown: str
    game_definition: str
    real_world_definition: str
    tags: List[str]
    difficulty: int
    region: Optional[str] = None
    contexts: Optional[List[str]] = None
    character: Optional[str] = None
    hsk_level: Optional[int] = None
    quest: Optional[str] = None
    is_quote: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary for compatibility with existing code."""
        return {
            'id': self.id,
            'mandarin': self.mandarin,
            'pinyin': self.pinyin,
            'english': self.english,
            'literal_breakdown': self.literal_breakdown,
            'game_definition': self.game_definition,
            'real_world_definition': self.real_world_definition,
            'tags': self.tags,
            'difficulty': self.difficulty,
            'region': self.region,
            'contexts': self.contexts,
            'character': self.character,
            'hsk_level': self.hsk_level,
            'quest': self.quest,
            'is_quote': self.is_quote,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Term':
        """Create Term from dictionary."""
        return cls(
            id=data.get('id', ''),
            mandarin=data.get('mandarin', ''),
            pinyin=data.get('pinyin', ''),
            english=data.get('english', ''),
            literal_breakdown=data.get('literal_breakdown', ''),
            game_definition=data.get('game_definition', ''),
            real_world_definition=data.get('real_world_definition', ''),
            tags=data.get('tags', []),
            difficulty=data.get('difficulty', 1),
            region=data.get('region'),
            contexts=data.get('contexts'),
            character=data.get('character'),
            hsk_level=data.get('hsk_level'),
            quest=data.get('quest'),
            is_quote=data.get('is_quote', False),
        )


@dataclass
class DialogueSample:
    """Sample dialogue for contextual learning."""
    mandarin: str
    english: str
    notes: Optional[str] = None
    region: Optional[str] = None
    tone: Optional[str] = None


@dataclass
class ContextDefinition:
    """Gameplay context configuration."""
    id: str
    label: str
    description: str
    recommended_tags: List[str]
    icon: str


@dataclass
class RegionDefinition:
    """Region definition for Teyvat."""
    id: str
    label: str
    chinese_name: str
    pinyin: str
    description: str
    icon: str
