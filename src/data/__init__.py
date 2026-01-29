# -*- coding: utf-8 -*-
"""
Data Package

Contains vocabulary database, type definitions, and context metadata
for context-aware Genshin Impact translations.
"""

from .vocabulary import VOCABULARY
from .types import Term, TermTag, Region, GameContext, HSKLevel
from .contexts import CONTEXTS, REGIONS, get_context, get_region

__all__ = [
    'VOCABULARY',
    'Term',
    'TermTag',
    'Region',
    'GameContext',
    'HSKLevel',
    'CONTEXTS',
    'REGIONS',
    'get_context',
    'get_region',
]
