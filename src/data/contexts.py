# -*- coding: utf-8 -*-
"""
Context and Region Definitions

Gameplay contexts and Teyvat region metadata for the Genshin Translator.
Converted from TypeScript contexts.ts.
"""

from typing import Dict, List
from .types import ContextDefinition, RegionDefinition


# Gameplay contexts for passive learning mode
CONTEXTS: Dict[str, ContextDefinition] = {
    'combat': ContextDefinition(
        id='combat',
        label='Combat',
        description='Fighting enemies, using abilities, exploring',
        recommended_tags=['combat', 'enemy', 'mechanic'],
        icon=''
    ),
    'story': ContextDefinition(
        id='story',
        label='Story',
        description='Quests, dialogues, cutscenes',
        recommended_tags=['dialogue', 'character', 'location'],
        icon=''
    ),
    'menu': ContextDefinition(
        id='menu',
        label='Menu',
        description='Inventory, character builds, settings',
        recommended_tags=['ui', 'item'],
        icon=''
    ),
}


# Teyvat region definitions
REGIONS: Dict[str, RegionDefinition] = {
    'general': RegionDefinition(
        id='general',
        label='All Regions',
        chinese_name='通用',
        pinyin='tōng yòng',
        description='Terms used across all of Teyvat',
        icon=''
    ),
    'mondstadt': RegionDefinition(
        id='mondstadt',
        label='Mondstadt',
        chinese_name='蒙德',
        pinyin='Méng dé',
        description='City of Freedom, European-inspired',
        icon=''
    ),
    'liyue': RegionDefinition(
        id='liyue',
        label='Liyue',
        chinese_name='璃月',
        pinyin='Lí yuè',
        description='Harbor of Stone and Contracts, Chinese-inspired',
        icon=''
    ),
    'inazuma': RegionDefinition(
        id='inazuma',
        label='Inazuma',
        chinese_name='稻妻',
        pinyin='Dào qī',
        description='Nation of Eternity, Japanese-inspired',
        icon=''
    ),
    'sumeru': RegionDefinition(
        id='sumeru',
        label='Sumeru',
        chinese_name='须弥',
        pinyin='Xū mí',
        description='Nation of Wisdom, Middle Eastern/South Asian-inspired',
        icon=''
    ),
    'fontaine': RegionDefinition(
        id='fontaine',
        label='Fontaine',
        chinese_name='枫丹',
        pinyin='Fēng dān',
        description='Nation of Justice, French-inspired',
        icon=''
    ),
    'natlan': RegionDefinition(
        id='natlan',
        label='Natlan',
        chinese_name='纳塔',
        pinyin='Nà tǎ',
        description='Nation of War, Latin American-inspired',
        icon=''
    ),
    'snezhnaya': RegionDefinition(
        id='snezhnaya',
        label='Snezhnaya',
        chinese_name='至冬',
        pinyin='Zhì dōng',
        description='Nation of the Tsaritsa, Russian-inspired',
        icon=''
    ),
}


def get_context(context_id: str) -> ContextDefinition:
    """Get context definition by ID."""
    return CONTEXTS.get(context_id, CONTEXTS['story'])


def get_region(region_id: str) -> RegionDefinition:
    """Get region definition by ID."""
    return REGIONS.get(region_id, REGIONS['general'])


def get_all_regions() -> List[RegionDefinition]:
    """Get all region definitions."""
    return list(REGIONS.values())


def get_all_contexts() -> List[ContextDefinition]:
    """Get all context definitions."""
    return list(CONTEXTS.values())
