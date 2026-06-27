# -*- coding: utf-8 -*-
"""
Source language configuration for OCR, translation, and vocabulary lookup.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Tuple


@dataclass(frozen=True)
class SourceLanguage:
    """Supported source language metadata."""

    code: str
    label: str
    paddle_lang: str
    translate_locale: str
    normalize_lookup: bool = False


SOURCE_LANGUAGES: Dict[str, SourceLanguage] = {
    "chi_sim": SourceLanguage(
        code="chi_sim",
        label="Chinese (Simplified)",
        paddle_lang="ch",
        translate_locale="zh-CN",
        normalize_lookup=False,
    ),
    "chi_tra": SourceLanguage(
        code="chi_tra",
        label="Chinese (Traditional)",
        paddle_lang="chinese_cht",
        translate_locale="zh-TW",
        normalize_lookup=True,
    ),
}

SOURCE_LANGUAGE_OPTIONS: Tuple[Tuple[str, str], ...] = tuple(
    (language.label, language.code) for language in SOURCE_LANGUAGES.values()
)


def get_source_language(code: str) -> SourceLanguage:
    """Return supported source language metadata."""
    try:
        return SOURCE_LANGUAGES[code]
    except KeyError as exc:
        raise ValueError(f"Unsupported source language: {code}") from exc


def get_paddle_lang(code: str) -> str:
    """Return the PaddleOCR language code for an app source language."""
    return get_source_language(code).paddle_lang


def get_translate_locale(code: str) -> str:
    """Return the translation locale for an app source language."""
    return get_source_language(code).translate_locale


def get_source_label(code: str) -> str:
    """Return the display label for an app source language."""
    return get_source_language(code).label


@lru_cache(maxsize=1)
def _get_t2s_converter():
    """Lazily load OpenCC for Traditional-to-Simplified conversion."""
    try:
        from opencc import OpenCC

        return OpenCC("t2s")
    except ImportError:
        return None


_T2S_FALLBACK_MAP = str.maketrans(
    {
        "\u937e": "\u949f",
        "\u96e2": "\u79bb",
        "\u6d3e": "\u6d3e",
        "\u8499": "\u8499",
        "\u8aaa": "\u8bf4",
        "\u5192": "\u5192",
        "\u96aa": "\u9669",
        "\u5bb6": "\u5bb6",
        "\u5354": "\u534f",
        "\u6703": "\u4f1a",
        "\u9f8d": "\u9f99",
        "\u98a8": "\u98ce",
        "\u706b": "\u706b",
        "\u6c34": "\u6c34",
        "\u96f7": "\u96f7",
        "\u51b0": "\u51b0",
        "\u5ca9": "\u5ca9",
        "\u8349": "\u8349",
        "\u5143": "\u5143",
        "\u7d20": "\u7d20",
        "\u4e4b": "\u4e4b",
        "\u773c": "\u773c",
        "\u528d": "\u5251",
        "\u5f35": "\u5f20",
        "\u958b": "\u5f00",
        "\u95dc": "\u5173",
        "\u9ede": "\u70b9",
        "\u982d": "\u5934",
        "\u968a": "\u961f",
        "\u9577": "\u957f",
    }
)


def normalize_for_lookup(text: str, source_lang: str) -> str:
    """
    Normalize source text for internal Simplified Chinese vocabulary lookup.

    Display code should keep using the original OCR text. This function exists
    only for matching against the curated Simplified Chinese vocabulary.
    """
    if not text:
        return ""

    language = get_source_language(source_lang)
    if not language.normalize_lookup:
        return text

    converter = _get_t2s_converter()
    if converter is not None:
        return converter.convert(text)

    return text.translate(_T2S_FALLBACK_MAP)


def make_cache_key(source_lang: str, purpose: str, text: str) -> str:
    """Build a source-language-aware translation cache key."""
    get_source_language(source_lang)
    if not purpose:
        raise ValueError("Cache key purpose is required")
    return f"{source_lang}:{purpose}:{text or ''}"
