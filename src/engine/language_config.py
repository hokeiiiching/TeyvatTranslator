# -*- coding: utf-8 -*-
"""
Source language configuration for OCR, translation, and vocabulary lookup.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Tuple


class TraditionalChineseSupportError(RuntimeError):
    """Raised when required Traditional Chinese conversion data is unavailable."""


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
    except Exception as exc:
        raise TraditionalChineseSupportError(
            "Traditional Chinese support requires OpenCC and its conversion "
            "dictionaries. Reinstall TeyvatTranslator. Developers can run "
            "'pip install opencc-python-reimplemented>=0.1.7'."
        ) from exc


@lru_cache(maxsize=len(SOURCE_LANGUAGES))
def validate_source_language_support(code: str) -> None:
    """Fail early if the selected source language is not fully operational."""
    language = get_source_language(code)
    if not language.normalize_lookup:
        return

    try:
        probe = _get_t2s_converter().convert("\u937e\u96e2")
    except TraditionalChineseSupportError:
        raise
    except Exception as exc:
        raise TraditionalChineseSupportError(
            "Traditional Chinese conversion data could not be loaded. "
            "Reinstall TeyvatTranslator."
        ) from exc

    if probe != "\u949f\u79bb":
        raise TraditionalChineseSupportError(
            "Traditional Chinese conversion failed its startup check. "
            "Reinstall TeyvatTranslator."
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

    validate_source_language_support(source_lang)
    try:
        return _get_t2s_converter().convert(text)
    except TraditionalChineseSupportError:
        raise
    except Exception as exc:
        raise TraditionalChineseSupportError(
            "Traditional Chinese text could not be normalized for vocabulary lookup."
        ) from exc


def make_cache_key(source_lang: str, purpose: str, text: str) -> str:
    """Build a source-language-aware translation cache key."""
    get_source_language(source_lang)
    if not purpose:
        raise ValueError("Cache key purpose is required")
    return f"{source_lang}:{purpose}:{text or ''}"
