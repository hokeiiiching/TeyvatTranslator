# Traditional Chinese Support Roadmap

This roadmap is the required implementation plan for adding Traditional Chinese support. Future AI agents must follow it strictly, in order, and must not ship a partial implementation that only changes the UI label or only switches the OCR model.

## Goal

Support Genshin Impact running in Traditional Chinese while preserving the existing Simplified Chinese workflow.

The finished feature must:

- Let the user choose Simplified or Traditional Chinese before starting capture.
- Use the correct PaddleOCR recognition model for the selected source script.
- Display the original OCR text exactly as captured.
- Generate pinyin for the captured Chinese text.
- Translate Traditional Chinese dialogue to English.
- Preserve vocabulary, speaker, descriptor, and RAG matching by normalizing Traditional text to Simplified for internal lookup.
- Keep Simplified Chinese behavior unchanged.

## Non-Goals

Do not add Japanese, Korean, or other language support as part of this work.

Do not convert the curated vocabulary database to Traditional Chinese manually.

Do not rewrite the OCR, RAG, or translation architecture unless a smaller compatibility layer cannot solve the problem.

Do not fix unrelated mojibake or copy text issues unless they block this feature directly.

## Pre-Implementation Findings

Relevant existing code:

- `src/ui/main_window.py`
  - `OCR_LANGUAGE_CODES` and `TRANSLATE_LANGUAGE_CODES` already contain `"Chinese (Traditional)": "chi_tra"`.
  - `_create_translate_tab()` still presents the app as fixed to Simplified Chinese.
  - `_on_start_translation()` hard-codes `from_lang = "chi_sim"` and `to_lang = "eng"`.
- `src/engine/ocr.py`
  - `OCRWorker.OCR_TO_TRANSLATE` already maps `chi_tra` to `zh-TW`.
  - Global PaddleOCR initialization currently uses `PaddleOCR(lang='ch', ...)`, which is Simplified Chinese oriented.
  - `get_paddle_ocr()` returns one shared OCR instance, so it cannot safely serve both Simplified and Traditional models without refactoring.
  - Speaker detection, descriptor detection, pinyin generation, RAG lookup, translation cache, and display update all happen in `_process_translation()`.
- `src/engine/rag.py`, `src/engine/context.py`, and `src/data/database.py`
  - Vocabulary lookup is based on the `mandarin` field, which is stored in Simplified Chinese.
  - Traditional OCR text will not reliably match these entries without normalization.
- `src/engine/translator.py`
  - MarianMT uses `Helsinki-NLP/opus-mt-zh-en`.
  - Google Translate fallback auto-detects the source language and targets English.

## Required External Capability

Traditional-to-Simplified normalization is required for internal lookup. Prefer adding:

```text
opencc-python-reimplemented>=0.1.7
```

Use it via `OpenCC('t2s')` for Traditional-to-Simplified normalization. If a different OpenCC package is chosen, document the import and conversion API in this file before implementation continues.

PaddleOCR language codes must be verified against the installed PaddleOCR version before coding the model selector. Expected mapping:

```python
{
    "chi_sim": "ch",
    "chi_tra": "chinese_cht",
}
```

If the installed version rejects `chinese_cht`, stop and document the actual supported language code before changing production code.

## Strict Implementation Phases

### Phase 1: Add Language Configuration

Create a small source-language configuration layer instead of scattering string literals.

Recommended file:

- `src/engine/language_config.py`

It should define:

- A source language dataclass or typed dict with:
  - app code: `chi_sim` or `chi_tra`
  - display label
  - PaddleOCR language code
  - Google/source locale code if needed
  - whether internal lookup normalization is required
- `get_source_language(code: str)`
- `get_paddle_lang(code: str)`
- `normalize_for_lookup(text: str, source_lang: str)`

Rules:

- `normalize_for_lookup(text, "chi_sim")` must return `text` unchanged.
- `normalize_for_lookup(text, "chi_tra")` must convert Traditional Chinese to Simplified Chinese.
- The function must preserve punctuation and line breaks as much as OpenCC allows.
- The normalization function must be deterministic and safe for empty strings.

### Phase 2: Update Dependencies

Add the chosen OpenCC dependency to `requirements.txt`.

Do not add heavy conversion libraries or online services for script conversion.

### Phase 3: Refactor PaddleOCR Instance Management

Update `src/engine/ocr.py` so OCR instances are keyed by PaddleOCR language code.

Current behavior has one global instance:

```python
_global_paddle_ocr = PaddleOCR(lang='ch', ...)
```

Required behavior:

- Store OCR instances in a dictionary keyed by PaddleOCR language code.
- Keep lazy loading.
- Keep warmup behavior.
- Keep existing PaddleOCR performance flags:
  - `use_doc_orientation_classify=False`
  - `use_doc_unwarping=False`
  - `use_textline_orientation=False`
- Pass `self.from_lang` through to the OCR getter.
- The selected source language must determine the PaddleOCR model.

Recommended function shape:

```python
def get_paddle_ocr(source_lang: str = "chi_sim"):
    paddle_lang = get_paddle_lang(source_lang)
    ...
```

Important:

- Do not silently reuse a Simplified OCR instance for Traditional Chinese.
- Do not break `preload_ocr()`. Either preload only the default Simplified model or update it to accept a source language.
- If multiple OCR models are loaded, make thread locking language-aware.

### Phase 4: Add Source Language UI

Update `src/ui/main_window.py`.

Required UI behavior:

- Add a source-language selector in the Translate tab.
- Supported source options for this feature:
  - Chinese (Simplified)
  - Chinese (Traditional)
- Target language remains English.
- Default source language remains Chinese (Simplified).
- The status text or language info must reflect the selected source language.
- `_on_start_translation()` must read the selected source language and pass its code to `OCRWorker`.

Rules:

- Do not expose unsupported languages already present in the old dictionaries unless they actually work end to end.
- Do not leave the old hard-coded Simplified label in place.
- Keep the existing capture workflow intact.

### Phase 5: Normalize Internal Lookup Text

Update `_process_translation()` in `src/engine/ocr.py` carefully.

The display text must remain the original OCR text. Internal matching should use normalized Simplified text when the source is Traditional.

Required working variables:

- `text_original`: full OCR text as captured.
- `text_lookup`: normalized-for-lookup version of `text_original`.
- `speaker_original`: displayed speaker name.
- `speaker_lookup`: normalized speaker name for vocabulary/RAG matching.
- `descriptor_original`: displayed descriptor.
- `descriptor_lookup`: normalized descriptor for translation/cache if needed.
- `dialogue_original`: displayed dialogue.
- `dialogue_lookup`: normalized dialogue for vocabulary/RAG matching.

Rules:

- Speaker and descriptor detection may run on original text, but RAG exact matching must run on lookup text.
- Pinyin should be generated from original displayed text unless testing proves pypinyin is more reliable after normalization. If pinyin is generated from lookup text for accuracy, document that choice in the code comment and tests.
- `translate_window.update_translation()` must receive the original display dialogue, not the normalized lookup dialogue.
- Vocabulary translations may come from matches found with normalized lookup text.
- Any context fields that are visibly displayed as Chinese should use original text where appropriate.
- Any exact-match comparison against `match.get('mandarin')` must compare to the lookup form.

### Phase 6: Fix Translation Cache Keys

The current cache uses only text. Traditional and Simplified forms can normalize to the same lookup text but differ in display text and possibly source translation path.

Update cache key behavior in `src/engine/ocr.py`:

- Include source language in translation cache keys.
- Keep descriptor and dialogue caches distinct if their behavior differs.
- Avoid cross-polluting `chi_sim` and `chi_tra` cached results unless the cache is explicitly keyed by normalized text and source language.

Recommended cache key format:

```text
{source_lang}:{purpose}:{text}
```

Examples:

- `chi_tra:dialogue:<traditional dialogue text>`
- `chi_tra:descriptor:<traditional descriptor text>`
- `chi_sim:dialogue:<simplified dialogue text>`

### Phase 7: Translation Path Review

Keep `Translator(target_lang='en')` unless implementation discovers a real failure translating Traditional Chinese.

Rules:

- MarianMT may accept both Simplified and Traditional Chinese. Test this before changing the translation backend.
- Google fallback already uses source auto-detection, so do not force a source language unless a test shows auto-detection is wrong.
- If normalizing Traditional to Simplified before MarianMT improves offline translation, make that an explicit, tested branch and keep Google fallback on original text unless evidence says otherwise.

### Phase 8: Tests

There is no obvious existing test suite. Add one.

Recommended structure:

- `tests/test_language_config.py`
- `tests/test_traditional_lookup.py`

Minimum tests:

- `normalize_for_lookup("", "chi_tra") == ""`
- Simplified text is unchanged.
- A Traditional term such as `\u937e\u96e2` normalizes to the Simplified vocabulary form `\u949f\u79bb`.
- A Traditional phrase containing punctuation and line breaks keeps the same rough structure after normalization.
- `get_paddle_lang("chi_sim") == "ch"`.
- `get_paddle_lang("chi_tra") == "chinese_cht"` or the verified installed-code equivalent.
- RAG or keyword lookup can find a Simplified vocabulary entry when given normalized Traditional input.
- Cache key construction separates `chi_sim` and `chi_tra`.

If GUI testing is difficult, keep UI tests light and verify the selected combo value is passed to `OCRWorker` with a small unit or integration seam.

### Phase 9: Manual Verification

Run these checks before declaring the feature complete:

1. Start the app.
2. Confirm the default source language is Chinese (Simplified).
3. Start capture with Simplified selected and verify existing behavior still works.
4. Switch source language to Chinese (Traditional).
5. Confirm logs show PaddleOCR using the Traditional model.
6. Capture Traditional Chinese Genshin dialogue.
7. Confirm displayed Chinese remains Traditional.
8. Confirm pinyin appears above characters.
9. Confirm English translation appears.
10. Confirm speaker lookup works for known names after normalization.
11. Confirm vocabulary exact matches work for known terms after normalization.
12. Confirm translation cache entries do not overwrite Simplified entries.

## Acceptance Criteria

The feature is complete only when all of the following are true:

- The user can choose Simplified or Traditional Chinese from the main window.
- Simplified remains the default and behaves as before.
- Traditional uses a Traditional-capable PaddleOCR model.
- Traditional OCR text is displayed unchanged.
- Pinyin and English translation are shown for Traditional text.
- Known Traditional speaker names and short vocabulary terms match existing Simplified vocabulary through normalization.
- Translation cache keys are source-language aware.
- Automated tests cover normalization, PaddleOCR language mapping, lookup behavior, and cache key separation.
- README or architecture docs mention Traditional Chinese support and any extra dependency.

## Implementation Warnings

Do not treat `chi_tra` as only a Google Translate locale. OCR and lookup must also know about it.

Do not change the vocabulary schema unless normalization proves insufficient.

Do not convert the display text to Simplified. Users choosing Traditional should see Traditional.

Do not remove `chi_sim` fast paths or make Simplified pay a heavy conversion cost.

Do not rely on visual manual testing alone. The normalization and lookup behavior must have tests.

Do not leave old comments saying the app is fixed to Simplified Chinese once this is implemented.

## Suggested Commit Breakdown

1. Add language config and OpenCC dependency with tests.
2. Refactor PaddleOCR model selection by source language.
3. Add UI selector and pass selected source language to `OCRWorker`.
4. Normalize lookup paths in OCR processing and RAG/speaker matching.
5. Update cache keying and tests.
6. Update docs and perform manual verification.

Each commit should keep the app runnable. If a phase requires a temporary compatibility shim, document it in the commit message and remove it before final acceptance.
