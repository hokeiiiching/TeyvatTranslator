# TeyvatTranslator

<div align="center">

**Learn Mandarin through Genshin Impact**

I wanted to improve my spoken and written (simplified) (and now traditional!) chinese.
I like playing Genshin Impact.

So I built this.

Real-time translation overlay with pinyin and vocabulary learning.

Non-GitHub download page: https://teyvattranslator.vercel.app/

</div>

---

## Download

**[Download TeyvatTranslator (Windows)](https://github.com/kaseyho/TeyvatTranslator/releases/latest)**

### How to Use

1. **Download** `TeyvatTranslator-Setup.exe` from the link above
2. **Run** the installer and follow the prompts
3. **Launch** from Start Menu or Desktop shortcut
4. Choose Simplified or Traditional Chinese, select the Genshin Impact window, and click **Start Translation**

> First launch downloads translation models (~300MB) and may take a while. OCR
> model assets are also downloaded on first use. Simplified and Traditional
> share the same verified Chinese OCR profile, so this download happens once.

---

## Features

- **Live Translation**: Real-time OCR of game dialogue
- **Chinese Script Support**: Supports Simplified and Traditional Chinese game text
- **Pinyin Display**: See pronunciation for every character
- **Genshin Vocabulary**: 700+ game-specific terms with context
- **Offline Mode**: Works offline after first download
- **Auto Fallback**: Falls back to Google Translate if needed

Traditional Chinese uses the exact same capture, PP-OCRv6 recognition,
filtering, stability, and translation flow as Simplified Chinese. Captured text
stays Traditional in the overlay; OpenCC only normalizes a private lookup copy
so the curated vocabulary and speaker names work for both scripts.

### Diagnostics

Each launch records a detailed text log, a machine-readable
`pipeline-events.jsonl` timeline, the first three raw/preprocessed captures,
and up to 20 event-time captures around OCR candidates and translation
failures. In the app, click **Open Diagnostics** to inspect them. On Windows
they are stored under `%LOCALAPPDATA%\TeyvatTranslator\diagnostics`.

The diagnostics correlate capture-target selection, region coordinates,
worker/frame/job IDs, raw OCR lines, confidence and bounding boxes, choice
layout evidence, every translation attempt, cache decisions, retry backoff,
overlay updates, timing, and full exception traces. Reproduce the problem once,
then share the newest timestamped session folder indicated by
`latest-session.txt`.

---

## Requirements

- Windows 10/11
- ~2GB disk space (includes translation models)

---

## For Developers

<details>
<summary>Run from source</summary>

```bash
# Install dependencies
pip install -r requirements.txt

# Run
python main.py
```

Traditional Chinese lookup requires `opencc-python-reimplemented`; it is
installed by `requirements.txt` and its conversion dictionaries are included in
the PyInstaller build.

</details>

<details>
<summary>Build installer for distribution</summary>

Requires [Inno Setup 6](https://jrsoftware.org/isdl.php) (free).

```bash
# Build executable + create installer
python build.py --clean --installer
```

Output: `dist/TeyvatTranslator-v1.1.3-Setup.exe`

</details>

---

## Disclaimer

> This is an **UNOFFICIAL** fan-made application. Not affiliated with HoYoverse/COGNOSPHERE. For educational purposes only.

---

## License

MIT — See [LICENSE](LICENSE)
