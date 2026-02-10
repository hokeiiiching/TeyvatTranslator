# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for TeyvatTranslator
Creates a standalone Windows executable
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect PaddlePaddle's dynamically-loaded DLLs (mklml.dll, mkldnn.dll, etc.)
# These are loaded at runtime via ctypes and PyInstaller won't detect them automatically.
import importlib
_paddle_spec = importlib.util.find_spec('paddle')
_paddle_libs_dir = os.path.join(os.path.dirname(_paddle_spec.origin), 'libs') if _paddle_spec else None
paddle_binaries = []
if _paddle_libs_dir and os.path.isdir(_paddle_libs_dir):
    for f in os.listdir(_paddle_libs_dir):
        if f.endswith('.dll'):
            paddle_binaries.append((os.path.join(_paddle_libs_dir, f), '.'))
    print(f"Collected {len(paddle_binaries)} PaddlePaddle DLLs from {_paddle_libs_dir}")

# Collect data files from packages that need them
datas = [
    ('src/data/genshin.db', 'src/data'),
    ('src/data/ocr_config.json', 'src/data'),
    ('src/ui/styles.qss', 'src/ui'),
    ('assets', 'assets'),
]

# PaddleOCR downloads models to user cache at runtime - no need to bundle model data
# But PaddleX needs its .version file
# chromadb / sentence_transformers are optional (RAG falls back to keyword search)
datas += collect_data_files('pypinyin')
datas += collect_data_files('paddlex')

# Hidden imports - only what the app actually uses
hiddenimports = [
    # PyQt6
    'PyQt6.QtCore',
    'PyQt6.QtGui', 
    'PyQt6.QtWidgets',
    'PyQt6.sip',
    
    # PaddleOCR + PaddlePaddle
    'paddleocr',
    'paddle',
    'paddle.fluid',
    
    # Transformers - only MarianMT for translation
    'transformers',
    'transformers.models.marian',
    'sentencepiece',
    'torch',
    'torch.nn',
    'torch.cuda',
    
    # Other direct dependencies
    'pypinyin',
    'pypinyin.style',
    'deep_translator',
    'google.generativeai',
    'PIL',
    'cv2',
    'numpy',
    'win32gui',
    'win32ui',
    'win32con',
    'win32api',
    'win32process',
    'pyttsx3',
    'pyperclip',
    'shapely',
    'pyclipper',
    'sacremoses',
]

# Collect submodules for packages with dynamic imports
hiddenimports += collect_submodules('paddleocr')
hiddenimports += collect_submodules('paddle')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=paddle_binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # GUI frameworks we don't use
        'matplotlib',
        'tkinter',
        
        # Testing / dev tools
        'pytest',
        'jupyter',
        'IPython',
        
        # ML frameworks we don't use
        'tensorflow',
        'tensorboard',
        'keras',
        'jax',
        'flax',
        
        # Heavy optional deps (not needed — RAG falls back to keyword search)
        'chromadb',
        'sentence_transformers',
        'hnswlib',

        # Test/dev packages
        'transformers.testing_utils',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TeyvatTranslator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Temporarily enabled for debugging — set False for release
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.ico' if os.path.exists('assets/icon.ico') else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='TeyvatTranslator',
)
