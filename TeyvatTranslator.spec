# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for TeyvatTranslator
Creates a standalone Windows executable
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect data files from various packages
datas = [
    ('src/data/genshin.db', 'src/data'),
    ('src/data/ocr_config.json', 'src/data'),
    ('src/ui/styles.qss', 'src/ui'),
    ('assets', 'assets'),
]

# Add package data files
datas += collect_data_files('paddleocr')
datas += collect_data_files('pypinyin')
datas += collect_data_files('chromadb')
datas += collect_data_files('sentence_transformers')

# Hidden imports for complex packages
hiddenimports = [
    # PyQt6
    'PyQt6.QtCore',
    'PyQt6.QtGui', 
    'PyQt6.QtWidgets',
    'PyQt6.sip',
    
    # PaddleOCR and PaddlePaddle
    'paddle',
    'paddle.fluid',
    'paddleocr',
    'paddleocr.ppocr',
    'paddleocr.tools',
    'shapely',
    'pyclipper',
    'imgaug',
    'lmdb',
    
    # Transformers / PyTorch
    'transformers',
    'transformers.models.marian',
    'sentencepiece',
    'torch',
    'torch.nn',
    'torch.cuda',
    
    # Sentence Transformers
    'sentence_transformers',
    
    # ChromaDB
    'chromadb',
    'chromadb.config',
    'hnswlib',
    
    # Other dependencies
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
    'pyttsx3',
    'pyperclip',
]

# Collect all submodules for complex packages
hiddenimports += collect_submodules('paddleocr')
hiddenimports += collect_submodules('transformers')
hiddenimports += collect_submodules('sentence_transformers')
hiddenimports += collect_submodules('chromadb')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'tkinter',
        'pytest',
        'jupyter',
        'IPython',
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
    console=False,  # No console window
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
