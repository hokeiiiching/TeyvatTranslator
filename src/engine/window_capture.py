# -*- coding: utf-8 -*-
"""
Window Capture Module

Provides utilities for capturing specific application windows on Windows.
Uses pywin32 to enumerate windows and capture window content.
"""

import ctypes
from typing import Optional, List, Tuple, Dict
from ctypes import wintypes

import numpy as np
from PIL import Image

try:
    import win32gui
    import win32ui
    import win32con
    import win32process
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False
    print("pywin32 not available, window capture disabled")


def get_window_list() -> List[Dict[str, any]]:
    """
    Get a list of all visible windows.
    
    Returns:
        List of dictionaries with window info:
        - hwnd: Window handle
        - title: Window title
        - class_name: Window class name
        - rect: (left, top, right, bottom) coordinates
    """
    if not HAS_WIN32:
        return []
    
    windows = []
    
    def enum_callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:  # Only include windows with titles
                try:
                    rect = win32gui.GetWindowRect(hwnd)
                    class_name = win32gui.GetClassName(hwnd)
                    
                    # Filter out tiny windows (likely system tray icons)
                    width = rect[2] - rect[0]
                    height = rect[3] - rect[1]
                    if width > 100 and height > 100:
                        windows.append({
                            'hwnd': hwnd,
                            'title': title,
                            'class_name': class_name,
                            'rect': rect,
                            'size': (width, height)
                        })
                except:
                    pass
        return True
    
    win32gui.EnumWindows(enum_callback, None)
    
    # Sort by title for easier selection
    windows.sort(key=lambda w: w['title'].lower())
    
    return windows


def find_genshin_window() -> Optional[Dict]:
    """
    Automatically find the Genshin Impact window.
    
    Returns:
        Window info dict if found, None otherwise.
    """
    windows = get_window_list()
    
    # Look for common Genshin Impact window titles
    genshin_keywords = ['genshin impact', '原神', 'genshinimpact']
    
    for window in windows:
        title_lower = window['title'].lower()
        for keyword in genshin_keywords:
            if keyword in title_lower:
                print(f"Found Genshin window: '{window['title']}'")
                return window
    
    return None


def get_dialogue_region(window_width: int, window_height: int) -> Tuple[int, int, int, int]:
    """
    Calculate the dialogue region for Genshin Impact.
    
    Genshin dialogue appears at the bottom center of the screen.
    This returns (x, y, width, height) relative to the window.
    
    Based on analysis of Genshin Impact UI:
    - Character name appears around 73-76% down (centered, above yellow line)
    - Dialogue text is at ~77%-95% of screen height
    - We capture both speaker name AND dialogue
    
    Args:
        window_width: Width of the game window
        window_height: Height of the game window
        
    Returns:
        Tuple of (x, y, width, height) for the dialogue region
    """
    # Capture speaker name + dialogue text
    # Speaker name appears at ~73-76% down, dialogue ends at ~95%
    
    margin_x = int(window_width * 0.08)  # 8% margin on each side
    region_width = window_width - (margin_x * 2)  # 84% of width
    
    # Start at 73% down (to capture speaker name above yellow line), capture until 95% (22% of height)
    region_y = int(window_height * 0.73)
    region_height = int(window_height * 0.22)
    
    return (margin_x, region_y, region_width, region_height)


def capture_window_dialogue(hwnd: int) -> Optional[np.ndarray]:
    """
    Capture only the dialogue region of a window.
    
    Uses screen grab at window position instead of PrintWindow because
    PrintWindow returns black for DirectX/hardware-accelerated games.
    
    Only captures when the target window is in the foreground to prevent
    capturing other windows when user alt-tabs.
    
    Args:
        hwnd: Window handle
        
    Returns:
        NumPy array of dialogue region, or None on error
    """
    if not HAS_WIN32:
        return None
    
    try:
        # Check if target window is the foreground window
        foreground_hwnd = win32gui.GetForegroundWindow()
        if foreground_hwnd != hwnd:
            # Target window is not in focus - skip capture
            return None
        
        # Get window position on screen
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        window_width = right - left
        window_height = bottom - top
        
        # Calculate dialogue region (relative to window)
        rx, ry, rw, rh = get_dialogue_region(window_width, window_height)
        
        # Convert to screen coordinates
        screen_left = left + rx
        screen_top = top + ry
        screen_right = screen_left + rw
        screen_bottom = screen_top + rh
        
        # Use screen grab at window position (works with DirectX games)
        from PIL import ImageGrab
        img = ImageGrab.grab(bbox=(screen_left, screen_top, screen_right, screen_bottom))
        
        if img is None:
            return None
        
        img_np = np.array(img)
        
        return img_np
        
    except Exception as e:
        print(f"Dialogue capture error: {e}")
        import traceback
        traceback.print_exc()
        return None


def capture_window(hwnd: int, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
    """
    Capture the contents of a specific window.
    
    Args:
        hwnd: Window handle from get_window_list()
        region: Optional (x, y, width, height) relative to window.
                If None, captures entire window.
    
    Returns:
        NumPy array of the captured image (RGB format), or None on error.
    """
    if not HAS_WIN32:
        return None
    
    try:
        # Get window dimensions
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        window_width = right - left
        window_height = bottom - top
        
        if window_width <= 0 or window_height <= 0:
            return None
        
        # Get device contexts
        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        
        # Create bitmap
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mfc_dc, window_width, window_height)
        save_dc.SelectObject(bitmap)
        
        # Use PrintWindow for better compatibility with some windows
        # PW_RENDERFULLCONTENT = 0x00000002 for Windows 8.1+
        result = ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 2)
        
        if result == 0:
            # Fallback to BitBlt if PrintWindow fails
            save_dc.BitBlt((0, 0), (window_width, window_height), mfc_dc, (0, 0), win32con.SRCCOPY)
        
        # Convert to numpy array
        bmp_info = bitmap.GetInfo()
        bmp_str = bitmap.GetBitmapBits(True)
        
        img = np.frombuffer(bmp_str, dtype=np.uint8)
        img = img.reshape((bmp_info['bmHeight'], bmp_info['bmWidth'], 4))
        
        # Convert BGRA to RGB
        img = img[:, :, :3][:, :, ::-1]
        
        # Cleanup
        win32gui.DeleteObject(bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)
        
        # Apply region crop if specified
        if region:
            rx, ry, rw, rh = region
            # Ensure region is within bounds
            rx = max(0, min(rx, window_width))
            ry = max(0, min(ry, window_height))
            rw = min(rw, window_width - rx)
            rh = min(rh, window_height - ry)
            
            if rw > 0 and rh > 0:
                img = img[ry:ry+rh, rx:rx+rw]
        
        return img
        
    except Exception as e:
        print(f"Window capture error: {e}")
        return None


def bring_window_to_front(hwnd: int) -> bool:
    """
    Bring a window to the foreground.
    
    Args:
        hwnd: Window handle
        
    Returns:
        True if successful
    """
    if not HAS_WIN32:
        return False
    
    try:
        win32gui.SetForegroundWindow(hwnd)
        return True
    except:
        return False


# Quick test when run directly
if __name__ == "__main__":
    print("\n=== Window Capture Test ===\n")
    
    windows = get_window_list()
    print(f"Found {len(windows)} windows:\n")
    
    for i, w in enumerate(windows[:15]):
        print(f"  {i+1}. [{w['hwnd']}] {w['title'][:50]} ({w['size'][0]}x{w['size'][1]})")
    
    print("\n--- Searching for Genshin ---")
    genshin = find_genshin_window()
    
    if genshin:
        print(f"\nCapturing Genshin window...")
        img = capture_window(genshin['hwnd'])
        if img is not None:
            print(f"  Captured: {img.shape}")
            # Save test capture
            Image.fromarray(img).save("test_capture.png")
            print("  Saved to test_capture.png")
    else:
        print("  Genshin Impact not running")
