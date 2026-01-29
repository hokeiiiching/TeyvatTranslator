# -*- coding: utf-8 -*-
"""
OCR Configuration and Diagnostics

Provides:
- Configurable OCR parameters for easy tuning
- Automatic problem detection with suggested fixes
- Performance tracking and analytics
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import json
import os


class OCRProblem(Enum):
    """Types of OCR problems that can be detected."""
    LOW_CONTRAST = "low_contrast"
    TOO_SMALL = "too_small"
    TOO_DARK = "too_dark"
    TOO_BRIGHT = "too_bright"
    NO_TEXT = "no_text"
    NOISE = "noise"
    BLUR = "blur"
    WRONG_LANGUAGE = "wrong_language"
    PARTIAL_TEXT = "partial_text"


@dataclass
class OCRConfig:
    """
    Configurable OCR parameters.
    
    Adjust these values to tune OCR accuracy for different scenarios.
    """
    
    # === Preprocessing Settings ===
    min_height_for_ocr: int = 40          # Auto-scale if smaller
    max_scale_factor: float = 3.0          # Maximum upscaling
    
    # === Thresholding Settings ===
    adaptive_block_size: int = 11          # Block size for adaptive threshold (must be odd)
    adaptive_c: int = 2                    # Constant subtracted from mean
    
    # === Contrast Settings ===
    clahe_clip_limit: float = 2.0          # CLAHE clip limit
    clahe_grid_size: int = 8               # CLAHE grid size
    
    # === Noise Settings ===
    gaussian_blur_kernel: int = 3          # Gaussian blur kernel size
    median_blur_kernel: int = 3            # Median blur kernel size
    
    # === Morphology Settings ===
    dilate_kernel_size: int = 2            # Dilation kernel size
    dilate_iterations: int = 1             # Number of dilation iterations
    
    # === Detection Thresholds ===
    brightness_dark_threshold: int = 128   # Below = dark background
    contrast_warning_threshold: int = 50   # Below = low contrast warning
    std_warning_threshold: float = 5.0     # Below = uniform color warning
    
    
    def save(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'OCRConfig':
        """Load config from JSON file."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                return cls(**data)
        return cls()


@dataclass
class OCRDiagnostic:
    """Result of OCR diagnostic analysis."""
    problems: List[OCRProblem]
    suggestions: List[str]
    confidence: float  # 0-1, how confident we are in the OCR result
    stats: Dict[str, float]
    
    def has_problems(self) -> bool:
        return len(self.problems) > 0
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        if not self.problems:
            return "✅ No problems detected"
        
        lines = [f"⚠️ {len(self.problems)} problem(s) detected:"]
        for problem in self.problems:
            lines.append(f"  - {problem.value.replace('_', ' ').title()}")
        
        if self.suggestions:
            lines.append("\n💡 Suggestions:")
            for suggestion in self.suggestions:
                lines.append(f"  - {suggestion}")
        
        return "\n".join(lines)


class OCRDiagnostics:
    """
    Automatic OCR problem detection and suggestion system.
    
    Analyzes images and OCR results to identify issues and
    suggest configuration changes.
    """
    
    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        self.history: List[Dict] = []
    
    def analyze_image(
        self, 
        gray_image,  # numpy array
        ocr_text: str = "",
        expected_language: str = "chi_sim"
    ) -> OCRDiagnostic:
        """
        Analyze an image and OCR result to detect problems.
        
        Args:
            gray_image: Grayscale numpy array
            ocr_text: The OCR result (if available)
            expected_language: Expected language code
            
        Returns:
            OCRDiagnostic with problems and suggestions
        """
        import numpy as np
        
        problems = []
        suggestions = []
        stats = {}
        
        height, width = gray_image.shape[:2]
        
        # === Analyze Image Statistics ===
        min_val = int(gray_image.min())
        max_val = int(gray_image.max())
        mean_val = float(np.mean(gray_image))
        std_val = float(np.std(gray_image))
        contrast = max_val - min_val
        
        stats = {
            'width': width,
            'height': height,
            'min_pixel': min_val,
            'max_pixel': max_val,
            'mean_brightness': mean_val,
            'std_deviation': std_val,
            'contrast': contrast,
        }
        
        # === Problem: Image Too Small ===
        if height < 20 or width < 50:
            problems.append(OCRProblem.TOO_SMALL)
            suggestions.append(
                f"Image too small ({width}x{height}). "
                f"Try selecting a larger region or increase min_height_for_ocr from {self.config.min_height_for_ocr} to {max(60, height * 3)}"
            )
        
        # === Problem: Low Contrast ===
        if contrast < self.config.contrast_warning_threshold:
            problems.append(OCRProblem.LOW_CONTRAST)
            suggestions.append(
                f"Low contrast ({contrast}). "
                f"Try increasing clahe_clip_limit from {self.config.clahe_clip_limit} to {self.config.clahe_clip_limit + 1.0}"
            )
        
        # === Problem: Uniform Color (No Text) ===
        if std_val < self.config.std_warning_threshold:
            problems.append(OCRProblem.NO_TEXT)
            suggestions.append(
                f"Image appears uniform (std={std_val:.1f}). "
                f"This area likely contains no text. Try selecting a different region."
            )
        
        # === Problem: Too Dark ===
        if mean_val < 15:
            problems.append(OCRProblem.TOO_DARK)
            suggestions.append(
                f"Image very dark (mean={mean_val:.1f}). "
                f"Increase brightness_dark_threshold or check if game window is visible."
            )
        
        # === Problem: Too Bright ===
        if mean_val > 240:
            problems.append(OCRProblem.TOO_BRIGHT)
            suggestions.append(
                f"Image very bright (mean={mean_val:.1f}). "
                f"This may be a blank area. Try selecting text region specifically."
            )
        
        # === Analyze OCR Text (if provided) ===
        if ocr_text:
            chinese_count = sum(1 for c in ocr_text if '\u4e00' <= c <= '\u9fff')
            total_chars = len(ocr_text.replace(' ', '').replace('\n', ''))
            
            stats['total_chars'] = total_chars
            stats['chinese_chars'] = chinese_count
            
            # Problem: No Chinese characters detected (for Chinese OCR)
            if expected_language.startswith('chi') and chinese_count == 0 and total_chars > 0:
                problems.append(OCRProblem.WRONG_LANGUAGE)
                suggestions.append(
                    f"No Chinese characters detected in '{ocr_text[:30]}...'. "
                    f"Check if PaddleOCR ch language pack is installed."
                )
            
            # Problem: Very few characters (partial recognition)
            if 0 < total_chars < 3:
                problems.append(OCRProblem.PARTIAL_TEXT)
                suggestions.append(
                    f"Only {total_chars} character(s) detected. "
                    f"Try different PSM modes or adjust threshold values."
                )
        
        # === Calculate Confidence ===
        confidence = 1.0
        confidence -= len(problems) * 0.15  # Each problem reduces confidence
        confidence = max(0.0, min(1.0, confidence))
        
        stats['confidence'] = confidence
        
        # Store in history for trend analysis
        self.history.append({
            'stats': stats,
            'problems': [p.value for p in problems],
            'ocr_text': ocr_text[:100] if ocr_text else ""
        })
        
        return OCRDiagnostic(
            problems=problems,
            suggestions=suggestions,
            confidence=confidence,
            stats=stats
        )
    
    def get_recommended_config_changes(self) -> Dict[str, any]:
        """
        Analyze history to recommend config changes.
        
        Returns dict of parameter -> suggested value
        """
        if len(self.history) < 5:
            return {}
        
        recommendations = {}
        
        # Count problem frequency
        problem_counts = {}
        for entry in self.history[-20:]:  # Last 20 frames
            for problem in entry['problems']:
                problem_counts[problem] = problem_counts.get(problem, 0) + 1
        
        # Make recommendations based on common problems
        if problem_counts.get('low_contrast', 0) > 3:
            recommendations['clahe_clip_limit'] = self.config.clahe_clip_limit + 0.5
            
        if problem_counts.get('too_small', 0) > 3:
            recommendations['min_height_for_ocr'] = self.config.min_height_for_ocr + 20
            
        if problem_counts.get('noise', 0) > 3:
            recommendations['median_blur_kernel'] = 5
        
        return recommendations
    
    def print_analysis(self, diagnostic: OCRDiagnostic) -> None:
        """Print diagnostic analysis to console."""
        print(f"\n{'='*60}")
        print("  🔍 OCR DIAGNOSTIC ANALYSIS")
        print(f"{'='*60}")
        
        print(f"\n  📊 IMAGE STATISTICS:")
        for key, value in diagnostic.stats.items():
            if isinstance(value, float):
                print(f"     {key}: {value:.2f}")
            else:
                print(f"     {key}: {value}")
        
        print(f"\n  📈 CONFIDENCE: {diagnostic.confidence*100:.0f}%")
        
        print(f"\n  {diagnostic.get_summary()}")
        print(f"{'='*60}")


# === Global config instance ===
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'ocr_config.json')
ocr_config = OCRConfig.load(CONFIG_PATH)
ocr_diagnostics = OCRDiagnostics(ocr_config)
