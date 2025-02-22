#!/usr/bin/env python3
"""
TIFF Analysis Tool

Analyzes TIFF images for potential corruption patterns and glitches.
Detection thresholds can be modified via command-line parameters.
Use the "--dynamic" flag to compute thresholds dynamically.
If glitches are detected, heatmaps highlighting glitch regions will be generated.
"""

import os
import sys
import logging
import argparse
from PIL import Image, ImageFile, UnidentifiedImageError
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Tuple, List
import platform
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Allow PIL to load large images
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Default Constants for detection thresholds (used when --dynamic is NOT specified) ---
BASE_THRESHOLD_FACTOR: float = 0.15
LOCAL_THRESHOLD_STD_FACTOR: float = 0.5
ANOMALY_RATIO_THRESHOLD: float = 0.3
SIGNIFICANCE_MULTIPLIER: float = 2.0
GROUP_SEVERITY_MULTIPLIER: float = 3.0
MAX_GROUP_SIZE: int = 20
MIN_GROUP_LENGTH: int = 2
MAX_ALIGNMENT_DIFF: int = 5
REGULARITY_THRESHOLD: float = 0.7
REPEATED_PATTERN_RATIO: float = 0.1
CLUSTER_GAP_THRESHOLD: float = 0.05
# -------------------------------------------------------------------------------

def compute_dynamic_thresholds(luminance: np.ndarray, axis: int) -> Dict[str, float]:
    """
    Computes dynamic thresholds for glitch detection along the given axis.
    
    :param luminance: 2D array of image luminance.
    :param axis: Axis along which to compute differences (0 for rows, 1 for columns).
    :return: A dictionary containing dynamic thresholds.
    
    The function computes the absolute differences between adjacent pixels
    and uses, for example, the 90th percentile of these differences as the base threshold.
    This approach is inspired by adaptive thresholding methods such as Otsu's method.
    """
    if axis == 0:
        diffs = np.abs(luminance[1:, :] - luminance[:-1, :])
    else:
        diffs = np.abs(luminance[:, 1:] - luminance[:, :-1])
    
    flat_diffs = diffs.flatten()
    dynamic_base_threshold = np.percentile(flat_diffs, 95)
    dynamic_std = np.std(flat_diffs)
    
    return {
        'base_threshold': dynamic_base_threshold,
        'std': dynamic_std,
    }

def get_tiff_info(img: Image.Image) -> Dict[str, Any]:
    """
    Extract detailed information from a TIFF image.
    
    This function retrieves basic metadata such as format, mode, dimensions,
    as well as TIFF-specific tags like BitsPerSample, Compression, PhotometricInterpretation, and DPI.
    
    Source: Adobe TIFF Specification:
    https://www.adobe.io/open/standards/TIFF.html
    """
    info: Dict[str, Any] = {}
    try:
        info['format'] = img.format
        info['mode'] = img.mode
        info['size'] = img.size
        if hasattr(img, 'tag'):
            tags = img.tag
            if 258 in tags:
                info['bits_per_sample'] = tags[258]
            if 259 in tags:
                compression_codes = {
                    1: "Uncompressed",
                    2: "CCITT 1D",
                    3: "CCITT Group 3",
                    4: "CCITT Group 4",
                    5: "LZW",
                    6: "Old JPEG",
                    7: "JPEG",
                    8: "Adobe Deflate",
                    32773: "PackBits"
                }
                comp_value = tags[259][0]
                info['compression'] = compression_codes.get(comp_value, f"Unknown ({comp_value})")
            if 262 in tags:
                photo_codes = {
                    0: "WhiteIsZero",
                    1: "BlackIsZero",
                    2: "RGB",
                    3: "Palette",
                    4: "Mask",
                    5: "CMYK",
                    6: "YCbCr",
                    8: "CIELab"
                }
                photo_value = tags[262][0]
                info['photometric'] = photo_codes.get(photo_value, f"Unknown ({photo_value})")
            if 296 in tags:
                unit_codes = {1: "None", 2: "Inches", 3: "Centimeters"}
                unit_value = tags[296][0]
                info['resolution_unit'] = unit_codes.get(unit_value, f"Unknown ({unit_value})")
            if 282 in tags and 283 in tags:
                info['dpi'] = (float(tags[282][0][0]) / float(tags[282][0][1]),
                               float(tags[283][0][0]) / float(tags[283][0][1]))
    except Exception as e:
        info['error'] = str(e)
        logger.error("Error extracting TIFF info: %s", e)
    return info

def compute_luminance(img_array: np.ndarray) -> np.ndarray:
    """
    Compute the luminance of an image array.
    
    For an RGB image, the luminance is computed using:
    
        Luminance = 0.2989 × Red + 0.5870 × Green + 0.1140 × Blue
    
    These coefficients come from the ITU-R BT.601 standard and are used because
    the human eye is more sensitive to green, moderately to red, and least to blue.
    For a single-channel image, the original array is returned.
    
    Sources:
      - ITU-R BT.601 Standard: https://en.wikipedia.org/wiki/Rec._601
      - Wikipedia: YUV: https://en.wikipedia.org/wiki/YUV
    """
    if img_array.ndim == 3 and img_array.shape[2] >= 3:
        return 0.2989 * img_array[:, :, 0] + 0.5870 * img_array[:, :, 1] + 0.1140 * img_array[:, :, 2]
    return img_array

def detect_glitches(luminance: np.ndarray, axis: int, use_dynamic: bool = False) -> Dict[str, Any]:
    """
    Detect glitches (abnormal changes in brightness) along the specified axis.
    
    :param luminance: 2D array representing the image's luminance.
    :param axis: 0 to detect horizontal glitches (rows), 1 for vertical glitches (columns).
    :param use_dynamic: If True, use dynamic thresholding; otherwise, use fixed thresholds.
    
    The function compares differences between adjacent rows or columns with a threshold.
    If a sufficient percentage of pixels exceed the threshold, that row/column is flagged.
    Consecutive flagged rows/columns are grouped into a glitch region.
    
    This step isolates areas of the image where the brightness changes unexpectedly.
    """
    key_name = "lines" if axis == 0 else "columns"
    glitch_info: Dict[str, Any] = {
        'detected': False,
        key_name: [],
        'count': 0,
        'severity': []
    }
    
    potential_glitches: List[Tuple[int, float]] = []
    dim = luminance.shape[axis]
    unit_label = "rows" if axis == 0 else "cols"
    
    if use_dynamic:
        thresholds = compute_dynamic_thresholds(luminance, axis)
        base_threshold = thresholds['base_threshold']
        dynamic_std = thresholds['std']
    else:
        global_mean = np.mean(luminance)
        global_std = np.std(luminance)
        base_threshold = global_mean * BASE_THRESHOLD_FACTOR + (global_std * LOCAL_THRESHOLD_STD_FACTOR)
    
    logger.info("Scanning %s for glitches...", unit_label)
    for i in tqdm(range(1, dim - 1), desc=f"Scanning {unit_label}", unit=unit_label):
        if axis == 0:
            diff_prev = np.abs(luminance[i, :] - luminance[i - 1, :])
            diff_next = np.abs(luminance[i, :] - luminance[i + 1, :])
        else:
            diff_prev = np.abs(luminance[:, i] - luminance[:, i - 1])
            diff_next = np.abs(luminance[:, i] - luminance[:, i + 1])
        
        anomaly_prev = np.mean(diff_prev > base_threshold)
        anomaly_next = np.mean(diff_next > base_threshold)
        
        if anomaly_prev > ANOMALY_RATIO_THRESHOLD or anomaly_next > ANOMALY_RATIO_THRESHOLD:
            max_diff = max(np.max(diff_prev), np.max(diff_next))
            if use_dynamic:
                if max_diff > dynamic_std * SIGNIFICANCE_MULTIPLIER:
                    potential_glitches.append((i, max_diff))
            else:
                if max_diff > global_std * SIGNIFICANCE_MULTIPLIER:
                    potential_glitches.append((i, max_diff))
    
    if potential_glitches:
        logger.info("Analyzing detected glitch regions along %s...", unit_label)
        current_group = [potential_glitches[0][0]]
        current_severity = potential_glitches[0][1]
        for index, severity in potential_glitches[1:]:
            if index - current_group[-1] <= 1:
                current_group.append(index)
                current_severity = max(current_severity, severity)
            else:
                if len(current_group) >= MIN_GROUP_LENGTH:
                    if use_dynamic:
                        if current_severity > dynamic_std * GROUP_SEVERITY_MULTIPLIER:
                            if (max(current_group) - min(current_group) + 1) <= MAX_GROUP_SIZE:
                                glitch_info[key_name].append((min(current_group), max(current_group)))
                                glitch_info['severity'].append(current_severity)
                    else:
                        if current_severity > global_std * GROUP_SEVERITY_MULTIPLIER:
                            if (max(current_group) - min(current_group) + 1) <= MAX_GROUP_SIZE:
                                glitch_info[key_name].append((min(current_group), max(current_group)))
                                glitch_info['severity'].append(current_severity)
                current_group = [index]
                current_severity = severity
        if len(current_group) >= MIN_GROUP_LENGTH:
            if use_dynamic:
                if current_severity > dynamic_std * GROUP_SEVERITY_MULTIPLIER:
                    if (max(current_group) - min(current_group) + 1) <= MAX_GROUP_SIZE:
                        glitch_info[key_name].append((min(current_group), max(current_group)))
                        glitch_info['severity'].append(current_severity)
            else:
                if current_severity > global_std * GROUP_SEVERITY_MULTIPLIER:
                    if (max(current_group) - min(current_group) + 1) <= MAX_GROUP_SIZE:
                        glitch_info[key_name].append((min(current_group), max(current_group)))
                        glitch_info['severity'].append(current_severity)
    
    if glitch_info[key_name]:
        glitch_info['detected'] = True
        glitch_info['count'] = len(glitch_info[key_name])
        sorted_glitches = sorted(zip(glitch_info[key_name], glitch_info['severity']),
                                   key=lambda x: x[1], reverse=True)
        glitch_info[key_name] = [g[0] for g in sorted_glitches]
        glitch_info['severity'] = [g[1] for g in sorted_glitches]
        logger.info("Total %s glitches detected: %d", unit_label, glitch_info['count'])
    
    return glitch_info

def detect_horizontal_glitches(img_array: np.ndarray, use_dynamic: bool = False) -> Dict[str, Any]:
    """
    Detect horizontal glitches (along rows).
    
    Converts the image to grayscale and calls detect_glitches with axis=0.
    """
    luminance = compute_luminance(img_array)
    return detect_glitches(luminance, axis=0, use_dynamic=use_dynamic)

def detect_vertical_glitches(img_array: np.ndarray, use_dynamic: bool = False) -> Dict[str, Any]:
    """
    Detect vertical glitches (along columns).
    
    Converts the image to grayscale and calls detect_glitches with axis=1.
    """
    luminance = compute_luminance(img_array)
    return detect_glitches(luminance, axis=1, use_dynamic=use_dynamic)

def analyze_corruption_pattern(img_array: np.ndarray, glitch_lines: List[Tuple[int, int]], orientation: str = 'horizontal') -> Dict[str, Any]:
    """
    Analyze the detected glitch regions to determine the probable origin of the corruption.
    
    This function extracts additional features:
      - Glitch center positions and widths,
      - Mean and standard deviation of glitch intensities,
      - A measure of periodicity using autocorrelation of glitch positions.
    
    Heuristic classification:
      - **Buffer Corruption:** If glitches are highly periodic (autocorrelation > 0.8), narrow (width ≤ MAX_ALIGNMENT_DIFF), and have a mean intensity above the overall image mean, it suggests a buffering/caching error.
      - **Memory Corruption:** If glitches are very dark (low mean intensity) with little variation, it may indicate memory corruption.
      - **Write Corruption:** Otherwise, the glitches are attributed to errors during file writing or data transfer.
    
    Debug information is added to help fine-tune parameters.
    
    Sources:
      - ITU-R BT.601 for luminance conversion.
      - Fundamentals of autocorrelation and signal periodicity in [Digital Signal Processing](https://en.wikipedia.org/wiki/Digital_signal_processing).
      - Concepts from "Digital Image Processing" by Gonzalez & Woods.
    """
    pattern_info: Dict[str, Any] = {
        'type': 'unknown',
        'confidence': 0.0,
        'details': [],
        'probable_cause': None
    }
    
    try:
        luminance = compute_luminance(img_array)
        if orientation == 'vertical':
            luminance = luminance.T
        
        # Extract glitch center positions and widths
        glitch_positions = [ (start + end) / 2 for (start, end) in glitch_lines ]
        glitch_widths = [ end - start + 1 for (start, end) in glitch_lines ]
        
        # Compute the average intensity in each glitch region
        glitch_intensities = []
        for (start, end) in glitch_lines:
            region = luminance[start:end+1, :]
            glitch_intensities.append(np.mean(region))
        mean_glitch_intensity = np.mean(glitch_intensities) if glitch_intensities else 0
        std_glitch_intensity = np.std(glitch_intensities) if glitch_intensities else 0
        
        # Compute periodicity using autocorrelation of glitch positions
        if len(glitch_positions) >= 3:
            pos_array = np.array(glitch_positions)
            pos_array = pos_array - pos_array.mean()  # center the data
            autocorr = np.correlate(pos_array, pos_array, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
            periodicity = autocorr[1] if len(autocorr) > 1 else 0
        else:
            periodicity = 0
        
        # Determine if glitches are narrow (well-aligned)
        narrow_glitches = all(width <= MAX_ALIGNMENT_DIFF for width in glitch_widths)
        
        # Add debug details
        pattern_info['details'].append(f"Mean glitch intensity: {mean_glitch_intensity:.2f}")
        pattern_info['details'].append(f"Std glitch intensity: {std_glitch_intensity:.2f}")
        pattern_info['details'].append(f"Periodicity (autocorrelation at lag 1): {periodicity:.2f}")
        pattern_info['details'].append(f"Glitches are narrow: {narrow_glitches}")
        
        overall_mean = np.mean(luminance)
        
        # Heuristic classification
        if periodicity > 0.8 and narrow_glitches and mean_glitch_intensity > overall_mean:
            pattern_info['type'] = 'buffer_corruption'
            pattern_info['confidence'] = 0.85
            pattern_info['probable_cause'] = f"Likely a buffer/cache error during writing ({orientation})."
        elif mean_glitch_intensity < 20 and std_glitch_intensity < 10:
            pattern_info['type'] = 'memory_corruption'
            pattern_info['confidence'] = 0.75
            pattern_info['probable_cause'] = f"Likely a memory corruption error during processing ({orientation})."
        else:
            pattern_info['type'] = 'write_corruption'
            pattern_info['confidence'] = 0.65
            pattern_info['probable_cause'] = f"Likely an error during file writing or data transfer ({orientation})."
        
    except Exception as e:
        pattern_info['details'].append(f"Error analyzing patterns: {str(e)}")
        logger.error("Error in analyze_corruption_pattern: %s", e)
    
    return pattern_info

def analyze_tiff(file_path: str, use_dynamic: bool) -> Dict[str, Any]:
    """
    Perform a complete analysis of a TIFF image.
    
    This function verifies the file, extracts metadata, converts the image to a NumPy array,
    detects horizontal and vertical glitches, computes pixel statistics, and analyzes glitch patterns.
    All results are compiled into a dictionary.
    
    :param file_path: Path to the TIFF file.
    :param use_dynamic: If True, use dynamic threshold computation.
    :return: Dictionary containing the analysis results.
    """
    results: Dict[str, Any] = {
        'valid': False,
        'file_size': 0,
        'header_valid': False,
        'ifd_valid': False,
        'data_valid': False,
        'pixel_valid': False,
        'tiff_info': None,
        'horizontal_glitch_info': None,
        'vertical_glitch_info': None,
        'horizontal_pattern_info': None,
        'vertical_pattern_info': None,
        'statistics': {},
        'errors': []
    }
    
    try:
        logger.info("Checking file: %s", file_path)
        if not os.path.exists(file_path):
            results['errors'].append("The file does not exist")
            return results
        
        results['file_size'] = os.path.getsize(file_path)
        
        logger.info("Analyzing TIFF header...")
        with open(file_path, 'rb') as f:
            header = f.read(8)
            if len(header) < 8:
                results['errors'].append("Incomplete TIFF header")
                return results
            byte_order = header[:2]
            if byte_order not in [b'II', b'MM']:
                results['errors'].append("Invalid header format")
                return results
            results['header_valid'] = True
        
        logger.info("Loading and analyzing image...")
        try:
            with Image.open(file_path) as img:
                results['tiff_info'] = get_tiff_info(img)
                img.load()
                results['data_valid'] = True
                img_array = np.array(img)
                if img_array.size > 0:
                    results['horizontal_glitch_info'] = detect_horizontal_glitches(img_array, use_dynamic=use_dynamic)
                    if results['horizontal_glitch_info'].get('lines'):
                        results['horizontal_pattern_info'] = analyze_corruption_pattern(
                            img_array, 
                            results['horizontal_glitch_info']['lines'],
                            orientation='horizontal'
                        )
                    
                    results['vertical_glitch_info'] = detect_vertical_glitches(img_array, use_dynamic=use_dynamic)
                    if results['vertical_glitch_info'].get('columns'):
                        results['vertical_pattern_info'] = analyze_corruption_pattern(
                            img_array, 
                            results['vertical_glitch_info']['columns'],
                            orientation='vertical'
                        )
                    
                    results['pixel_valid'] = not (results['horizontal_glitch_info']['detected'] or 
                                                  results['vertical_glitch_info']['detected'])
                    results['statistics'] = {
                        'mean': float(np.mean(img_array)),
                        'std': float(np.std(img_array)),
                        'min': float(np.min(img_array)),
                        'max': float(np.max(img_array))
                    }
                    # Save the image array in the results for further use (e.g., creating heatmaps)
                    results['img_array'] = img_array
        except UnidentifiedImageError as e:
            results['errors'].append(f"Image load error: {str(e)}")
            logger.error("Image load error: %s", e)
            return results
        
        try:
            with Image.open(file_path) as img:
                if hasattr(img, 'tag'):
                    results['ifd_valid'] = True
        except Exception:
            results['ifd_valid'] = False
        
        if (results['header_valid'] and 
            results['data_valid'] and 
            results['ifd_valid'] and 
            results['pixel_valid'] and 
            not results['errors']):
            results['valid'] = True
        
    except Exception as e:
        results['errors'].append(f"General error: {str(e)}")
        logger.error("General error in analyze_tiff: %s", e)
    
    return results

def print_analysis_results(results: Dict[str, Any]) -> None:
    """
    Display the analysis report in a formatted, human-readable manner.
    """
    print("\n=== TIFF Analysis Report ===")
    print(f"Overall validity: {'✓' if results['valid'] else '✗'}")
    print(f"File size: {results['file_size']:,} bytes")
    
    print("\nValidations:")
    print(f"- TIFF header: {'✓' if results['header_valid'] else '✗'}")
    print(f"- Image data: {'✓' if results['data_valid'] else '✗'}")
    print(f"- IFD structure: {'✓' if results['ifd_valid'] else '✗'}")
    print(f"- Pixel integrity: {'✓' if results.get('pixel_valid', False) else '✗'}")
    
    if results['tiff_info']:
        info = results['tiff_info']
        print("\nTIFF Information:")
        print(f"- Format: {info.get('format', 'N/A')}")
        print(f"- Mode: {info.get('mode', 'N/A')}")
        if 'size' in info:
            print(f"- Dimensions: {info['size'][0]}x{info['size'][1]} pixels")
        if 'bits_per_sample' in info:
            print(f"- Bits per sample: {info['bits_per_sample']}")
        if 'compression' in info:
            print(f"- Compression: {info['compression']}")
        if 'photometric' in info:
            print(f"- Photometric interpretation: {info['photometric']}")
        if 'resolution_unit' in info:
            print(f"- Resolution unit: {info['resolution_unit']}")
        if 'dpi' in info:
            print(f"- Resolution: {info['dpi'][0]:.0f}x{info['dpi'][1]:.0f} DPI")
    
    if results.get('horizontal_glitch_info'):
        print("\nHorizontal Glitch Analysis:")
        h_info = results['horizontal_glitch_info']
        if h_info['detected']:
            print("⚠️ Horizontal glitches detected:")
            print(f"- Number of affected zones: {h_info['count']}")
            print("- Glitch positions (ordered by severity):")
            for (start, end), severity in zip(h_info.get('lines', []), h_info.get('severity', [])):
                print(f"  • Rows {start} to {end} (intensity: {severity:.2f})")
        else:
            print("✓ No horizontal glitches detected")
    
    if results.get('vertical_glitch_info'):
        print("\nVertical Glitch Analysis:")
        v_info = results['vertical_glitch_info']
        if v_info['detected']:
            print("⚠️ Vertical glitches detected:")
            print(f"- Number of affected zones: {v_info['count']}")
            print("- Glitch positions (ordered by severity):")
            for (start, end), severity in zip(v_info.get('columns', []), v_info.get('severity', [])):
                print(f"  • Columns {start} to {end} (intensity: {severity:.2f})")
        else:
            print("✓ No vertical glitches detected")
    
    if results.get('statistics'):
        stats_dict = results['statistics']
        print("\nPixel Statistics:")
        print(f"- Mean: {stats_dict['mean']:.2f}")
        print(f"- Standard Deviation: {stats_dict['std']:.2f}")
        print(f"- Min: {stats_dict['min']}")
        print(f"- Max: {stats_dict['max']}")
    
    if results.get('horizontal_pattern_info'):
        print("\nHorizontal Corruption Pattern Analysis:")
        pattern = results['horizontal_pattern_info']
        if pattern['probable_cause']:
            print(f"Probable cause: {pattern['probable_cause']}")
            print(f"Confidence: {pattern['confidence'] * 100:.0f}%")
            if pattern['details']:
                print("\nAnalysis details:")
                for detail in pattern['details']:
                    print(f"- {detail}")
    
    if results.get('vertical_pattern_info'):
        print("\nVertical Corruption Pattern Analysis:")
        pattern = results['vertical_pattern_info']
        if pattern['probable_cause']:
            print(f"Probable cause: {pattern['probable_cause']}")
            print(f"Confidence: {pattern['confidence'] * 100:.0f}%")
            if pattern['details']:
                print("\nAnalysis details:")
                for detail in pattern['details']:
                    print(f"- {detail}")
    
    if results['errors']:
        print("\nDetected errors:")
        for error in results['errors']:
            print(f"- {error}")


def open_file(filepath: str) -> None:
    """
    Open a file using the default application based on the OS.
    """
    system = platform.system()
    try:
        if system == 'Darwin':  # macOS
            os.system(f'open "{filepath}"')
        elif system == 'Windows':
            os.startfile(filepath)
        else:  # Assume Linux or other Unix
            os.system(f'xdg-open "{filepath}"')
    except Exception as e:
        print(f"Could not open file {filepath}: {e}")

def create_heatmap(results: Dict[str, Any], img_array: np.ndarray) -> None:
    """
    Create and save heatmaps of detected glitches.
    
    For vertical glitches:
      - A mask is created by marking the glitch column ranges.
      - This mask is overlaid on the grayscale version of the image.
    
    For horizontal glitches:
      - A mask is created by marking the glitch row ranges.
      - This mask is overlaid on the grayscale image.
    
    The heatmaps are saved as 'vertical_glitches_heatmap.png' and 'horizontal_glitches_heatmap.png'.
    After saving, the function attempts to open the saved heatmap files automatically.
    """
    # Convert image to grayscale for visualization
    gray_img = compute_luminance(img_array)
    
    # Improve figure readability: set a larger figure size and use a descriptive colormap.
    figsize = (12, 10)
    
    # Create vertical glitches heatmap if detected
    if results.get('vertical_glitch_info') and results['vertical_glitch_info'].get('detected'):
        mask_vert = np.zeros_like(gray_img)
        for (start, end) in results['vertical_glitch_info']['columns']:
            mask_vert[:, start:end+1] = 1  # Mark glitch columns
        
        plt.figure(figsize=figsize)
        plt.imshow(gray_img, cmap='gray')
        # Use a colormap like 'hot' or 'viridis' to highlight glitches
        plt.imshow(mask_vert, cmap='viridis', alpha=0.95)
        plt.title('Vertical Glitches Heatmap')
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        cbar = plt.colorbar(label='Glitch Presence')
        plt.tight_layout()
        vert_filename = 'vertical_glitches_heatmap.png'
        plt.savefig(vert_filename, dpi=150)
        plt.close()
        logger.info(f"Vertical glitches heatmap saved as '{vert_filename}'.")
        open_file(vert_filename)
    
    # Create horizontal glitches heatmap if detected
    if results.get('horizontal_glitch_info') and results['horizontal_glitch_info'].get('detected'):
        mask_horiz = np.zeros_like(gray_img)
        for (start, end) in results['horizontal_glitch_info']['lines']:
            mask_horiz[start:end+1, :] = 1  # Mark glitch rows
        
        plt.figure(figsize=figsize)
        plt.imshow(gray_img, cmap='gray')
        plt.imshow(mask_horiz, cmap='viridis', alpha=0.95)
        plt.title('Horizontal Glitches Heatmap')
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        cbar = plt.colorbar(label='Glitch Presence')
        plt.tight_layout()
        horiz_filename = 'horizontal_glitches_heatmap.png'
        plt.savefig(horiz_filename, dpi=150)
        plt.close()
        logger.info(f"Horizontal glitches heatmap saved as '{horiz_filename}'.")
        open_file(horiz_filename)


def main() -> None:
    global BASE_THRESHOLD_FACTOR, LOCAL_THRESHOLD_STD_FACTOR, ANOMALY_RATIO_THRESHOLD
    global SIGNIFICANCE_MULTIPLIER, GROUP_SEVERITY_MULTIPLIER, MAX_GROUP_SIZE, MIN_GROUP_LENGTH
    global MAX_ALIGNMENT_DIFF, REGULARITY_THRESHOLD, REPEATED_PATTERN_RATIO, CLUSTER_GAP_THRESHOLD

    parser = argparse.ArgumentParser(
        description="TIFF Analysis Tool with modifiable detection thresholds."
    )
    parser.add_argument("file", help="Path to the TIFF file to analyze")
    parser.add_argument("--base-threshold-factor", type=float, default=BASE_THRESHOLD_FACTOR,
                        help=f"Base threshold factor (default: {BASE_THRESHOLD_FACTOR})")
    parser.add_argument("--local-threshold-std-factor", type=float, default=LOCAL_THRESHOLD_STD_FACTOR,
                        help=f"Local threshold std factor (default: {LOCAL_THRESHOLD_STD_FACTOR})")
    parser.add_argument("--anomaly-ratio-threshold", type=float, default=ANOMALY_RATIO_THRESHOLD,
                        help=f"Anomaly ratio threshold (default: {ANOMALY_RATIO_THRESHOLD})")
    parser.add_argument("--significance-multiplier", type=float, default=SIGNIFICANCE_MULTIPLIER,
                        help=f"Significance multiplier (default: {SIGNIFICANCE_MULTIPLIER})")
    parser.add_argument("--group-severity-multiplier", type=float, default=GROUP_SEVERITY_MULTIPLIER,
                        help=f"Group severity multiplier (default: {GROUP_SEVERITY_MULTIPLIER})")
    parser.add_argument("--max-group-size", type=int, default=MAX_GROUP_SIZE,
                        help=f"Maximum group size (default: {MAX_GROUP_SIZE})")
    parser.add_argument("--min-group-length", type=int, default=MIN_GROUP_LENGTH,
                        help=f"Minimum group length (default: {MIN_GROUP_LENGTH})")
    parser.add_argument("--max-alignment-diff", type=int, default=MAX_ALIGNMENT_DIFF,
                        help=f"Maximum alignment difference (default: {MAX_ALIGNMENT_DIFF})")
    parser.add_argument("--regularity-threshold", type=float, default=REGULARITY_THRESHOLD,
                        help=f"Regularity threshold (default: {REGULARITY_THRESHOLD})")
    parser.add_argument("--repeated-pattern-ratio", type=float, default=REPEATED_PATTERN_RATIO,
                        help=f"Repeated pattern ratio (default: {REPEATED_PATTERN_RATIO})")
    parser.add_argument("--cluster-gap-threshold", type=float, default=CLUSTER_GAP_THRESHOLD,
                        help=f"Cluster gap threshold (default: {CLUSTER_GAP_THRESHOLD})")
    parser.add_argument("--dynamic", action="store_true",
                        help="Use dynamic threshold computation based on image statistics")
    args = parser.parse_args()

    BASE_THRESHOLD_FACTOR = args.base_threshold_factor
    LOCAL_THRESHOLD_STD_FACTOR = args.local_threshold_std_factor
    ANOMALY_RATIO_THRESHOLD = args.anomaly_ratio_threshold
    SIGNIFICANCE_MULTIPLIER = args.significance_multiplier
    GROUP_SEVERITY_MULTIPLIER = args.group_severity_multiplier
    MAX_GROUP_SIZE = args.max_group_size
    MIN_GROUP_LENGTH = args.min_group_length
    MAX_ALIGNMENT_DIFF = args.max_alignment_diff
    REGULARITY_THRESHOLD = args.regularity_threshold
    REPEATED_PATTERN_RATIO = args.repeated_pattern_ratio
    CLUSTER_GAP_THRESHOLD = args.cluster_gap_threshold

    print("\n" + "="*50)
    print("{:^50}".format("TIFF Data Analysis"))
    print("{:^50}".format("by Yan Senez"))
    print("="*50 + "\n")
    
    file_path = args.file
    results = analyze_tiff(file_path, use_dynamic=args.dynamic)
    print_analysis_results(results)
    
    # If the analysis contains an image array and glitches were detected, create heatmaps.
    if 'img_array' in results and results.get('vertical_glitch_info', {}).get('detected'):
        create_heatmap(results, results['img_array'])

if __name__ == "__main__":
    main()
