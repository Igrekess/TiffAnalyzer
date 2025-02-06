import os
import sys
from PIL import Image, ImageFile
import numpy as np
import warnings
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None

def get_tiff_info(img):
    """
    Extract detailed information from a TIFF image.
    """
    info = {}
    try:
        # Basic information
        info['format'] = img.format
        info['mode'] = img.mode
        info['size'] = img.size
        
        # TIFF metadata
        if hasattr(img, 'tag'):
            tags = img.tag
            if 258 in tags:  # BitsPerSample
                info['bits_per_sample'] = tags[258]
            if 259 in tags:  # Compression
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
            if 262 in tags:  # PhotometricInterpretation
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
            if 296 in tags:  # ResolutionUnit
                unit_codes = {1: "None", 2: "Inches", 3: "Centimeters"}
                unit_value = tags[296][0]
                info['resolution_unit'] = unit_codes.get(unit_value, f"Unknown ({unit_value})")
            if 282 in tags and 283 in tags:  # XResolution and YResolution
                info['dpi'] = (float(tags[282][0][0]) / float(tags[282][0][1]),
                               float(tags[283][0][0]) / float(tags[283][0][1]))
    except Exception as e:
        info['error'] = str(e)
    return info

def detect_vertical_glitches(img_array):
    """
    Detect vertical glitch columns in the image.
    """
    glitch_info = {
        'detected': False,
        'columns': [],
        'count': 0,
        'severity': []
    }
    
    try:
        height, width = img_array.shape[:2]
        
        if len(img_array.shape) == 3:
            luminance = 0.2989 * img_array[:, :, 0] + 0.5870 * img_array[:, :, 1] + 0.1140 * img_array[:, :, 2]
        else:
            luminance = img_array
            
        luminance = luminance.T  # Transpose to analyze columns
        
        potential_glitches = []
        print("\nAnalyzing image columns...")
        print("Searching for potential vertical glitches...")
        
        # Calculate global mean and standard deviation for adaptive thresholds
        global_mean = np.mean(luminance)
        global_std = np.std(luminance)
        base_threshold = global_mean * 0.15  # 15% of the mean

        for x in tqdm(range(1, width - 1), desc="Scanning columns", unit="cols"):
            # Compute differences with adjacent columns
            diff_prev = np.abs(luminance[x] - luminance[x - 1])
            diff_next = np.abs(luminance[x] - luminance[x + 1])
            
            # Adaptive thresholds based on local statistics
            local_threshold = base_threshold + (global_std * 0.5)  # Add 50% of the std dev
            anomaly_prev = np.mean(diff_prev > local_threshold)
            anomaly_next = np.mean(diff_next > local_threshold)
            
            # More strict anomaly check
            if (anomaly_prev > 0.4 or anomaly_next > 0.4):  # At least 40% of the pixels must be anomalous
                max_diff = max(np.max(diff_prev), np.max(diff_next))
                if max_diff > global_std * 2:  # The difference must be significant
                    potential_glitches.append((x, max_diff))
        
        if potential_glitches:
            print("\nAnalyzing detected vertical glitch regions...")
            current_start = potential_glitches[0][0]
            current_severity = potential_glitches[0][1]
            current_group = [current_start]
            glitch_count = 0
            
            for x, severity in potential_glitches[1:]:
                # Grouping strictly consecutive columns
                if x - current_group[-1] <= 1:
                    current_group.append(x)
                    current_severity = max(current_severity, severity)
                else:
                    if len(current_group) >= 2:  # At least 2 consecutive columns
                        if current_severity > global_std * 3:  # Severity must be highly significant
                            glitch_count += 1
                            range_size = max(current_group) - min(current_group) + 1
                            if range_size <= 20:  # Limit glitch region size
                                print(f"\r⚠️  Vertical glitch #{glitch_count} detected - Columns {min(current_group)} to {max(current_group)} (intensity: {current_severity:.2f})", end="", flush=True)
                                glitch_info['columns'].append((min(current_group), max(current_group)))
                                glitch_info['severity'].append(current_severity)
                    current_start = x
                    current_group = [x]
                    current_severity = severity
            
            # Process the final group with the same criteria
            if len(current_group) >= 2 and current_severity > global_std * 3:
                range_size = max(current_group) - min(current_group) + 1
                if range_size <= 20:
                    glitch_count += 1
                    print(f"\r⚠️  Vertical glitch #{glitch_count} detected - Columns {min(current_group)} to {max(current_group)} (intensity: {current_severity:.2f})", end="", flush=True)
                    glitch_info['columns'].append((min(current_group), max(current_group)))
                    glitch_info['severity'].append(current_severity)
            
            print("\n")
        
        # Update final information
        if glitch_info['columns']:
            glitch_info['detected'] = True
            glitch_info['count'] = len(glitch_info['columns'])
            
            # Sort glitches by severity
            sorted_glitches = sorted(zip(glitch_info['columns'], glitch_info['severity']),
                                       key=lambda x: x[1], reverse=True)
            glitch_info['columns'] = [g[0] for g in sorted_glitches]
            glitch_info['severity'] = [g[1] for g in sorted_glitches]
            
            print(f"\nTotal: {glitch_info['count']} vertical glitches detected\n")
    
    except Exception as e:
        print(f"Error detecting vertical glitches: {str(e)}")
    
    return glitch_info

def detect_horizontal_glitches(img_array):
    """
    Detect horizontal glitch rows in the image.
    """
    glitch_info = {
        'detected': False,
        'lines': [],
        'count': 0,
        'severity': []
    }
    
    try:
        height, width = img_array.shape[:2]
        
        if len(img_array.shape) == 3:
            luminance = 0.2989 * img_array[:, :, 0] + 0.5870 * img_array[:, :, 1] + 0.1140 * img_array[:, :, 2]
        else:
            luminance = img_array
        
        potential_glitches = []
        print("\nAnalyzing image rows...")
        print("Searching for potential horizontal glitches...")
        
        # Calculate global mean and standard deviation for adaptive thresholds
        global_mean = np.mean(luminance)
        global_std = np.std(luminance)
        base_threshold = global_mean * 0.15  # 15% of the mean

        for y in tqdm(range(1, height - 1), desc="Scanning rows", unit="rows"):
            # Compute differences with adjacent rows
            diff_prev = np.abs(luminance[y] - luminance[y - 1])
            diff_next = np.abs(luminance[y] - luminance[y + 1])
            
            # Adaptive thresholds based on local statistics
            local_threshold = base_threshold + (global_std * 0.5)  # Add 50% of the std dev
            anomaly_prev = np.mean(diff_prev > local_threshold)
            anomaly_next = np.mean(diff_next > local_threshold)
            
            # More strict anomaly check
            if (anomaly_prev > 0.4 or anomaly_next > 0.4):  # At least 40% of the pixels must be anomalous
                max_diff = max(np.max(diff_prev), np.max(diff_next))
                if max_diff > global_std * 2:  # The difference must be significant
                    potential_glitches.append((y, max_diff))
        
        if potential_glitches:
            print("\nAnalyzing detected horizontal glitch regions...")
            current_start = potential_glitches[0][0]
            current_severity = potential_glitches[0][1]
            current_group = [current_start]
            glitch_count = 0
            
            for y, severity in potential_glitches[1:]:
                # Grouping strictly consecutive rows
                if y - current_group[-1] <= 1:
                    current_group.append(y)
                    current_severity = max(current_severity, severity)
                else:
                    if len(current_group) >= 2:  # At least 2 consecutive rows
                        if current_severity > global_std * 3:  # Severity must be highly significant
                            glitch_count += 1
                            range_size = max(current_group) - min(current_group) + 1
                            if range_size <= 20:  # Limit glitch region size
                                print(f"\r⚠️  Horizontal glitch #{glitch_count} detected - Rows {min(current_group)} to {max(current_group)} (intensity: {current_severity:.2f})", end="", flush=True)
                                glitch_info['lines'].append((min(current_group), max(current_group)))
                                glitch_info['severity'].append(current_severity)
                    current_start = y
                    current_group = [y]
                    current_severity = severity
            
            # Process the final group with the same criteria
            if len(current_group) >= 2 and current_severity > global_std * 3:
                range_size = max(current_group) - min(current_group) + 1
                if range_size <= 20:
                    glitch_count += 1
                    print(f"\r⚠️  Horizontal glitch #{glitch_count} detected - Rows {min(current_group)} to {max(current_group)} (intensity: {current_severity:.2f})", end="", flush=True)
                    glitch_info['lines'].append((min(current_group), max(current_group)))
                    glitch_info['severity'].append(current_severity)
            
            print("\n")
        
        # Update final information
        if glitch_info['lines']:
            glitch_info['detected'] = True
            glitch_info['count'] = len(glitch_info['lines'])
            
            # Sort glitches by severity
            sorted_glitches = sorted(zip(glitch_info['lines'], glitch_info['severity']),
                                       key=lambda x: x[1], reverse=True)
            glitch_info['lines'] = [g[0] for g in sorted_glitches]
            glitch_info['severity'] = [g[1] for g in sorted_glitches]
            
            print(f"\nTotal: {glitch_info['count']} horizontal glitches detected\n")
    
    except Exception as e:
        print(f"Error detecting horizontal glitches: {str(e)}")
    
    return glitch_info

def analyze_corruption_pattern(img_array, glitch_lines, orientation='horizontal'):
    """
    Detailed analysis of corruption patterns to identify their probable origin.
    """
    pattern_info = {
        'type': 'unknown',
        'confidence': 0,
        'details': [],
        'probable_cause': None
    }
    
    try:
        if len(img_array.shape) == 3:
            luminance = 0.2989 * img_array[:, :, 0] + 0.5870 * img_array[:, :, 1] + 0.1140 * img_array[:, :, 2]
        else:
            luminance = img_array
            
        if orientation == 'vertical':
            luminance = luminance.T
        
        # Analyze the regularity of intervals between glitches
        intervals = []
        for i in range(1, len(glitch_lines)):
            interval = glitch_lines[i][0] - glitch_lines[i - 1][1]
            intervals.append(interval)
        
        if intervals:
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            regularity = 1 - (std_interval / mean_interval if mean_interval > 0 else 0)
            
            # Analyze pixel values in corrupted zones
            glitch_values = []
            for start, end in glitch_lines:
                glitch_region = luminance[start:end + 1, :]
                glitch_values.extend(glitch_region.flatten())
            
            # Glitch characteristics
            is_periodic = regularity > 0.7
            has_zero_values = any(v == 0 for v in glitch_values)
            has_repeated_pattern = len(set(glitch_values)) < len(glitch_values) * 0.1
            is_aligned = all(g[1] - g[0] < 5 for g in glitch_lines)
            
            # Identify the type of corruption
            direction = "vertical" if orientation == 'vertical' else "horizontal"
            if is_periodic and is_aligned:
                pattern_info['type'] = 'buffer_corruption'
                pattern_info['confidence'] = 0.8
                pattern_info['probable_cause'] = f"Buffer/cache error during writing ({direction})"
                pattern_info['details'].append(f"Average interval between glitches: {mean_interval:.1f} pixels")
                
            elif has_repeated_pattern and has_zero_values:
                pattern_info['type'] = 'memory_corruption'
                pattern_info['confidence'] = 0.7
                pattern_info['probable_cause'] = f"RAM issue during processing ({direction})"
                pattern_info['details'].append("Repetitive patterns detected in corrupted areas")
                
            elif is_aligned and not is_periodic:
                pattern_info['type'] = 'write_corruption'
                pattern_info['confidence'] = 0.6
                pattern_info['probable_cause'] = f"Interruption or error during file writing ({direction})"
                pattern_info['details'].append("Aligned but non-periodic corruptions")
            
            # Analyze spatial distribution
            dimension = img_array.shape[1] if orientation == 'vertical' else img_array.shape[0]
            relative_positions = [(start / dimension, end / dimension) for start, end in glitch_lines]
            
            # Group the corruptions
            clusters = []
            current_cluster = [relative_positions[0]]
            
            for pos in relative_positions[1:]:
                if pos[0] - current_cluster[-1][1] < 0.05:
                    current_cluster.append(pos)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [pos]
            clusters.append(current_cluster)
            
            # Cluster analysis
            if len(clusters) == 1 and len(clusters[0]) > 5:
                pattern_info['details'].append("Corruptions concentrated in a single area")
                if pattern_info['probable_cause']:
                    pattern_info['probable_cause'] += " - Likely a one-time event"
            elif len(clusters) > 3:
                pattern_info['details'].append(f"Corruptions distributed across {len(clusters)} distinct areas")
                if pattern_info['probable_cause']:
                    pattern_info['probable_cause'] += " - Potentially recurring issue"
                    
    except Exception as e:
        pattern_info['details'].append(f"Error analyzing patterns: {str(e)}")
    
    return pattern_info

def analyze_tiff(file_path):
    """
    Analyze a TIFF image and check for signs of corruption.
    """
    results = {
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
        'errors': []
    }
    
    try:
        print("\nChecking the file...")
        if not os.path.exists(file_path):
            results['errors'].append("The file does not exist")
            return results
            
        results['file_size'] = os.path.getsize(file_path)
        
        print("Analyzing the TIFF header...")
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
        
        print("Loading and analyzing the image...")
        with Image.open(file_path) as img:
            results['tiff_info'] = get_tiff_info(img)
            
            try:
                img.load()
                results['data_valid'] = True
                
                print("Converting and preparing data...")
                img_array = np.array(img)
                if img_array.size > 0:
                    # Detect horizontal glitches
                    results['horizontal_glitch_info'] = detect_horizontal_glitches(img_array)
                    if results['horizontal_glitch_info']['lines']:
                        results['horizontal_pattern_info'] = analyze_corruption_pattern(
                            img_array, 
                            results['horizontal_glitch_info']['lines'],
                            orientation='horizontal'
                        )
                    
                    # Detect vertical glitches
                    results['vertical_glitch_info'] = detect_vertical_glitches(img_array)
                    if results['vertical_glitch_info']['columns']:
                        results['vertical_pattern_info'] = analyze_corruption_pattern(
                            img_array, 
                            results['vertical_glitch_info']['columns'],
                            orientation='vertical'
                        )
                    
                    # The image is considered valid if no glitches are detected
                    results['pixel_valid'] = not (results['horizontal_glitch_info']['detected'] or 
                                                  results['vertical_glitch_info']['detected'])
                    
                    print("Calculating statistics...")
                    results['statistics'] = {
                        'mean': float(np.mean(img_array)),
                        'std': float(np.std(img_array)),
                        'min': float(np.min(img_array)),
                        'max': float(np.max(img_array))
                    }
            except Exception as e:
                results['errors'].append(f"Error analyzing image data: {str(e)}")
                return results
            
            if hasattr(img, 'tag'):
                results['ifd_valid'] = True
            
            if (results['header_valid'] and 
                results['data_valid'] and 
                results['ifd_valid'] and
                results['pixel_valid'] and
                len(results['errors']) == 0):
                results['valid'] = True
                
    except Exception as e:
        results['errors'].append(f"General error: {str(e)}")
    
    return results

def print_analysis_results(results):
    """
    Display the analysis results in a formatted manner.
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

    if 'horizontal_glitch_info' in results:
        print("\nHorizontal Glitch Analysis:")
        if results['horizontal_glitch_info']['detected']:
            print("⚠️ Horizontal glitches detected:")
            print(f"- Number of affected zones: {results['horizontal_glitch_info']['count']}")
            print("- Glitch positions (ordered by severity):")
            for i, ((start, end), severity) in enumerate(zip(results['horizontal_glitch_info']['lines'], 
                                                                results['horizontal_glitch_info']['severity'])):
                print(f"  • Rows {start} to {end} (intensity: {severity:.2f})")
        else:
            print("✓ No horizontal glitches detected")

    if 'vertical_glitch_info' in results:
        print("\nVertical Glitch Analysis:")
        if results['vertical_glitch_info']['detected']:
            print("⚠️ Vertical glitches detected:")
            print(f"- Number of affected zones: {results['vertical_glitch_info']['count']}")
            print("- Glitch positions (ordered by severity):")
            for i, ((start, end), severity) in enumerate(zip(results['vertical_glitch_info']['columns'], 
                                                                results['vertical_glitch_info']['severity'])):
                print(f"  • Columns {start} to {end} (intensity: {severity:.2f})")
        else:
            print("✓ No vertical glitches detected")
    
    if 'statistics' in results:
        print("\nPixel Statistics:")
        print(f"- Mean: {results['statistics']['mean']:.2f}")
        print(f"- Standard Deviation: {results['statistics']['std']:.2f}")
        print(f"- Min: {results['statistics']['min']}")
        print(f"- Max: {results['statistics']['max']}")
    
    if 'horizontal_pattern_info' in results and results['horizontal_pattern_info']:
        print("\nHorizontal Corruption Pattern Analysis:")
        pattern = results['horizontal_pattern_info']
        if pattern['probable_cause']:
            print(f"Probable cause: {pattern['probable_cause']}")
            print(f"Confidence: {pattern['confidence'] * 100:.0f}%")
            if pattern['details']:
                print("\nAnalysis details:")
                for detail in pattern['details']:
                    print(f"- {detail}")

    if 'vertical_pattern_info' in results and results['vertical_pattern_info']:
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

def print_header():
    """Prints the program header."""
    print("\n" + "=" * 50)
    print("{:^50}".format("TIFF Data Analysis"))
    print("{:^50}".format("by Yan Senez"))
    print("=" * 50 + "\n")

if __name__ == "__main__":
    print_header()
    
    if len(sys.argv) != 2:
        print("Usage: python analyze_tiff.py <file.tiff>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    results = analyze_tiff(file_path)
    print_analysis_results(results)
