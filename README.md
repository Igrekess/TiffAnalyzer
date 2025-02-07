# TIFF Analysis Tool

TIFF Analysis Tool is a Python-based application designed to detect and analyze glitches (unexpected anomalies or corruptions) in TIFF images. The tool extracts metadata, computes pixel brightness (luminance), detects abnormal changes (glitches) along rows and columns, groups these anomalies, and then uses a series of heuristics to suggest the probable origin of the corruption.

## Project Background & Future Direction
This tool was initially developed to help a client sort through a large batch of TIFF files that had suffered from corruption issues during their development process in Capture One. While solving this specific problem, the project evolved into a personal exploration of file structure analysis and corruption detection.

The project is now being expanded with two main objectives:
- Providing a deeper understanding of image file structures
- Extending the analysis capabilities to RAW file formats

---

## Table of Contents

1. [Overview](#overview)
2. [Interfaces](#interfaces)
   - [Command Line Interface](#command-line-interface)
   - [Graphical User Interface](#graphical-user-interface)
3. [Function-by-Function Explanation with Sources](#function-by-function-explanation-with-sources)
4. [Usage](#usage)
5. [Parameters and Settings](#parameters-and-settings)
6. [Installation](#installation)
7. [Requirements](#requirements)

---

## Overview

The tool processes a TIFF image in several stages:

1. **Metadata Extraction:** Reads and validates the image file and extracts metadata.
2. **Luminance Computation:** Converts the image from color (RGB) to grayscale using weighted coefficients.
3. **Glitch Detection:** Examines differences between adjacent rows or columns to detect anomalies.
4. **Grouping:** Groups consecutive anomalies into glitch regions.
5. **Pattern Analysis:** Extracts features from these regions (such as intensity, periodicity, and alignment) and uses heuristics to classify the type of corruption.
6. **Reporting:** Produces a detailed report including all findings and debug information.
7. **Visualization:** (GUI only) Provides heatmap visualization of detected glitches.

---

## Interfaces

### Command Line Interface
The traditional command-line interface offers full control over analysis parameters and is ideal for batch processing or automation.

```bash
python3 analyse_tiff.py image.tiff [options]
```

### Graphical User Interface
The new GUI provides an intuitive interface with real-time visualization and interactive controls:

**Features:**
- Interactive file selection
- Real-time analysis progress
- Adjustable transparency heatmaps
- Tabbed interface for results and visualizations
- Timestamped reports
- Dynamic parameter adjustment

**To launch the GUI:**
```bash
./analyse.sh --gui
```

---

## Function-by-Function Explanation with Sources

### get_tiff_info

**Purpose:**  
Extracts basic metadata and TIFF-specific information from the image.

**Logic:**  
- Reads the image format, mode, dimensions, and other TIFF tags like BitsPerSample, Compression, PhotometricInterpretation, and DPI.

**Significance:**  
Metadata is essential for understanding the image context and verifying its integrity.

**Source:**  
- General TIFF documentation (see [TIFF Specification](https://www.adobe.io/open/standards/TIFF.html)).

---

### compute_luminance

**Purpose:**  
Converts a color image to a grayscale image by computing the luminance.

**Logic:**  
- For an RGB image, it uses the formula:  
  **Luminance = 0.2989 × Red + 0.5870 × Green + 0.1140 × Blue**
  
**Why these Coefficients?**  
- These values come from the ITU-R BT.601 standard and are chosen because the human eye is more sensitive to green, less to red, and even less to blue. This produces a grayscale image that accurately reflects perceived brightness.

**Source:**  
- [ITU-R BT.601 standard](https://en.wikipedia.org/wiki/Rec._601)  
- [Wikipedia: Y'UV](https://en.wikipedia.org/wiki/YUV)

**Significance:**  
Working in grayscale simplifies further analysis by reducing the data from three channels to one, focusing solely on brightness variations.

---

### compute_dynamic_thresholds

**Purpose:**  
Calculates thresholds for glitch detection dynamically based on the image's local statistics.

**Logic:**  
- Computes the absolute differences between adjacent rows (or columns).
- Flattens these differences and calculates the 90th percentile as the base threshold and computes the standard deviation.
  
**Why Dynamic Thresholding?**  
- Dynamic thresholds can adapt to varying contrast and noise levels in different images, potentially improving glitch detection accuracy over fixed thresholds.

**Source:**  
- General principles of adaptive thresholding in image processing (see [Otsu's Method](https://en.wikipedia.org/wiki/Otsu%27s_method) for related concepts).

---

### detect_glitches

**Purpose:**  
Identifies rows or columns where the pixel intensity changes significantly compared to neighboring rows/columns.

**Logic:**  
- Iterates through rows or columns.
- Compares the average difference (anomaly ratio) with a threshold (either fixed or dynamic).
- Groups consecutive anomalies into glitch regions and records the maximum difference (intensity) in that group.

**Significance:**  
This method helps isolate regions where the image data behaves abnormally—often a sign of corruption.

---

### detect_horizontal_glitches & detect_vertical_glitches

**Purpose:**  
Wrapper functions that detect glitches along rows (horizontal) and columns (vertical) respectively.

**Logic:**  
- They first compute the luminance of the image.
- Then, they call `detect_glitches` with the appropriate axis value.
- They support both fixed and dynamic threshold modes.

**Significance:**  
Different corruptions may appear more clearly in one direction; analyzing both provides a comprehensive view of the image's integrity.

---

### analyze_corruption_pattern

**Purpose:**  
Examines the characteristics of detected glitch regions to infer the likely cause of corruption.

**Logic:**  
- **Feature Extraction:**  
  - **Glitch Positions and Widths:** Calculates the center and width of each glitch region.
  - **Glitch Intensity:** Computes the average pixel intensity in each glitch region.
  - **Periodicity:** Uses autocorrelation of glitch center positions to measure how regularly glitches occur.
- **Heuristic Classification:**  
  - **Buffer Corruption:** If glitches occur very regularly (high periodicity), are narrow (well-aligned), and are brighter than the overall image mean, this may indicate a buffering or caching error during writing.
  - **Memory Corruption:** If glitches are very dark (very low intensity) with little variation, this might suggest memory corruption.
  - **Write Corruption:** Otherwise, the glitches might result from errors during file writing or data transfer.
  
**Sources and Rationale:**  
- **Periodicity and Buffer Corruption:**  
  - Regular (periodic) patterns often indicate systematic errors. Buffer-related issues can cause periodic repetitions due to cyclic data handling.  
  - *Reference:* Basic signal processing texts (e.g., [Digital Signal Processing](https://en.wikipedia.org/wiki/Digital_signal_processing)) explain autocorrelation and periodicity.
- **Dark Glitches and Memory Corruption:**  
  - Memory corruption can lead to missing or zeroed data, causing regions to appear very dark.  
  - *Reference:* Common observations in digital imaging; see textbooks such as *Digital Image Processing* by Gonzalez and Woods.
  
**Significance:**  
By combining these features, the tool provides a probable explanation for the detected glitches, helping diagnose the source of corruption.

---

## Usage

### Command Line Mode
```bash
# Basic analysis
./analyse.sh input.tiff

# Analysis with dynamic thresholds
./analyse.sh input.tiff --dynamic

# Analysis with custom parameters
./analyse.sh input.tiff --base-threshold-factor 0.2 --max-group-size 30
```

### GUI Mode
1. Launch the GUI:
```bash
./analyse.sh --gui
```

2. Use the interface:
   - Click "Select TIFF File" to choose an image
   - Adjust analysis parameters if needed
   - Click "Analyze" to start processing
   - Use the tabs to view:
     - Text report
     - Horizontal glitch heatmap
     - Vertical glitch heatmap
   - Adjust heatmap transparency using the slider

## Parameters and Settings

### GUI Settings
- **Dynamic Threshold:** Enables automatic threshold computation
- **Base Threshold Factor:** Base factor for glitch detection (0.0 - 1.0)
- **Max Group Size:** Maximum size of glitch groups (1 - 100)
- **Heatmap Transparency:** Adjustable visualization opacity (0% - 100%)

### Command Line Parameters

- **`file`**
**Description:** Path to the TIFF file to analyze.
**Example:** `image.tiff`

- **`--base-threshold-factor`**
**Description:** Factor used to calculate the base threshold in fixed mode.
**Default value:** 0.15
**Example:** `--base-threshold-factor 0.2`

- **`--local-threshold-std-factor`**
**Description:** Factor applied to local standard deviation to adjust threshold in fixed mode.
**Default value:** 0.5
**Example:** `--local-threshold-std-factor 0.4`

- **`--anomaly-ratio-threshold`**
**Description:** Minimum percentage of pixels exceeding threshold to consider a line/column as anomalous.
**Default value:** 0.4
**Example:** `--anomaly-ratio-threshold 0.45`

- **`--significance-multiplier`**
**Description:** Multiplier applied to standard deviation to judge significance of an anomaly.
**Default value:** 2.0
**Example:** `--significance-multiplier 1.8`

- **`--group-severity-multiplier`**
**Description:** Multiplier for group significance determination.
**Default value:** 3.0
**Example:** `--group-severity-multiplier 2.5`

- **`--max-group-size`**
**Description:** Maximum size for a valid glitch group.
**Default value:** 20
**Example:** `--max-group-size 30`

- **`--min-group-length`**
**Description:** Minimum length required for a valid glitch group.
**Default value:** 2
**Example:** `--min-group-length 3`

- **`--max-alignment-diff`**
**Description:** Maximum difference for well-aligned glitch consideration.
**Default value:** 5
**Example:** `--max-alignment-diff 4`

- **`--dynamic`**
**Description:** Activates dynamic threshold mode
**Type:** Flag (no value needed)
**Example:** `--dynamic`

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
```

2. Install dependencies:
```bash
./analyse.sh --gui  # Will automatically set up the virtual environment and install dependencies
```

## Requirements

### Core Dependencies
- Python 3.8+
- NumPy
- Pillow
- SciPy
- Matplotlib

### GUI Dependencies
- PyQt6
- Matplotlib
- tqdm

All dependencies are automatically installed when using `analyse.sh`.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Author

Yan Senez - Initial work
