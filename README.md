# TIFF Experimental Analysis Tool

TIFF Analysis Tool is a Python-based application designed to detect and analyze glitches (unexpected anomalies or corruptions) in TIFF images. The tool extracts metadata, computes pixel brightness (luminance), detects abnormal changes (glitches) along rows and columns, groups these anomalies, and then uses a series of heuristics to suggest the probable origin of the corruption.

---

## Table of Contents

1. [Overview](#overview)
2. [Function-by-Function Explanation with Sources](#function-by-function-explanation-with-sources)
   - [get_tiff_info](#get_tiff_info)
   - [compute_luminance](#compute_luminance)
   - [compute_dynamic_thresholds](#compute_dynamic_thresholds)
   - [detect_glitches](#detect_glitches)
   - [detect_horizontal_glitches & detect_vertical_glitches](#detect_horizontal_glitches--detect_vertical_glitches)
   - [analyze_corruption_pattern](#analyze_corruption_pattern)
   - [analyze_tiff](#analyze_tiff)
   - [print_analysis_results](#print_analysis_results)
   - [main](#main)
3. [Glitch Identification Strategies and Heuristics](#glitch-identification-strategies-and-heuristics)
4. [Usage](#usage)
5. [Conclusion](#conclusion)

---

## Overview

The tool processes a TIFF image in several stages:

1. **Metadata Extraction:** Reads and validates the image file and extracts metadata.
2. **Luminance Computation:** Converts the image from color (RGB) to grayscale using weighted coefficients.
3. **Glitch Detection:** Examines differences between adjacent rows or columns to detect anomalies.
4. **Grouping:** Groups consecutive anomalies into glitch regions.
5. **Pattern Analysis:** Extracts features from these regions (such as intensity, periodicity, and alignment) and uses heuristics to classify the type of corruption.
6. **Reporting:** Produces a detailed report including all findings and debug information.

---

## Function-by-Function Explanation

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
Calculates thresholds for glitch detection dynamically based on the image’s local statistics.

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

### analyze_tiff

**Purpose:**  
Conducts a full analysis of the TIFF image.

**Logic:**  
- Verifies file existence and header validity.
- Loads the image and extracts metadata.
- Converts the image to an array and computes pixel statistics.
- Detects horizontal and vertical glitches.
- Analyzes glitch patterns to infer the probable cause.
- Compiles all results into a comprehensive report.

**Significance:**  
This function ties together all individual analyses and provides a complete picture of the image’s condition.

---

### print_analysis_results

**Purpose:**  
Prints the analysis report in an organized, human-readable format.

**Logic:**  
- Displays overall image validity, metadata, glitch detection results (including number and positions of glitches), pixel statistics, and the results of the corruption pattern analysis.
- Lists any errors encountered during processing.

**Significance:**  
A clear report helps users understand the state of the image and identify potential problems.

---

### main

**Purpose:**  
Serves as the entry point of the script.

**Logic:**  
- Uses the `argparse` module to parse command-line arguments (including the TIFF file path and optional parameters such as `--dynamic`).
- Updates global threshold parameters based on user input.
- Calls `analyze_tiff` and then `print_analysis_results` to perform and display the full analysis.

**Significance:**  
This function makes the tool user-friendly and configurable via the command line.

---
## Complete List of Additional Parameters
Here is the list of additional command line options and their functions:

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
**Description:** Multiplier applied to standard deviation (fixed or dynamic) to judge significance of an anomaly.
**Default value:** 2.0
**Example:** `--significance-multiplier 1.8`

- **`--group-severity-multiplier`**
**Description:** Multiplier used to determine if a group of glitches has sufficient intensity to be considered significant.
**Default value:** 3.0
**Example:** `--group-severity-multiplier 2.5`

- **`--max-group-size`**
**Description:** Maximum size (in number of lines or columns) that a group of glitches can have to be considered valid.
**Default value:** 20
**Example:** `--max-group-size 30`

- **`--min-group-length`**
**Description:** Minimum length (number of consecutive lines/columns) required for a group to be considered a glitch.
**Default value:** 2
**Example:** `--min-group-length 3`

- **`--max-alignment-diff`**
**Description:** Maximum difference between start and end of a glitch for it to be considered "narrow" and thus well-aligned.
**Default value:** 5
**Example:** `--max-alignment-diff 4`

- **`--regularity-threshold`**
**Description:** Threshold for measuring glitch regularity (periodicity) via autocorrelation.
**Default value:** 0.7
**Example:** `--regularity-threshold 0.75`

- **`--repeated-pattern-ratio`**
**Description:** Ratio used to estimate value repetitiveness in glitch areas.
**Default value:** 0.1
**Example:** `--repeated-pattern-ratio 0.12`

- **`--cluster-gap-threshold`**
**Description:** Maximum allowed gap between two glitch clusters to consider them as belonging to the same group.
**Default value:** 0.05
**Example:** `--cluster-gap-threshold 0.04`

- **`--dynamic`**
**Description:** Activates dynamic threshold mode, which calculates thresholds from local image statistics instead of using fixed values.
**Type:** Flag (no value needed)
**Example:** `--dynamic`

---

## Glitch Identification Strategies and Heuristics

1. **Threshold-Based Detection:**
   - **Fixed Thresholds:**  
     Uses overall image brightness statistics (mean and standard deviation) with fixed multipliers.
   - **Dynamic Thresholds:**  
     Calculates thresholds based on the local distribution of pixel differences (e.g., using the 90th or 80th percentile).  
     *Source:* Adaptive thresholding techniques in image processing literature.
  
2. **Anomaly Grouping:**  
   Groups consecutive rows or columns that exceed the threshold, considering a group as a “glitch region.” This helps filter out noise.
  
3. **Feature Extraction:**
   - **Glitch Positions & Widths:**  
     Identifies where glitches occur and their extent. Narrow glitches (small width) are considered well-aligned.
   - **Glitch Intensity:**  
     Measures the average brightness within a glitch. Brighter glitches compared to the overall mean can indicate buffer issues.
   - **Periodicity:**  
     Uses autocorrelation to assess if glitches occur at regular intervals. Regularity (high autocorrelation) suggests systematic errors such as buffer corruption.
     *Source:* Fundamentals of autocorrelation in signal processing.
  
4. **Heuristic Classification:**
   - **Buffer Corruption:**  
     Suggested when glitches are narrow, occur periodically (high autocorrelation), and are brighter than the overall image. This may be due to a repeating error in data buffering.
   - **Memory Corruption:**  
     Suspected when glitch regions are very dark (low intensity) with little variation, indicating that pixel data may have been lost.
   - **Write Corruption:**  
     Applied as a default when neither of the above conditions is clearly met, suggesting errors during file writing or transfer.
   *Source:* Observations from digital imaging error analysis (see *Digital Image Processing* by Gonzalez and Woods).

---

## Usage

### Direct Execution

To run the script directly:
```bash
python3 analyse_tiff.py image.tiff
```
For dynamic thresholding:
```
python3 analyse_tiff.py image.tiff --dynamic
```
Additional parameters (e.g., --max-group-size 30) can be passed as needed:
```
python3 analyse_tiff.py image.tiff --dynamic --max-group-size 30
```

### Using the Bash Wrapper

A Bash wrapper script (e.g., run_analysis.sh) is provided to simplify environment setup:

**Make it executable:**
```
chmod +x run_analysis.sh
```

**Run it with the TIFF file:**
```
    ./run_analysis.sh image.tiff --dynamic
```

Extra parameters are passed directly to the Python script.


This TIFF Analysis Tool employs a series of methods—from metadata extraction and luminance computation to glitch detection and pattern analysis—to diagnose potential corruption in images. The tool uses both fixed and dynamic threshold strategies and leverages features such as glitch intensity, periodicity, and alignment to heuristically classify the origin of the corruption. 

Feel free to experiment with the parameters and review the debug output to better understand how the tool works with your images.
