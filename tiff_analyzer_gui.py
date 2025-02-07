#!/usr/bin/env python3
import sys
import os
from datetime import datetime
from typing import Optional
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QProgressBar, QCheckBox, QSpinBox, QDoubleSpinBox,
                            QTabWidget, QTextEdit, QScrollArea, QFrame, QSlider)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.ndimage import zoom

from analyse_tiff import analyze_tiff, compute_luminance

class AnalysisWorker(QThread):
    """Thread worker for TIFF analysis with detailed progress"""
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int, str)  # Progress percentage and description
    error = pyqtSignal(str)

    def __init__(self, file_path: str, use_dynamic: bool):
        super().__init__()
        self.file_path = file_path
        self.use_dynamic = use_dynamic

    def run(self):
        try:
            # Analysis steps with descriptions
            steps = [
                (10, "Loading file..."),
                (20, "Analyzing TIFF structure..."),
                (30, "Checking metadata..."),
                (40, "Scanning horizontal lines..."),
                (60, "Scanning vertical lines..."),
                (80, "Processing patterns..."),
                (90, "Generating report..."),
            ]
            
            for progress, description in steps:
                self.progress.emit(progress, description)
                self.msleep(100)

            results = analyze_tiff(self.file_path, self.use_dynamic)
            self.progress.emit(100, "Analysis complete")
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

class HeatmapWidget(QWidget):
    """Widget for displaying heatmaps with adjustable transparency and optimized size"""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Add alpha slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Heatmap Transparency:"))
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(50)  # Default 50% transparency
        self.alpha_slider.valueChanged.connect(self.update_alpha)
        slider_layout.addWidget(self.alpha_slider)
        self.alpha_label = QLabel("50%")
        slider_layout.addWidget(self.alpha_label)
        layout.addLayout(slider_layout)
        
        # Matplotlib figure with reduced size
        self.figure, self.ax = plt.subplots(figsize=(6, 4), dpi=80)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Store data for replotting
        self.current_img_array = None
        self.current_mask = None
        self.current_title = None
        
        # Maximum dimensions for display
        self.max_display_size = (800, 600)
        
    def resize_array(self, array):
        """Resize array if it's too large"""
        if array is None:
            return None
            
        current_height, current_width = array.shape[:2]
        max_width, max_height = self.max_display_size
        
        # Calculate scaling factor
        scale_width = min(1.0, max_width / current_width)
        scale_height = min(1.0, max_height / current_height)
        scale = min(scale_width, scale_height)
        
        # Only resize if necessary
        if scale < 1.0:
            new_width = int(current_width * scale)
            new_height = int(current_height * scale)
            factors = [scale, scale] + [1] * (array.ndim - 2)
            return zoom(array, factors, order=1)
        
        return array

    def plot_heatmap(self, img_array: np.ndarray, mask: np.ndarray, title: str):
        """Plot heatmap with size optimization"""
        # Store original data
        self.current_img_array = img_array
        self.current_mask = mask
        self.current_title = title
        
        # Clear previous plot
        self.ax.clear()
        
        # Convert to grayscale and resize
        gray_img = compute_luminance(img_array)
        resized_gray = self.resize_array(gray_img)
        resized_mask = self.resize_array(mask)
        
        # Plot resized data
        self.ax.imshow(resized_gray, cmap='gray')
        alpha = self.alpha_slider.value() / 100.0
        self.ax.imshow(resized_mask, cmap='jet', alpha=alpha)
        
        # Update title with dimensions
        dims = f"{img_array.shape[1]}x{img_array.shape[0]}"
        display_dims = f"{resized_gray.shape[1]}x{resized_gray.shape[0]}"
        if dims != display_dims:
            self.ax.set_title(f"{title}\nOriginal: {dims} - Display: {display_dims}")
        else:
            self.ax.set_title(title)
        
        # Remove axes
        self.ax.axis('off')
        
        # Tight layout to optimize space
        self.figure.tight_layout()
        self.canvas.draw()

    def update_alpha(self):
        """Update transparency"""
        alpha_value = self.alpha_slider.value()
        self.alpha_label.setText(f"{alpha_value}%")
        if self.current_img_array is not None:
            self.plot_heatmap(self.current_img_array, self.current_mask, self.current_title)

class SettingsWidget(QWidget):
    """Widget for analysis settings"""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        
        # Dynamic parameters
        self.dynamic_cb = QCheckBox("Use Dynamic Threshold")
        self.dynamic_cb.setChecked(True)  # Default to True
        layout.addWidget(self.dynamic_cb)
        
        # Base parameters with spinboxes
        params_layout = QVBoxLayout()
        
        # Base threshold
        base_threshold_layout = QHBoxLayout()
        base_threshold_layout.addWidget(QLabel("Base Threshold Factor:"))
        self.base_threshold_spin = QDoubleSpinBox()
        self.base_threshold_spin.setRange(0.0, 1.0)
        self.base_threshold_spin.setValue(0.15)
        self.base_threshold_spin.setSingleStep(0.01)
        base_threshold_layout.addWidget(self.base_threshold_spin)
        params_layout.addLayout(base_threshold_layout)
        
        # Group size
        group_size_layout = QHBoxLayout()
        group_size_layout.addWidget(QLabel("Max Group Size:"))
        self.group_size_spin = QSpinBox()
        self.group_size_spin.setRange(1, 100)
        self.group_size_spin.setValue(20)
        group_size_layout.addWidget(self.group_size_spin)
        params_layout.addLayout(group_size_layout)
        
        layout.addLayout(params_layout)
        self.setLayout(layout)

class TIFFAnalyzerGUI(QMainWindow):
    """Main window of the application"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TIFF Analyzer")
        self.setGeometry(100, 100, 1200, 800)
        
        self.current_file: Optional[str] = None
        self.current_results: Optional[dict] = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel (controls)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # File selection
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.file_button = QPushButton("Select TIFF File")
        self.file_button.clicked.connect(self.select_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_button)
        left_layout.addLayout(file_layout)
        
        # Settings
        self.settings_widget = SettingsWidget()
        left_layout.addWidget(self.settings_widget)
        
        # Progress section
        progress_group = QFrame()
        progress_layout = QVBoxLayout(progress_group)
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Ready")
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)
        left_layout.addWidget(progress_group)
        
        # Analyze button
        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self.start_analysis)
        self.analyze_button.setEnabled(False)
        left_layout.addWidget(self.analyze_button)
        
        main_layout.addWidget(left_panel, stretch=1)
        
        # Right panel (results)
        right_panel = QTabWidget()
        
        # Report tab
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        right_panel.addTab(self.report_text, "Report")
        
        # Heatmap tabs
        self.horizontal_heatmap = HeatmapWidget()
        right_panel.addTab(self.horizontal_heatmap, "Horizontal Glitches")
        
        self.vertical_heatmap = HeatmapWidget()
        right_panel.addTab(self.vertical_heatmap, "Vertical Glitches")
        
        main_layout.addWidget(right_panel, stretch=2)
    
    def select_file(self):
        """Open file selection dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select TIFF File",
            "",
            "TIFF Images (*.tif *.tiff)"
        )
        if file_path:
            self.current_file = file_path
            self.file_label.setText(os.path.basename(file_path))
            self.analyze_button.setEnabled(True)
    
    def start_analysis(self):
        """Start analysis in separate thread"""
        if not self.current_file:
            return
        
        self.analyze_button.setEnabled(False)
        self.progress_bar.setValue(0)
        
        self.worker = AnalysisWorker(
            self.current_file,
            self.settings_widget.dynamic_cb.isChecked()
        )
        self.worker.finished.connect(self.analysis_finished)
        self.worker.progress.connect(self.update_progress)
        self.worker.error.connect(self.handle_error)
        
        self.worker.start()
    
    def update_progress(self, value: int, description: str):
        """Update progress bar and description"""
        self.progress_bar.setValue(value)
        self.progress_label.setText(description)
    
    def update_report(self, results):
        """Update analysis report with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = []
        report.append(f"=== TIFF Analysis Report ===")
        report.append(f"Analysis Date: {timestamp}\n")
        
        report.append(f"File: {self.current_file}")
        report.append(f"Size: {results['file_size']:,} bytes\n")
        
        report.append(f"Overall validity: {'✓' if results['valid'] else '✗'}")
        
        report.append("\nValidations:")
        report.append(f"- TIFF header: {'✓' if results['header_valid'] else '✗'}")
        report.append(f"- Image data: {'✓' if results['data_valid'] else '✗'}")
        report.append(f"- IFD structure: {'✓' if results['ifd_valid'] else '✗'}")
        report.append(f"- Pixel integrity: {'✓' if results.get('pixel_valid', False) else '✗'}\n")
        
        if results['tiff_info']:
            info = results['tiff_info']
            report.append("TIFF Information:")
            report.append(f"- Format: {info.get('format', 'N/A')}")
            report.append(f"- Mode: {info.get('mode', 'N/A')}")
            if 'size' in info:
                report.append(f"- Dimensions: {info['size'][0]}x{info['size'][1]} pixels")
        
        if results.get('statistics'):
            stats = results['statistics']
            report.append("\nImage Statistics:")
            report.append(f"- Mean: {stats['mean']:.2f}")
            report.append(f"- Std Dev: {stats['std']:.2f}")
            report.append(f"- Min: {stats['min']:.2f}")
            report.append(f"- Max: {stats['max']:.2f}")
        
        self.report_text.setText('\n'.join(report))
    
    def analysis_finished(self, results):
        """Handle analysis completion"""
        self.current_results = results
        self.analyze_button.setEnabled(True)
        
        self.update_report(results)
        
        if 'img_array' in results:
            if results.get('horizontal_glitch_info', {}).get('detected'):
                mask_horiz = np.zeros_like(compute_luminance(results['img_array']))
                for start, end in results['horizontal_glitch_info']['lines']:
                    mask_horiz[start:end+1, :] = 1
                self.horizontal_heatmap.plot_heatmap(
                    results['img_array'],
                    mask_horiz,
                    'Horizontal Glitches'
                )
            
            if results.get('vertical_glitch_info', {}).get('detected'):
                mask_vert = np.zeros_like(compute_luminance(results['img_array']))
                for start, end in results['vertical_glitch_info']['columns']:
                    mask_vert[:, start:end+1] = 1
                self.vertical_heatmap.plot_heatmap(
                    results['img_array'],
                    mask_vert,
                    'Vertical Glitches'
                )
    
    def handle_error(self, error_msg):
        """Handle analysis errors"""
        self.analyze_button.setEnabled(True)
        self.progress_label.setText("Error")
        self.report_text.setText(f"Error during analysis:\n{error_msg}")

def main():
    app = QApplication(sys.argv)
    window = TIFFAnalyzerGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()