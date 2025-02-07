#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Define paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$SCRIPT_DIR/tiff_venv"
GUI_SCRIPT="$SCRIPT_DIR/tiff_analyzer_gui.py"
CLI_SCRIPT="$SCRIPT_DIR/analyse_tiff.py"

# Function to display error messages in red
error() {
    echo -e "\033[31mError: $1\033[0m"
    exit 1
}

# Function to display success messages in green
success() {
    echo -e "\033[32m$1\033[0m"
}

# Check if Python3 is installed
if ! command -v python3 &> /dev/null; then
    error "Python3 is not installed on your system"
fi

# Create and activate the virtual environment if necessary
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR" || error "Unable to create the virtual environment"
    
    echo "Installing dependencies..."
    source "$VENV_DIR/bin/activate" || error "Unable to activate the virtual environment"
    pip install --upgrade pip > /dev/null
    pip install Pillow numpy scipy tqdm matplotlib PyQt6 > /dev/null || error "Unable to install dependencies"
    success "Installation completed successfully"
else
    source "$VENV_DIR/bin/activate" || error "Unable to activate the virtual environment"
fi

# Check for --gui flag
if [[ "${1:-}" == "--gui" ]]; then
    # Verify that the GUI script exists
    if [ ! -f "$GUI_SCRIPT" ]; then
        error "The GUI script does not exist: $GUI_SCRIPT"
    fi
    # Launch GUI version
    python3 "$GUI_SCRIPT"
else
    # Check that at least one argument is provided (the TIFF file)
    if [ "$#" -lt 1 ]; then
        error "Usage: $0 [--gui] or $0 <file.tiff> [additional parameters]"
    fi

    FILE_PATH="$1"
    shift 1  # Remove the first argument

    # Verify that the file exists
    if [ ! -f "$FILE_PATH" ]; then
        error "File $FILE_PATH does not exist"
    fi

    # Verify that the file has a TIFF extension (case-insensitive)
    if [[ ! "$FILE_PATH" =~ \.(tiff|TIF|tif|TIFF)$ ]]; then
        error "The file must be a TIFF file (.tiff, .tif)"
    fi

    # Verify that the Python script exists
    if [ ! -f "$CLI_SCRIPT" ]; then
        error "The Python script does not exist: $CLI_SCRIPT"
    fi

    # Launch the CLI version with the file and any additional parameters
    python3 "$CLI_SCRIPT" "$FILE_PATH" "$@"
fi

# Deactivate the virtual environment
deactivate

exit 0