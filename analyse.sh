#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Define paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$SCRIPT_DIR/tiff_venv"
PYTHON_SCRIPT="$SCRIPT_DIR/analyse_tiff.py"

# Function to display error messages in red
error() {
    echo -e "\033[31mError: $1\033[0m"
    exit 1
}

# Function to display success messages in green
success() {
    echo -e "\033[32m$1\033[0m"
}

# Check that at least one argument is provided (the TIFF file)
if [ "$#" -lt 1 ]; then
    error "Usage: $0 <file.tiff> [additional parameters]"
fi

FILE_PATH="$1"
shift 1  # Remove the first argument (file path) so that "$@" holds additional parameters

# Verify that the file exists
if [ ! -f "$FILE_PATH" ]; then
    error "File $FILE_PATH does not exist"
fi

# Verify that the file has a TIFF extension (case-insensitive)
if [[ ! "$FILE_PATH" =~ \.(tiff|TIF|tif|TIFF)$ ]]; then
    error "The file must be a TIFF file (.tiff, .tif)"
fi

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
    pip install Pillow numpy scipy tqdm > /dev/null || error "Unable to install dependencies"
    success "Installation completed successfully"
else
    source "$VENV_DIR/bin/activate" || error "Unable to activate the virtual environment"
fi

# Verify that the Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    error "The Python script does not exist: $PYTHON_SCRIPT"
fi

# Launch the Python script with the file and any additional parameters
python3 "$PYTHON_SCRIPT" "$FILE_PATH" "$@"

# Deactivate the virtual environment
deactivate

exit 0
