#!/bin/bash

# Définition des chemins
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$SCRIPT_DIR/tiff_venv"
PYTHON_SCRIPT="$SCRIPT_DIR/analyse_tiff.py"

# Fonction pour afficher les messages d'erreur en rouge
error() {
    echo -e "\033[31mErreur: $1\033[0m"
    exit 1
}

# Fonction pour afficher les messages de succès en vert
success() {
    echo -e "\033[32m$1\033[0m"
}

# Vérification des arguments
if [ $# -ne 1 ]; then
    error "Usage: $0 <fichier.tiff>"
fi

# Vérification que le fichier existe
if [ ! -f "$1" ]; then
    error "Le fichier $1 n'existe pas"
fi

# Vérification que le fichier est bien un .tiff
if [[ ! "$1" =~ \.(tiff|TIFF|tif|TIF)$ ]]; then
    error "Le fichier doit être un fichier TIFF (.tiff, .tif)"
fi

# Vérification de la présence de Python
if ! command -v python3 &> /dev/null; then
    error "Python3 n'est pas installé sur votre système"
fi

# Création et activation du venv si nécessaire
if [ ! -d "$VENV_DIR" ]; then
    echo "Création de l'environnement virtuel..."
    python3 -m venv "$VENV_DIR" || error "Impossible de créer l'environnement virtuel"
    
    echo "Installation des dépendances..."
    source "$VENV_DIR/bin/activate" || error "Impossible d'activer l'environnement virtuel"
    pip install --upgrade pip > /dev/null
    pip install Pillow numpy scipy tqdm > /dev/null || error "Impossible d'installer les dépendances"
    success "Installation terminée avec succès"
else
    source "$VENV_DIR/bin/activate" || error "Impossible d'activer l'environnement virtuel"
fi

# Vérification que le script Python existe
if [ ! -f "$PYTHON_SCRIPT" ]; then
    error "Le script Python n'existe pas: $PYTHON_SCRIPT"
fi

# Lancement du script Python
python "$PYTHON_SCRIPT" "$1"

# Désactivation du venv
deactivate

exit 0