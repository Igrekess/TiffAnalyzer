# TiffAnalyzer

TiffAnalyzer est un outil rapidement conçu suite à un problème rencontré par un client pour analyser des fichiers TIFF et détecter des anomalies telles que les corruptions de données, les glitches horizontaux ou verticaux, et d'autres problèmes potentiels et tenter de déterminer leurs origines. Il combine un script Python pour l'analyse des fichiers avec un script Bash facilitant l'installation et l'exécution dans un environnement virtuel. Ce projet est en cours de développement, aucune garantie sur la validité des résultats n'est fournis à ce stade. 

## Fonctionnalités
- Analyse détaillée des fichiers TIFF pour valider leur intégrité.
- Détection des glitches horizontaux et verticaux, avec identification des patterns de corruption.
- Rapport complet des anomalies détectées, incluant leur sévérité et localisation.
- Création et gestion automatique d'un environnement virtuel Python pour une configuration simplifiée.

## Prérequis
- Python 3.x
- Modules Python : `Pillow`, `numpy`, `scipy`, `tqdm`
- Bash (pour exécuter le script de lancement)

## Installation
1. Clonez ce dépôt :
    ```bash
    git clone https://github.com/votre-utilisateur/tiffanalyzer.git
    cd tiffanalyzer
    ```

2. Assurez-vous que Python 3 est installé sur votre système :
    ```bash
    python3 --version
    ```

3. Donnez les permissions d'exécution au script Bash :
    ```bash
    chmod +x analyse.sh
    ```
## Utilisation
Pour analyser un fichier TIFF, exécutez simplement le script Bash :
```bash
./analyse.sh <chemin-vers-le-fichier.tiff>
 ```
Le script :

    Vérifie l'existence et le format du fichier.
    Configure automatiquement un environnement virtuel Python si nécessaire.
    Lance l'analyse avec le script Python analyse_tiff.py.

Résultats

Les résultats de l'analyse sont affichés directement dans la console, avec des informations détaillées sur les éventuelles corruptions détectées.
Exemple

./analyse.sh example.tiff

Structure du projet

    analyse_tiff.py : Le script Python principal pour l'analyse des fichiers TIFF.
    analyse.sh : Le script Bash pour l'installation et l'exécution simplifiée.
    README.md : Documentation du projet.

Licence

Ce projet est sous licence MIT. Consultez le fichier LICENSE pour plus de détails.

