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
    Extrait les informations détaillées d'une image TIFF.
    """
    info = {}
    try:
        # Informations de base
        info['format'] = img.format
        info['mode'] = img.mode
        info['size'] = img.size
        
        # Métadonnées TIFF
        if hasattr(img, 'tag'):
            tags = img.tag
            if 258 in tags:  # BitsPerSample
                info['bits_per_sample'] = tags[258]
            if 259 in tags:  # Compression
                compression_codes = {
                    1: "Non compressé",
                    2: "CCITT 1D",
                    3: "CCITT Group 3",
                    4: "CCITT Group 4",
                    5: "LZW",
                    6: "JPEG (ancien)",
                    7: "JPEG",
                    8: "Adobe Deflate",
                    32773: "PackBits"
                }
                comp_value = tags[259][0]
                info['compression'] = compression_codes.get(comp_value, f"Inconnu ({comp_value})")
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
                info['photometric'] = photo_codes.get(photo_value, f"Inconnu ({photo_value})")
            if 296 in tags:  # ResolutionUnit
                unit_codes = {1: "None", 2: "Pouces", 3: "Centimètres"}
                unit_value = tags[296][0]
                info['resolution_unit'] = unit_codes.get(unit_value, f"Inconnu ({unit_value})")
            if 282 in tags and 283 in tags:  # XResolution and YResolution
                info['dpi'] = (float(tags[282][0][0])/float(tags[282][0][1]),
                             float(tags[283][0][0])/float(tags[283][0][1]))
    except Exception as e:
        info['error'] = str(e)
    return info

def detect_vertical_glitches(img_array):
    """
    Détecte les lignes verticales de glitch dans l'image.
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
            luminance = 0.2989 * img_array[:,:,0] + 0.5870 * img_array[:,:,1] + 0.1140 * img_array[:,:,2]
        else:
            luminance = img_array
            
        luminance = luminance.T  # Transposer pour analyser les colonnes
        
        potential_glitches = []
        print("\nAnalyse des colonnes de l'image...")
        print("Recherche de glitches verticaux potentiels...")
        
        # Calculer la moyenne globale et l'écart-type pour des seuils adaptatifs
        global_mean = np.mean(luminance)
        global_std = np.std(luminance)
        base_threshold = global_mean * 0.15  # 15% de la moyenne

        for x in tqdm(range(1, width-1), desc="Scan des colonnes", unit="cols"):
            # Calculer les différences avec les colonnes adjacentes
            diff_prev = np.abs(luminance[x] - luminance[x-1])
            diff_next = np.abs(luminance[x] - luminance[x+1])
            
            # Seuils adaptatifs basés sur les statistiques locales
            local_threshold = base_threshold + (global_std * 0.5)  # Ajout de 50% de l'écart-type
            anomaly_prev = np.mean(diff_prev > local_threshold)
            anomaly_next = np.mean(diff_next > local_threshold)
            
            # Vérification plus stricte des anomalies
            if (anomaly_prev > 0.4 or anomaly_next > 0.4):  # Au moins 40% des pixels doivent être anormaux
                max_diff = max(np.max(diff_prev), np.max(diff_next))
                if max_diff > global_std * 2:  # La différence doit être significative
                    potential_glitches.append((x, max_diff))
        
        if potential_glitches:
            print("\nAnalyse des zones de glitch verticaux détectées...")
            current_start = potential_glitches[0][0]
            current_severity = potential_glitches[0][1]
            current_group = [current_start]
            glitch_count = 0
            
            for x, severity in potential_glitches[1:]:
                # Regroupement plus strict des colonnes
                if x - current_group[-1] <= 1:  # Les colonnes doivent être strictement consécutives
                    current_group.append(x)
                    current_severity = max(current_severity, severity)
                else:
                    if len(current_group) >= 2:  # Au moins 2 colonnes consécutives
                        if current_severity > global_std * 3:  # La sévérité doit être très significative
                            glitch_count += 1
                            range_size = max(current_group) - min(current_group) + 1
                            if range_size <= 20:  # Limiter la taille des zones de glitch
                                print(f"\r⚠️  Glitch vertical #{glitch_count} détecté - Colonnes {min(current_group)} à {max(current_group)} (intensité: {current_severity:.2f})", end="", flush=True)
                                glitch_info['columns'].append((min(current_group), max(current_group)))
                                glitch_info['severity'].append(current_severity)
                    current_start = x
                    current_group = [x]
                    current_severity = severity
            
            # Traiter le dernier groupe avec les mêmes critères
            if len(current_group) >= 2 and current_severity > global_std * 3:
                range_size = max(current_group) - min(current_group) + 1
                if range_size <= 20:
                    glitch_count += 1
                    print(f"\r⚠️  Glitch vertical #{glitch_count} détecté - Colonnes {min(current_group)} à {max(current_group)} (intensité: {current_severity:.2f})", end="", flush=True)
                    glitch_info['columns'].append((min(current_group), max(current_group)))
                    glitch_info['severity'].append(current_severity)
            
            print("\n")
        
        # Mettre à jour les informations finales
        if glitch_info['columns']:
            glitch_info['detected'] = True
            glitch_info['count'] = len(glitch_info['columns'])
            
            # Trier les glitches par sévérité
            sorted_glitches = sorted(zip(glitch_info['columns'], glitch_info['severity']), 
                                   key=lambda x: x[1], reverse=True)
            glitch_info['columns'] = [g[0] for g in sorted_glitches]
            glitch_info['severity'] = [g[1] for g in sorted_glitches]
            
            print(f"\nTotal : {glitch_info['count']} glitches verticaux détectés\n")
    
    except Exception as e:
        print(f"Erreur lors de la détection des glitches verticaux: {str(e)}")
    
    return glitch_info

def detect_horizontal_glitches(img_array):
    """
    Détecte les lignes horizontales de glitch dans l'image.
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
            luminance = 0.2989 * img_array[:,:,0] + 0.5870 * img_array[:,:,1] + 0.1140 * img_array[:,:,2]
        else:
            luminance = img_array
        
        potential_glitches = []
        print("\nAnalyse des lignes de l'image...")
        print("Recherche de glitches horizontaux potentiels...")
        
        # Calculer la moyenne globale et l'écart-type pour des seuils adaptatifs
        global_mean = np.mean(luminance)
        global_std = np.std(luminance)
        base_threshold = global_mean * 0.15  # 15% de la moyenne

        for y in tqdm(range(1, height-1), desc="Scan des lignes", unit="lignes"):
            # Calculer les différences avec les lignes adjacentes
            diff_prev = np.abs(luminance[y] - luminance[y-1])
            diff_next = np.abs(luminance[y] - luminance[y+1])
            
            # Seuils adaptatifs basés sur les statistiques locales
            local_threshold = base_threshold + (global_std * 0.5)  # Ajout de 50% de l'écart-type
            anomaly_prev = np.mean(diff_prev > local_threshold)
            anomaly_next = np.mean(diff_next > local_threshold)
            
            # Vérification plus stricte des anomalies
            if (anomaly_prev > 0.4 or anomaly_next > 0.4):  # Au moins 40% des pixels doivent être anormaux
                max_diff = max(np.max(diff_prev), np.max(diff_next))
                if max_diff > global_std * 2:  # La différence doit être significative
                    potential_glitches.append((y, max_diff))
        
        if potential_glitches:
            print("\nAnalyse des zones de glitch horizontaux détectées...")
            current_start = potential_glitches[0][0]
            current_severity = potential_glitches[0][1]
            current_group = [current_start]
            glitch_count = 0
            
            for y, severity in potential_glitches[1:]:
                # Regroupement plus strict des lignes
                if y - current_group[-1] <= 1:  # Les lignes doivent être strictement consécutives
                    current_group.append(y)
                    current_severity = max(current_severity, severity)
                else:
                    if len(current_group) >= 2:  # Au moins 2 lignes consécutives
                        if current_severity > global_std * 3:  # La sévérité doit être très significative
                            glitch_count += 1
                            range_size = max(current_group) - min(current_group) + 1
                            if range_size <= 20:  # Limiter la taille des zones de glitch
                                print(f"\r⚠️  Glitch horizontal #{glitch_count} détecté - Lignes {min(current_group)} à {max(current_group)} (intensité: {current_severity:.2f})", end="", flush=True)
                                glitch_info['lines'].append((min(current_group), max(current_group)))
                                glitch_info['severity'].append(current_severity)
                    current_start = y
                    current_group = [y]
                    current_severity = severity
            
            # Traiter le dernier groupe avec les mêmes critères
            if len(current_group) >= 2 and current_severity > global_std * 3:
                range_size = max(current_group) - min(current_group) + 1
                if range_size <= 20:
                    glitch_count += 1
                    print(f"\r⚠️  Glitch horizontal #{glitch_count} détecté - Lignes {min(current_group)} à {max(current_group)} (intensité: {current_severity:.2f})", end="", flush=True)
                    glitch_info['lines'].append((min(current_group), max(current_group)))
                    glitch_info['severity'].append(current_severity)
            
            print("\n")
        
        # Mettre à jour les informations finales
        if glitch_info['lines']:
            glitch_info['detected'] = True
            glitch_info['count'] = len(glitch_info['lines'])
            
            # Trier les glitches par sévérité
            sorted_glitches = sorted(zip(glitch_info['lines'], glitch_info['severity']), 
                                   key=lambda x: x[1], reverse=True)
            glitch_info['lines'] = [g[0] for g in sorted_glitches]
            glitch_info['severity'] = [g[1] for g in sorted_glitches]
            
            print(f"\nTotal : {glitch_info['count']} glitches horizontaux détectés\n")
    
    except Exception as e:
        print(f"Erreur lors de la détection des glitches horizontaux: {str(e)}")
    
    return glitch_info

def analyze_corruption_pattern(img_array, glitch_lines, orientation='horizontal'):
    """
    Analyse détaillée des patterns de corruption pour identifier leur origine probable.
    """
    pattern_info = {
        'type': 'inconnu',
        'confidence': 0,
        'details': [],
        'probable_cause': None
    }
    
    try:
        if len(img_array.shape) == 3:
            luminance = 0.2989 * img_array[:,:,0] + 0.5870 * img_array[:,:,1] + 0.1140 * img_array[:,:,2]
        else:
            luminance = img_array
            
        if orientation == 'vertical':
            luminance = luminance.T
        
        # Analyse de la régularité des intervalles entre glitches
        intervals = []
        for i in range(1, len(glitch_lines)):
            interval = glitch_lines[i][0] - glitch_lines[i-1][1]
            intervals.append(interval)
        
        if intervals:
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            regularity = 1 - (std_interval / mean_interval if mean_interval > 0 else 0)
            
            # Analyse des valeurs de pixels dans les zones corrompues
            glitch_values = []
            for start, end in glitch_lines:
                glitch_region = luminance[start:end+1, :]
                glitch_values.extend(glitch_region.flatten())
            
            # Caractéristiques des glitches
            is_periodic = regularity > 0.7
            has_zero_values = any(v == 0 for v in glitch_values)
            has_repeated_pattern = len(set(glitch_values)) < len(glitch_values) * 0.1
            is_aligned = all(g[1]-g[0] < 5 for g in glitch_lines)
            
            # Identification du type de corruption
            direction = "verticale" if orientation == 'vertical' else "horizontale"
            if is_periodic and is_aligned:
                pattern_info['type'] = 'corruption_buffer'
                pattern_info['confidence'] = 0.8
                pattern_info['probable_cause'] = f"Erreur de buffer/cache lors de l'écriture ({direction})"
                pattern_info['details'].append(f"Intervalle moyen entre glitches: {mean_interval:.1f} pixels")
                
            elif has_repeated_pattern and has_zero_values:
                pattern_info['type'] = 'corruption_memoire'
                pattern_info['confidence'] = 0.7
                pattern_info['probable_cause'] = f"Problème de mémoire vive pendant le traitement ({direction})"
                pattern_info['details'].append("Motifs répétitifs détectés dans les zones corrompues")
                
            elif is_aligned and not is_periodic:
                pattern_info['type'] = 'corruption_ecriture'
                pattern_info['confidence'] = 0.6
                pattern_info['probable_cause'] = f"Interruption ou erreur pendant l'écriture du fichier ({direction})"
                pattern_info['details'].append("Corruptions alignées mais non périodiques")
            
            # Analyse de la distribution spatiale
            dimension = img_array.shape[1] if orientation == 'vertical' else img_array.shape[0]
            relative_positions = [(start/dimension, end/dimension) for start, end in glitch_lines]
            
            # Regroupement des corruptions
            clusters = []
            current_cluster = [relative_positions[0]]
            
            for pos in relative_positions[1:]:
                if pos[0] - current_cluster[-1][1] < 0.05:
                    current_cluster.append(pos)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [pos]
            clusters.append(current_cluster)
            
            # Analyse des clusters
            if len(clusters) == 1 and len(clusters[0]) > 5:
                pattern_info['details'].append("Corruptions concentrées dans une seule zone")
                if pattern_info['probable_cause']:
                    pattern_info['probable_cause'] += " - Probablement un événement unique"
            elif len(clusters) > 3:
                pattern_info['details'].append(f"Corruptions réparties en {len(clusters)} zones distinctes")
                if pattern_info['probable_cause']:
                    pattern_info['probable_cause'] += " - Problème potentiellement récurrent"
                    
    except Exception as e:
        pattern_info['details'].append(f"Erreur lors de l'analyse des patterns: {str(e)}")
    
    return pattern_info

def analyze_tiff(file_path):
    """
    Analyse une image TIFF et vérifie si elle présente des signes de corruption.
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
        print("\nVérification du fichier...")
        if not os.path.exists(file_path):
            results['errors'].append("Le fichier n'existe pas")
            return results
            
        results['file_size'] = os.path.getsize(file_path)
        
        print("Analyse de l'en-tête TIFF...")
        with open(file_path, 'rb') as f:
            header = f.read(8)
            if len(header) < 8:
                results['errors'].append("En-tête TIFF incomplet")
                return results
                
            byte_order = header[:2]
            if byte_order not in [b'II', b'MM']:
                results['errors'].append("Format d'en-tête invalide")
                return results
                
            results['header_valid'] = True
        
        print("Chargement et analyse de l'image...")
        with Image.open(file_path) as img:
            results['tiff_info'] = get_tiff_info(img)
            
            try:
                img.load()
                results['data_valid'] = True
                
                print("Conversion et préparation des données...")
                img_array = np.array(img)
                if img_array.size > 0:
                    # Détection des glitches horizontaux
                    results['horizontal_glitch_info'] = detect_horizontal_glitches(img_array)
                    if results['horizontal_glitch_info']['lines']:
                        results['horizontal_pattern_info'] = analyze_corruption_pattern(
                            img_array, 
                            results['horizontal_glitch_info']['lines'],
                            orientation='horizontal'
                        )
                    
                    # Détection des glitches verticaux
                    results['vertical_glitch_info'] = detect_vertical_glitches(img_array)
                    if results['vertical_glitch_info']['columns']:
                        results['vertical_pattern_info'] = analyze_corruption_pattern(
                            img_array, 
                            results['vertical_glitch_info']['columns'],
                            orientation='vertical'
                        )
                    
                    # L'image est valide si aucun type de glitch n'est détecté
                    results['pixel_valid'] = not (results['horizontal_glitch_info']['detected'] or 
                                                results['vertical_glitch_info']['detected'])
                    
                    print("Calcul des statistiques...")
                    results['statistics'] = {
                        'mean': float(np.mean(img_array)),
                        'std': float(np.std(img_array)),
                        'min': float(np.min(img_array)),
                        'max': float(np.max(img_array))
                    }
            except Exception as e:
                results['errors'].append(f"Erreur lors de l'analyse des données: {str(e)}")
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
        results['errors'].append(f"Erreur générale: {str(e)}")
    
    return results

def print_analysis_results(results):
    """
    Affiche les résultats de l'analyse de manière formatée.
    """
    print("\n=== Rapport d'analyse TIFF ===")
    print(f"Validité globale: {'✓' if results['valid'] else '✗'}")
    print(f"Taille du fichier: {results['file_size']:,} octets")
    
    print("\nValidations:")
    print(f"- En-tête TIFF: {'✓' if results['header_valid'] else '✗'}")
    print(f"- Données image: {'✓' if results['data_valid'] else '✗'}")
    print(f"- Structure IFD: {'✓' if results['ifd_valid'] else '✗'}")
    print(f"- Intégrité pixels: {'✓' if results.get('pixel_valid', False) else '✗'}")
    
    if results['tiff_info']:
        info = results['tiff_info']
        print("\nInformations TIFF:")
        print(f"- Format: {info.get('format', 'N/A')}")
        print(f"- Mode: {info.get('mode', 'N/A')}")
        if 'size' in info:
            print(f"- Dimensions: {info['size'][0]}x{info['size'][1]} pixels")
        if 'bits_per_sample' in info:
            print(f"- Bits par échantillon: {info['bits_per_sample']}")
        if 'compression' in info:
            print(f"- Compression: {info['compression']}")
        if 'photometric' in info:
            print(f"- Interprétation photométrique: {info['photometric']}")
        if 'resolution_unit' in info:
            print(f"- Unité de résolution: {info['resolution_unit']}")
        if 'dpi' in info:
            print(f"- Résolution: {info['dpi'][0]:.0f}x{info['dpi'][1]:.0f} DPI")

    if 'horizontal_glitch_info' in results:
        print("\nAnalyse des glitches horizontaux:")
        if results['horizontal_glitch_info']['detected']:
            print("⚠️ Glitches horizontaux détectés:")
            print(f"- Nombre de zones touchées: {results['horizontal_glitch_info']['count']}")
            print("- Position des glitches (par ordre de sévérité):")
            for i, ((start, end), severity) in enumerate(zip(results['horizontal_glitch_info']['lines'], 
                                                           results['horizontal_glitch_info']['severity'])):
                print(f"  • Lignes {start} à {end} (intensité: {severity:.2f})")
        else:
            print("✓ Aucun glitch horizontal détecté")

    if 'vertical_glitch_info' in results:
        print("\nAnalyse des glitches verticaux:")
        if results['vertical_glitch_info']['detected']:
            print("⚠️ Glitches verticaux détectés:")
            print(f"- Nombre de zones touchées: {results['vertical_glitch_info']['count']}")
            print("- Position des glitches (par ordre de sévérité):")
            for i, ((start, end), severity) in enumerate(zip(results['vertical_glitch_info']['columns'], 
                                                           results['vertical_glitch_info']['severity'])):
                print(f"  • Colonnes {start} à {end} (intensité: {severity:.2f})")
        else:
            print("✓ Aucun glitch vertical détecté")
    
    if 'statistics' in results:
        print("\nStatistiques des pixels:")
        print(f"- Moyenne: {results['statistics']['mean']:.2f}")
        print(f"- Écart-type: {results['statistics']['std']:.2f}")
        print(f"- Min: {results['statistics']['min']}")
        print(f"- Max: {results['statistics']['max']}")
    
    if 'horizontal_pattern_info' in results and results['horizontal_pattern_info']:
        print("\nAnalyse du pattern de corruption horizontale:")
        pattern = results['horizontal_pattern_info']
        if pattern['probable_cause']:
            print(f"Origine probable: {pattern['probable_cause']}")
            print(f"Confiance: {pattern['confidence']*100:.0f}%")
            if pattern['details']:
                print("\nDétails de l'analyse:")
                for detail in pattern['details']:
                    print(f"- {detail}")

    if 'vertical_pattern_info' in results and results['vertical_pattern_info']:
        print("\nAnalyse du pattern de corruption verticale:")
        pattern = results['vertical_pattern_info']
        if pattern['probable_cause']:
            print(f"Origine probable: {pattern['probable_cause']}")
            print(f"Confiance: {pattern['confidence']*100:.0f}%")
            if pattern['details']:
                print("\nDétails de l'analyse:")
                for detail in pattern['details']:
                    print(f"- {detail}")

    if results['errors']:
        print("\nErreurs détectées:")
        for error in results['errors']:
            print(f"- {error}")

def print_header():
    """Affiche l'en-tête du programme."""
    print("\n" + "="*50)
    print("{:^50}".format("Analyse des données TIFF"))
    print("{:^50}".format("by Yan Senez"))
    print("="*50 + "\n")

if __name__ == "__main__":
    print_header()
    
    if len(sys.argv) != 2:
        print("Usage: python analyse_tiff.py <fichier.tiff>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    results = analyze_tiff(file_path)
    print_analysis_results(results)