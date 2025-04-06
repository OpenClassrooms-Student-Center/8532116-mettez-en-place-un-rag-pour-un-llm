# utils/data_loader.py
import os
import requests
import zipfile
import io
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Fonctions d'extraction de texte (similaires à votre simple_indexer.py) ---

def extract_text_from_pdf(file_path: str) -> Optional[str]:
    """Extrait le texte d'un fichier PDF."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        text = "".join(page.extract_text() + "\n" for page in reader.pages if page.extract_text())
        logging.info(f"Texte extrait de PDF: {file_path} ({len(text)} caractères)")
        return text
    except Exception as e:
        logging.error(f"Erreur extraction PDF {file_path}: {e}")
        return None

def extract_text_from_docx(file_path: str) -> Optional[str]:
    """Extrait le texte d'un fichier Word DOCX."""
    try:
        import docx
        doc = docx.Document(file_path)
        text = "\n".join(para.text for para in doc.paragraphs if para.text)
        logging.info(f"Texte extrait de DOCX: {file_path} ({len(text)} caractères)")
        return text
    except Exception as e:
        logging.error(f"Erreur extraction DOCX {file_path}: {e}")
        return None

def extract_text_from_txt(file_path: str) -> Optional[str]:
    """Extrait le texte d'un fichier texte brut."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        logging.info(f"Texte extrait de TXT: {file_path} ({len(text)} caractères)")
        return text
    except Exception as e:
        logging.error(f"Erreur extraction TXT {file_path}: {e}")
        return None

def extract_text_from_csv(file_path: str) -> Optional[str]:
    """Extrait le texte d'un fichier CSV (convertit en string)."""
    try:
        import pandas as pd
        # Lire avec une gestion d'erreur d'encodage plus robuste
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1') # Essayer un autre encodage courant
        except Exception as read_e:
             logging.warning(f"Erreur lecture CSV {file_path}: {read_e}. Tentative avec séparateur ';'")
             try:
                 df = pd.read_csv(file_path, sep=';')
             except UnicodeDecodeError:
                  df = pd.read_csv(file_path, sep=';', encoding='latin1')
             except Exception as read_e2:
                  logging.error(f"Impossible de lire le CSV {file_path}: {read_e2}")
                  return None

        text = df.to_string()
        logging.info(f"Texte extrait de CSV: {file_path} ({len(text)} caractères)")
        return text
    except ImportError:
        logging.warning("Pandas non installé. Impossible de lire les fichiers CSV.")
        return None
    except Exception as e:
        logging.error(f"Erreur extraction CSV {file_path}: {e}")
        return None

def extract_text_from_excel(file_path: str) -> Optional[str]:
    """Extrait le texte d'un fichier Excel (convertit en string)."""
    try:
        import pandas as pd
        df = pd.read_excel(file_path, sheet_name=None) # Lire toutes les feuilles
        text = ""
        if isinstance(df, dict): # Si plusieurs feuilles
             for sheet_name, sheet_df in df.items():
                 text += f"--- Feuille: {sheet_name} ---\n{sheet_df.to_string()}\n\n"
        else: # Si une seule feuille
             text = df.to_string()
        logging.info(f"Texte extrait de Excel: {file_path} ({len(text)} caractères)")
        return text
    except ImportError:
        logging.warning("Pandas ou openpyxl non installé. Impossible de lire les fichiers Excel.")
        return None
    except Exception as e:
        logging.error(f"Erreur extraction Excel {file_path}: {e}")
        return None

# --- Fonctions de chargement ---

def download_and_extract_zip(url: str, output_dir: str) -> bool:
    """Télécharge un fichier ZIP depuis une URL et l'extrait."""
    if not url:
        logging.warning("Aucune URL fournie pour le téléchargement.")
        return False
    try:
        logging.info(f"Téléchargement des données depuis {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status() # Vérifie les erreurs HTTP

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True) # Crée le dossier de sortie

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            logging.info(f"Extraction du contenu dans {output_dir}...")
            z.extractall(output_dir)
        logging.info("Téléchargement et extraction terminés.")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Erreur de téléchargement: {e}")
        return False
    except zipfile.BadZipFile:
        logging.error("Le fichier téléchargé n'est pas un ZIP valide.")
        return False
    except Exception as e:
        logging.error(f"Erreur inattendue lors du téléchargement/extraction: {e}")
        return False

def load_and_parse_files(input_dir: str) -> List[Dict[str, any]]:
    """
    Charge et parse récursivement les fichiers d'un répertoire.
    Retourne une liste de dictionnaires, chacun représentant un document.
    """
    documents = []
    input_path = Path(input_dir)
    if not input_path.is_dir():
        logging.error(f"Le répertoire d'entrée '{input_dir}' n'existe pas.")
        return []

    logging.info(f"Parcours du répertoire source: {input_dir}")
    for file_path in input_path.rglob("*.*"): # Parcourt tous les fichiers récursivement
        if file_path.is_file():
            relative_path = file_path.relative_to(input_path)
            # Le nom du dossier source est le premier composant du chemin relatif
            source_folder = relative_path.parts[0] if len(relative_path.parts) > 1 else "root"
            ext = file_path.suffix.lower()
            text = None

            logging.debug(f"Traitement du fichier: {relative_path} (Dossier source: {source_folder})")

            if ext == ".pdf":
                text = extract_text_from_pdf(str(file_path))
            elif ext == ".docx":
                text = extract_text_from_docx(str(file_path))
            elif ext == ".txt":
                text = extract_text_from_txt(str(file_path))
            elif ext == ".csv":
                text = extract_text_from_csv(str(file_path))
            elif ext in [".xlsx", ".xls"]:
                text = extract_text_from_excel(str(file_path))
            else:
                logging.warning(f"Type de fichier non supporté ignoré: {relative_path}")
                continue

            if text:
                documents.append({
                    "page_content": text,
                    "metadata": {
                        "source": str(relative_path), # Chemin relatif comme source
                        "filename": file_path.name,
                        "category": source_folder, # Utilise le nom du dossier comme catégorie/metadata
                        "full_path": str(file_path.resolve()) # Chemin absolu si besoin
                    }
                })
            else:
                 logging.warning(f"Aucun texte n'a pu être extrait de {relative_path}")

    logging.info(f"{len(documents)} documents chargés et parsés.")
    return documents