import os
import subprocess
import sys

def convert_documents_to_markdown(input_dir, output_dir):
    """
    Convertit tous les documents d'un répertoire d'entrée en Markdown
    dans un répertoire de sortie en utilisant l'outil Docling via subprocess.

    Args:
        input_dir (str): Le chemin vers le répertoire contenant les documents source.
        output_dir (str): Le chemin vers le répertoire où les fichiers Markdown seront sauvegardés.
    """
    # Vérifier si le répertoire d'entrée existe
    if not os.path.isdir(input_dir):
        print(f"Erreur : Le répertoire d'entrée '{input_dir}' n'existe pas.")
        return

    # Créer le répertoire de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Répertoire de sortie '{output_dir}' créé.")
        except OSError as e:
            print(f"Erreur lors de la création du répertoire de sortie '{output_dir}': {e}")
            return

    print(f"Début de la conversion des documents de '{input_dir}' vers '{output_dir}'...")

    # Parcourir tous les fichiers dans le répertoire d'entrée
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)

        # S'assurer que c'est bien un fichier (et non un sous-répertoire)
        if os.path.isfile(input_path):
            # Construire le nom du fichier de sortie avec l'extension .md
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}.md"
            output_path = os.path.join(output_dir, output_filename)

            print(f"\nTraitement de '{filename}'...")

            # Construire la commande docling
            # Syntaxe : docling <input_file> -o <output_file> -f <format>
            # Le format de sortie pour Markdown est 'md'
            cmd = ['docling', input_path, '-o', output_path, '-f', 'md']
            print(f"Exécution de la commande : {' '.join(cmd)}")

            try:
                # Exécuter la commande docling
                result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
                print(f"Succès : '{filename}' converti en '{output_filename}'.")
                # Optionnel : afficher la sortie standard de docling si besoin
                # if result.stdout:
                #     print(f"Sortie Docling : {result.stdout}")

            except subprocess.CalledProcessError as e:
                # Erreur lors de l'exécution de docling
                print(f"Erreur lors de la conversion de '{filename}'.")
                print(f"Commande échouée : {' '.join(e.cmd)}")
                print(f"Code de retour : {e.returncode}")
                print(f"Sortie d'erreur (stderr) : {e.stderr}")
            except FileNotFoundError:
                # Erreur si la commande 'docling' n'est pas trouvée
                print(f"Erreur : La commande 'docling' n'a pas été trouvée.")
                print("Vérifiez que Docling est correctement installé et que son exécutable est dans le PATH système.")
                print("Installation : pip install docling")
                return # Arrêter le script si docling n'est pas trouvé
            except Exception as e:
                # Capturer toute autre exception potentielle
                print(f"Une erreur inattendue est survenue lors du traitement de '{filename}': {e}")

    print(f"\nConversion terminée. Les fichiers Markdown se trouvent dans '{output_dir}'.")

# --- Configuration ---
# Remplacez ces chemins par les vôtres
# INPUT_DIRECTORY : Le dossier où vous avez décompressé 'inputs.zip'
# OUTPUT_DIRECTORY : Le dossier où vous voulez sauvegarder les fichiers .md

INPUT_DIRECTORY = 'inputs'  # Adaptez si vous avez décompressé ailleurs
OUTPUT_DIRECTORY = 'markdown_outputs' # Nom du dossier de sortie

# --- Exécution du script ---
if __name__ == "__main__":
    # Vérifier si Python 3 est utilisé (subprocess.run a des arguments différents en Python 2)
    if sys.version_info < (3, 5):
        print("Erreur : Ce script nécessite Python 3.5 ou une version ultérieure.")
    else:
        # Donner le chemin absolu pour éviter les ambiguïtés
        abs_input_dir = os.path.abspath(INPUT_DIRECTORY)
        abs_output_dir = os.path.abspath(OUTPUT_DIRECTORY)
        convert_documents_to_markdown(abs_input_dir, abs_output_dir)