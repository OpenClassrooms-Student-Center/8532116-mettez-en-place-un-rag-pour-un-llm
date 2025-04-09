# MistralChat.py (version RAG)
import streamlit as st
import os
import logging
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv

# --- Importations depuis vos modules ---
try:
    from utils.config import (
        MISTRAL_API_KEY, MODEL_NAME, SEARCH_K,
        APP_TITLE, COMMUNE_NAME
    )
    from utils.vector_store import VectorStoreManager
except ImportError as e:
    st.error(f"Erreur d'importation: {e}. Vérifiez la structure de vos dossiers et les fichiers dans 'utils'.")
    st.stop()


# --- Configuration du Logging ---
# Note: Streamlit peut avoir sa propre gestion de logs. Configurer ici est une bonne pratique.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# --- Configuration de l'API Mistral ---
api_key = MISTRAL_API_KEY
model = MODEL_NAME

if not api_key:
    st.error("Erreur : Clé API Mistral non trouvée (MISTRAL_API_KEY). Veuillez la définir dans le fichier .env.")
    st.stop()

try:
    client = MistralClient(api_key=api_key)
    logging.info("Client Mistral initialisé.")
except Exception as e:
    st.error(f"Erreur lors de l'initialisation du client Mistral : {e}")
    logging.exception("Erreur initialisation client Mistral")
    st.stop()

# --- Chargement du Vector Store (mis en cache) ---
@st.cache_resource # Garde le manager chargé en mémoire pour la session
def get_vector_store_manager():
    logging.info("Tentative de chargement du VectorStoreManager...")
    try:
        manager = VectorStoreManager()
        # Vérifie si l'index a bien été chargé par le constructeur
        if manager.index is None or not manager.document_chunks:
            st.error("L'index vectoriel ou les chunks n'ont pas pu être chargés.")
            st.warning("Assurez-vous d'avoir exécuté 'python indexer.py' après avoir placé vos fichiers dans le dossier 'inputs'.")
            logging.error("Index Faiss ou chunks non trouvés/chargés par VectorStoreManager.")
            return None # Retourne None si échec
        logging.info(f"VectorStoreManager chargé avec succès ({manager.index.ntotal} vecteurs).")
        return manager
    except FileNotFoundError:
         st.error("Fichiers d'index ou de chunks non trouvés.")
         st.warning("Veuillez exécuter 'python indexer.py' pour créer la base de connaissances.")
         logging.error("FileNotFoundError lors de l'init de VectorStoreManager.")
         return None
    except Exception as e:
        st.error(f"Erreur inattendue lors du chargement du VectorStoreManager: {e}")
        logging.exception("Erreur chargement VectorStoreManager")
        return None

vector_store_manager = get_vector_store_manager()

# --- Prompt Système pour RAG ---
# Adaptez ce prompt selon vos besoins
SYSTEM_PROMPT = f"""Tu es un assistant virtuel expert pour la mairie de {COMMUNE_NAME}.
Ta mission est de répondre aux questions des citoyens de manière précise, factuelle et polie, en te basant **exclusivement** sur les informations fournies dans le CONTEXTE ci-dessous.

Instructions importantes:
1.  **Base ta réponse UNIQUEMENT sur le CONTEXTE.** N'invente aucune information.
2.  Si le CONTEXTE contient l'information pour répondre à la QUESTION, synthétise-la clairement.
3.  Si le CONTEXTE ne contient PAS d'information pertinente pour répondre à la QUESTION, réponds poliment que tu n'as pas trouvé l'information dans la base de connaissances actuelle et suggère de contacter directement les services de la mairie. Ne cherche pas la réponse ailleurs.
4.  Ne réponds pas à des questions hors sujet (non liées à la mairie ou aux informations du contexte).
5.  Si possible, mentionne la source (par exemple, le nom du fichier) si elle est indiquée dans le contexte.
6.  Garde tes réponses concises et faciles à comprendre.

CONTEXTE FOURNI:
---
{{context_str}}
---

QUESTION DU CITOYEN:
{{question}}

RÉPONSE DE L'ASSISTANT MAIRIE:"""


# --- Initialisation de l'historique de conversation ---
if "messages" not in st.session_state:
    # Message d'accueil initial
    st.session_state.messages = [{"role": "assistant", "content": f"Bonjour, je suis l'assistant virtuel de la mairie de {COMMUNE_NAME}. Comment puis-je vous aider concernant nos services (basé sur ma base de connaissances) ?"}]

# --- Fonctions ---

def generer_reponse(prompt_messages: list[ChatMessage]) -> str:
    """
    Envoie le prompt (qui inclut maintenant le contexte) à l'API Mistral.
    """
    if not prompt_messages:
         logging.warning("Tentative de génération de réponse avec un prompt vide.")
         return "Je ne peux pas traiter une demande vide."
    try:
        logging.info(f"Appel à l'API Mistral modèle '{model}' avec {len(prompt_messages)} message(s).")
        # Log le contenu du prompt (peut être long) - commenter si trop verbeux
        # logging.debug(f"Prompt envoyé à l'API: {prompt_messages}")

        response = client.chat(
            model=model,
            messages=prompt_messages,
            temperature=0.1, # Température basse pour des réponses factuelles basées sur le contexte
            # top_p=0.9,
        )
        if response.choices and len(response.choices) > 0:
            logging.info("Réponse reçue de l'API Mistral.")
            return response.choices[0].message.content
        else:
            logging.warning("L'API n'a pas retourné de choix valide.")
            return "Désolé, je n'ai pas pu générer de réponse valide pour le moment."
    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'API Mistral: {e}")
        logging.exception("Erreur API Mistral pendant client.chat")
        return "Je suis désolé, une erreur technique m'empêche de répondre. Veuillez réessayer plus tard."

# --- Interface Utilisateur Streamlit ---
st.title(APP_TITLE)
st.caption(f"Assistant virtuel pour {COMMUNE_NAME} | Modèle: {model}")

# Affichage des messages de l'historique (pour l'UI)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Zone de saisie utilisateur
if prompt := st.chat_input(f"Posez votre question sur {COMMUNE_NAME}..."):
    # 1. Ajouter et afficher le message de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # === Début de la logique RAG ===

    # 2. Vérifier si le Vector Store est disponible
    if vector_store_manager is None:
        st.error("Le service de recherche de connaissances n'est pas disponible. Impossible de traiter votre demande.")
        logging.error("VectorStoreManager non disponible pour la recherche.")
        # On arrête ici car on ne peut pas faire de RAG
        st.stop()

    # 3. Rechercher le contexte dans le Vector Store
    try:
        logging.info(f"Recherche de contexte pour la question: '{prompt}' avec k={SEARCH_K}")
        search_results = vector_store_manager.search(prompt, k=SEARCH_K)
        logging.info(f"{len(search_results)} chunks trouvés dans le Vector Store.")
    except Exception as e:
        st.error(f"Une erreur est survenue lors de la recherche d'informations pertinentes: {e}")
        logging.exception(f"Erreur pendant vector_store_manager.search pour la query: {prompt}")
        search_results = [] # On continue sans contexte si la recherche échoue

    # 4. Formater le contexte pour le prompt LLM
    context_str = "\n\n---\n\n".join([
        f"Source: {res['metadata'].get('source', 'Inconnue')} (Score: {res['score']:.1f}%)\nContenu: {res['text']}"
        for res in search_results
    ])

    if not search_results:
        context_str = "Aucune information pertinente trouvée dans la base de connaissances pour cette question."
        logging.warning(f"Aucun contexte trouvé pour la query: {prompt}")

    # 5. Construire le prompt final pour l'API Mistral en utilisant le System Prompt RAG
    final_prompt_for_llm = SYSTEM_PROMPT.format(context_str=context_str, question=prompt)

    # Créer la liste de messages pour l'API (juste le prompt système/utilisateur combiné)
    messages_for_api = [
        # On pourrait séparer system et user, mais Mistral gère bien un long message user structuré
        ChatMessage(role="user", content=final_prompt_for_llm)
    ]

    # === Fin de la logique RAG ===


    # 6. Afficher indicateur + Générer la réponse de l'assistant via LLM
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.text("...") # Indicateur simple

        # Génération de la réponse de l'assistant en utilisant le prompt augmenté
        response_content = generer_reponse(messages_for_api)

        # Affichage de la réponse complète
        message_placeholder.write(response_content)

    # 7. Ajouter la réponse de l'assistant à l'historique (pour affichage UI)
    st.session_state.messages.append({"role": "assistant", "content": response_content})

# Petit pied de page optionnel
st.markdown("---")
st.caption("Propulsé par Mistral AI & Faiss | Mairie de " + COMMUNE_NAME)