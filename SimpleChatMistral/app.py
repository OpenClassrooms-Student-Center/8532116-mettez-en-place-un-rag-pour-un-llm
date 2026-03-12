import streamlit as st
import os
from mistralai.client import Mistral
from SimpleChatMistral.chat_message import ChatMessage
import logging # Ajout pour un meilleur débogage des erreurs API
from dotenv import load_dotenv
load_dotenv()

# Configuration du logging
logging.basicConfig(level=logging.INFO)

# --- 1. Importation des bibliothèques et configuration ---
st.set_page_config(page_title="Assistant Mairie", page_icon="🏛️")

# Récupération de la clé API Mistral depuis les variables d'environnement
# !! ATTENTION : Remplacez "VOTRE_CLE_API_MISTRAL_ICI" par votre clé si vous ne configurez pas de variable d'environnement !!
# Il est FORTEMENT recommandé d'utiliser une variable d'environnement.
api_key = os.environ.get("MISTRAL_API_KEY") 

# Vérification de la présence de la clé API
if not api_key:
    st.error("Clé API Mistral non trouvée. Veuillez définir la variable d'environnement MISTRAL_API_KEY.")
    # Vous pouvez aussi proposer une saisie directe (moins sécurisé)
    # api_key = st.text_input("Entrez votre clé API Mistral:", type="password")
    # if not api_key:
    st.stop() # Arrête l'exécution si la clé n'est pas fournie

try:
    client = Mistral(api_key=api_key)
    model = "mistral-large-latest" # Ou un autre modèle comme "mistral-small-latest"
except Exception as e:
    st.error(f"Erreur lors de l'initialisation du client Mistral : {e}")
    st.stop()

# --- 2. Initialisation de l'historique des conversations ---
if "messages" not in st.session_state:
    # Ajout d'un message système initial (optionnel mais peut guider le modèle)
    # st.session_state.messages = [
    #     ChatMessage(role="system", content="Tu es un assistant virtuel pour la mairie. Réponds aux questions des citoyens de manière claire et concise.")
    # ]
    # Initialisation avec le message d'accueil de l'assistant
    st.session_state.messages = [{"role": "assistant", "content": "Bonjour, je suis l'assistant virtuel de la mairie. Comment puis-je vous aider aujourd'hui?"}]

# --- 3. Construction du prompt avec l'historique ---
def construire_prompt_session(messages, max_messages=10):
    """
    Construit le prompt pour l'API Mistral en utilisant les messages récents.

    Args:
        messages (list): Liste complète des messages de la session.
        max_messages (int): Nombre maximum de messages récents à inclure.

    Returns:
        list[ChatMessage]: Liste de messages formatés pour l'API.
    """
    # Garde seulement les N derniers messages pour limiter la taille du prompt
    recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages

    # Convertit les dictionnaires en objets ChatMessage
    formatted_messages = [
        ChatMessage(role=msg["role"], content=msg["content"]).format()
        for msg in recent_messages
    ]

    # Optionnel : Ajouter un message système au début si ce n'est pas déjà fait
    # if not any(m.role == "system" for m in formatted_messages):
    #     formatted_messages.insert(0, ChatMessage(role="system", content="Tu es un assistant virtuel pour la mairie. Réponds aux questions des citoyens de manière claire et concise."))

    logging.info(f"Messages envoyés à l'API : {formatted_messages}") # Pour débogage
    return formatted_messages

# --- 4. Génération de réponses via l'API Mistral ---
def generer_reponse(prompt_messages):
    """
    Appelle l'API Mistral pour générer une réponse.

    Args:
        prompt_messages (list[ChatMessage]): Messages formatés à envoyer à l'API.

    Returns:
        str: Le contenu de la réponse générée ou un message d'erreur.
    """
    try:
        response = client.chat(
            model=model,
            messages=prompt_messages,
            # safe_prompt=True # Décommentez si vous voulez activer le mode sécurisé
        )
        # Vérification si la réponse contient des choix
        if response.choices:
            return response.choices[0].message.content
        else:
            logging.error("L'API Mistral n'a retourné aucun choix.")
            return "Je suis désolé, je n'ai pas pu générer de réponse. Aucune option retournée."
    except Exception as e:
        logging.error(f"Erreur lors de l'appel à l'API Mistral: {e}")
        # Fournir plus de détails si possible, par exemple sur les erreurs de quota
        st.error(f"Erreur lors de la génération de la réponse: {e}")
        return "Je suis désolé, j'ai rencontré un problème technique. Veuillez réessayer plus tard."

# --- 5. Interface utilisateur Streamlit ---
st.title("🏛️ Assistant Virtuel de la Mairie")
st.caption(f"Utilisation du modèle : {model}")

# Affichage des messages précédents de l'historique
# On itère sur une copie pour éviter les problèmes si la liste est modifiée pendant l'itération
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --- 6. Traitement des entrées utilisateur et génération de réponses ---
if prompt := st.chat_input("Posez votre question ici..."):
    # Ajout du message de l'utilisateur à l'historique interne
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Affichage immédiat du message de l'utilisateur dans l'interface
    with st.chat_message("user"):
        st.write(prompt)

    # Préparation du prompt avec l'historique récent pour l'API
    prompt_messages_for_api = construire_prompt_session(st.session_state.messages)

    # Affichage d'un indicateur de chargement pendant la génération
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.text("...") # Indicateur visuel simple

        # Génération de la réponse via l'API
        response_content = generer_reponse(prompt_messages_for_api)

        # Affichage de la réponse complète
        message_placeholder.write(response_content)

    # Ajout de la réponse de l'assistant à l'historique interne
    st.session_state.messages.append({"role": "assistant", "content": response_content})

# Optionnel : Ajouter un bouton pour effacer l'historique
if st.button("Effacer la conversation"):
    st.session_state.messages = [{"role": "assistant", "content": "Bonjour, je suis l'assistant virtuel de la mairie. Comment puis-je vous aider aujourd'hui?"}]
    st.rerun() # Recharge la page pour afficher l'état initial