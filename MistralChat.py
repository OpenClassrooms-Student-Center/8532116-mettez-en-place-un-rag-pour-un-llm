# app.py
import streamlit as st
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import logging
import datetime
from streamlit_feedback import streamlit_feedback # Importez le composant

# Importer nos modules locaux
from utils.config import APP_TITLE, COMMUNE_NAME, MISTRAL_API_KEY
from utils.vector_store import VectorStoreManager
from utils.database import log_interaction, update_feedback # Importez update_feedback
from utils.query_classifier import QueryClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📚",
    layout="wide"
)

# --- Initialisation (avec mise en cache Streamlit) ---

# Met en cache le VectorStoreManager pour éviter de recharger l'index à chaque interaction
@st.cache_resource
def get_vector_store():
    logging.info("Chargement du VectorStoreManager...")
    return VectorStoreManager()

# Met en cache le client Mistral
@st.cache_resource
def get_mistral_client():
    if not MISTRAL_API_KEY:
        st.error("Erreur: La clé API Mistral (MISTRAL_API_KEY) n'est pas configurée.")
        st.stop()
    logging.info("Initialisation du client Mistral...")
    return MistralClient(api_key=MISTRAL_API_KEY)

# Met en cache le classificateur de requêtes
@st.cache_resource
def get_query_classifier():
    logging.info("Initialisation du classificateur de requêtes...")
    return QueryClassifier()

# Charge le Vector Store, le client Mistral et le classificateur de requêtes
vector_store = get_vector_store()
client = get_mistral_client()
query_classifier = get_query_classifier()

# Initialise l'historique du chat dans l'état de la session s'il n'existe pas
if "messages" not in st.session_state:
    st.session_state.messages = []
# Initialise l'ID de la dernière interaction pour le feedback
if "last_interaction_id" not in st.session_state:
    st.session_state.last_interaction_id = None

# --- Interface Utilisateur ---

# Barre latérale (sidebar)
with st.sidebar:
    st.title(f"📚 {COMMUNE_NAME}")
    st.caption(f"Assistant virtuel municipal")

    # Bouton pour lancer une nouvelle conversation
    if st.button("🔄 Nouvelle conversation", use_container_width=True):
        # Réinitialiser l'historique des messages
        st.session_state.messages = []
        st.session_state.last_interaction_id = None
        st.rerun()  # Recharger l'application pour afficher la nouvelle conversation

    st.divider()

    # Paramètres de l'application
    st.subheader("⚙️ Paramètres")

    # Sélecteur de modèle Mistral
    model_options = {
        "mistral-small-latest": "Mistral Small (rapide)",
        "mistral-large-latest": "Mistral Large (précis)"
    }
    selected_model = st.selectbox(
        "Modèle LLM",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=0  # Small par défaut
    )

    # Slider pour le nombre de documents
    num_docs = st.slider(
        "Nombre de documents à récupérer",
        min_value=1,
        max_value=20,
        value=5,  # 5 par défaut
        step=1
    )

    # Slider pour le score minimum (en pourcentage)
    min_score_percent = st.slider(
        "Score minimum (filtrer les résultats faibles)",
        min_value=0,
        max_value=100,
        value=75,  # 75% par défaut
        step=5,
        format="%d%%"
    )
    # Convertir le pourcentage en valeur décimale (0-1)
    min_score = min_score_percent / 100.0

    st.divider()

    # Informations sur l'application
    st.subheader("📝 Informations")
    st.markdown(f"**Modèle sélectionné**: {model_options[selected_model]}")
    st.markdown(f"**Documents indexés**: {vector_store.index.ntotal if vector_store.index else 0}")

    # Informations sur la conversation actuelle
    if st.session_state.messages:
        st.info(f"{len(st.session_state.messages) // 2} échanges dans cette conversation")

        # Bouton pour télécharger la conversation
        # Préparer le contenu de la conversation au format texte
        conversation_text = "\n\n".join([
            f"{'Utilisateur' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in st.session_state.messages
        ])

        # Ajouter un en-tête avec la date et le titre
        header = f"Conversation avec l'assistant virtuel de {COMMUNE_NAME}\n"
        header += f"Date: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n"
        conversation_text = header + conversation_text

        # Bouton de téléchargement
        st.download_button(
            label="💾 Télécharger la conversation",
            data=conversation_text,
            file_name=f"conversation_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )

# Titre principal
st.title(f"📚 {APP_TITLE}")
st.caption(f"Posez vos questions sur {COMMUNE_NAME}")

# Affichage de l'historique du chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Afficher les sources si elles existent pour les messages de l'assistant
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("Sources utilisées"):
                for i, source in enumerate(message["sources"]):
                    # Accès sécurisé aux métadonnées
                    meta = source.get("metadata", {})
                    st.markdown(f"**Source {i+1}:** `{meta.get('source', 'N/A')}`")
                    st.markdown(f"*Score de similarité:* {source.get('score', 0.0):.2f}%")
                    if 'raw_score' in source:
                        st.markdown(f"*Score brut:* {source.get('raw_score', 0.0):.4f}")
                    st.markdown(f"*Catégorie:* `{meta.get('category', 'N/A')}`")
                    st.text_area(f"Extrait {i+1}", value=source.get("text", "")[:500]+"...", height=100, disabled=True, key=f"src_{message['timestamp']}_{i}") # Clé unique pour éviter les conflits


# Zone de saisie utilisateur en bas
if prompt := st.chat_input("Posez votre question ici..."):
    # Ajouter le message utilisateur à l'historique et l'afficher
    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": datetime.datetime.now().isoformat()})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Afficher un message d'attente
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🧠 Recherche d'informations et génération de la réponse...")

        # --- Logique de traitement de la requête ---
        try:
            # 1. Classifier la requête pour déterminer si elle nécessite RAG
            needs_rag, confidence, reason = query_classifier.needs_rag(prompt)

            # Afficher le résultat de la classification
            mode_str = "RAG" if needs_rag else "DIRECT"
            logging.info(f"Classification de la requête: {mode_str} (confiance: {confidence:.2f}) - Raison: {reason}")

            # Afficher un message indiquant le mode utilisé
            mode_info = st.empty()
            if needs_rag:
                mode_info.info(f"Mode RAG: Recherche d'informations spécifiques dans la base de connaissances (confiance: {confidence:.2f})")
                # 2. Recherche dans le Vector Store si nécessaire
                logging.info(f"Recherche de documents pour: '{prompt}' (max: {num_docs}, score min: {min_score})")
                retrieved_docs = vector_store.search(prompt, k=num_docs, min_score=min_score)
            else:
                mode_info.info(f"Mode Direct: Réponse basée sur les connaissances générales du modèle (confiance: {confidence:.2f})")
                # Pas de recherche dans le Vector Store
                retrieved_docs = []

            # 2. Préparer les données en fonction du mode
            if needs_rag and retrieved_docs:
                # Mode RAG avec documents trouvés
                logging.info(f"{len(retrieved_docs)} documents récupérés.")
                # Préparer le contexte pour le LLM
                context_str = "\n\n---\n\n".join([
                    f"Source: {doc['metadata'].get('source', 'Inconnue')} (Score: {doc['score']:.4f})\nContenu: {doc['text']}"
                    for doc in retrieved_docs
                ])
                sources_for_log = [ # Version simplifiée pour le log et l'affichage
                    {"text": doc["text"], "metadata": doc["metadata"], "score": doc["score"]}
                    for doc in retrieved_docs
                ]

                # Prompt système pour le mode RAG
                system_prompt = f"""Vous êtes un assistant virtuel pour {COMMUNE_NAME}.
Répondez à la question de l'utilisateur en vous basant UNIQUEMENT sur le contexte fourni ci-dessous.
Si l'information n'est pas dans le contexte, dites que vous ne savez pas ou que l'information n'est pas disponible dans les documents fournis.
Soyez concis et précis. Citez vos sources si possible (par exemple, en mentionnant le nom du fichier ou la catégorie trouvée dans les métadonnées).

Contexte fourni:
---
{context_str}
---
"""
            elif needs_rag and not retrieved_docs:
                # Mode RAG mais aucun document trouvé
                logging.warning("Aucun document pertinent trouvé.")
                context_str = "Aucune information pertinente trouvée dans les documents."
                sources_for_log = []

                # Prompt système pour le mode RAG sans résultats
                system_prompt = f"""Vous êtes un assistant virtuel pour {COMMUNE_NAME}.
L'utilisateur a posé une question qui semble concerner des informations spécifiques à la commune, mais aucune information pertinente n'a été trouvée dans notre base de connaissances.
Indiquez poliment que vous n'avez pas cette information spécifique et suggérez à l'utilisateur de reformuler sa question ou de contacter directement la mairie.
N'inventez pas d'informations sur {COMMUNE_NAME}.
"""
            else:
                # Mode Direct (sans RAG)
                context_str = "Mode direct: réponse basée sur les connaissances générales du modèle."
                sources_for_log = []

                # Prompt système pour le mode Direct
                system_prompt = f"""Vous êtes un assistant virtuel pour {COMMUNE_NAME}.
Répondez à la question de l'utilisateur en utilisant vos connaissances générales.
Soyez concis, précis et utile.
Si la question concerne des informations spécifiques à {COMMUNE_NAME} que vous ne connaissez pas, indiquez clairement que vous n'avez pas cette information spécifique.
N'inventez pas d'informations sur {COMMUNE_NAME}.
"""
            user_message = ChatMessage(role="user", content=prompt)
            system_message = ChatMessage(role="system", content=system_prompt)
            messages_for_api = [system_message, user_message]

            # 3. Appel à l'API Mistral Chat
            logging.info(f"Appel de l'API Mistral Chat avec le modèle {selected_model}...")
            chat_response = client.chat(
                model=selected_model,
                messages=messages_for_api
            )
            response_text = chat_response.choices[0].message.content
            logging.info("Réponse générée par Mistral.")

            # 4. Afficher la réponse et les sources
            message_placeholder.markdown(response_text)

            # Afficher les sources si disponibles (mode RAG avec résultats)
            if sources_for_log:
                with st.expander("Sources utilisées"):
                    for i, source in enumerate(sources_for_log):
                        meta = source.get("metadata", {})
                        st.markdown(f"**Source {i+1}:** `{meta.get('source', 'N/A')}`")
                        st.markdown(f"*Score de similarité:* {source.get('score', 0.0):.2f}%")
                        if 'raw_score' in source:
                            st.markdown(f"*Score brut:* {source.get('raw_score', 0.0):.4f}")
                        st.markdown(f"*Catégorie:* `{meta.get('category', 'N/A')}`")
                        st.text_area(f"Extrait {i+1}", value=source.get("text", "")[:500]+"...", height=100, disabled=True, key=f"src_new_{i}") # Clé unique
            elif needs_rag:
                # Mode RAG sans résultats
                st.info("Aucune source pertinente n'a été trouvée dans la base de connaissances pour cette question.")
            else:
                # Mode Direct
                st.info("Réponse générée en mode direct, sans consultation de la base de connaissances.")

            # 5. Enregistrer l'interaction dans la base de données (sans feedback initial)
            # Ajouter des métadonnées sur le mode utilisé
            metadata = {
                "mode": "RAG" if needs_rag else "DIRECT",
                "confidence": confidence,
                "reason": reason
            }

            interaction_id = log_interaction(
                query=prompt,
                response=response_text,
                sources=sources_for_log, # Stocke la liste de dicts
                metadata=metadata # Ajouter les métadonnées sur le mode
            )
            st.session_state.last_interaction_id = interaction_id # Garde l'ID pour le feedback
            logging.info(f"Interaction enregistrée avec ID: {interaction_id}")


            # Ajouter la réponse de l'assistant à l'historique pour affichage permanent
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "sources": sources_for_log, # Garder les sources pour réaffichage
                "timestamp": datetime.datetime.now().isoformat(),
                 "interaction_id": interaction_id # Lier le message à l'ID BDD
            })


        except Exception as e:
            # Vérifier si c'est une erreur API Mistral
            if hasattr(e, 'status_code') and hasattr(e, 'message'):
                logging.error(f"Erreur API Mistral: {e}")
                message_placeholder.error(f"Une erreur s'est produite lors de la communication avec l'API Mistral: {e}")
            else:
                logging.error(f"Erreur inattendue: {e}", exc_info=True)
                message_placeholder.error(f"Une erreur s'est produite: {e}")

            st.session_state.messages.append({"role": "assistant", "content": f"Erreur: {e}", "sources": [], "timestamp": datetime.datetime.now().isoformat(), "interaction_id": None})
            st.session_state.last_interaction_id = None # Pas d'ID si erreur avant log

# --- Section Feedback ---
# Placer le feedback après la boucle d'affichage et la zone de chat input
# On cible la *dernière* réponse de l'assistant pour le feedback
last_assistant_message = next((m for m in reversed(st.session_state.messages) if m["role"] == "assistant"), None)

# Vérifie si la dernière réponse a un ID d'interaction associé
current_interaction_id = last_assistant_message.get("interaction_id") if last_assistant_message else None

if current_interaction_id:
    # Utilisation de streamlit-feedback
    feedback = streamlit_feedback(
        feedback_type="thumbs", # "thumbs" ou "faces"
        optional_text_label="[Optionnel] Commentaires :",
        key=f"feedback_{current_interaction_id}", # Clé unique liée à l'interaction
        align="flex-start",  # Aligner à gauche
        on_submit=lambda x: logging.info(f"Feedback soumis: {x}")  # Log pour débogage
    )

    # Traitement du feedback s'il est donné
    if feedback:
        # Convertir le feedback en valeur numérique et texte
        feedback_score = feedback.get('score')

        # Vérifier si le score est valide
        # Le composant streamlit_feedback peut renvoyer des emojis au lieu de "thumbs_up"/"thumbs_down"
        if feedback_score == "👍" or feedback_score == "thumbs_up":
            feedback_score = "positive"
        elif feedback_score == "👎" or feedback_score == "thumbs_down":
            feedback_score = "negative"
        else:
            logging.warning(f"Score de feedback invalide: {feedback_score}")
            feedback_score = None

        # 1 pour positif, 0 pour négatif
        feedback_value = 1 if feedback_score == "positive" else 0 if feedback_score == "negative" else None

        # Texte pour la base de données ("positif" ou "négatif")
        feedback_text = "positif" if feedback_score == "positive" else "négatif" if feedback_score == "negative" else "N/A"

        # Emoji pour l'affichage dans l'interface
        feedback_emoji = "👍" if feedback_score == "positive" else "👎" if feedback_score == "negative" else "N/A"
        comment = feedback.get('text', None)

        # Mettre à jour l'interaction dans la base de données
        success = update_feedback(current_interaction_id, feedback_text, comment, feedback_value)
        if success:
            st.toast(f"Merci pour votre retour ({feedback_emoji}) !", icon="✅")
            # Optionnel: Désactiver les boutons après le premier clic pour éviter les soumissions multiples
            # Ceci est plus complexe à gérer avec la nature stateless de Streamlit sans callbacks avancés.
            # Pour la simplicité, on se contente de l'enregistrer. L'utilisateur peut re-cliquer mais seule la dernière valeur compte.
        else:
            st.toast("Erreur lors de l'enregistrement de votre retour.", icon="❌")

        # Optionnel : Effacer le feedback de l'état pour éviter re-soumission au re-run
        # st.session_state[f"feedback_{current_interaction_id}"] = None # Peut causer des pbs si mal géré

else:
    st.write("Posez une question pour pouvoir donner votre avis sur la réponse.")