# utils/query_classifier.py
import logging
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage


# Il peut être utile de définir les intentions possibles comme constantes
INTENT_RAG = "RAG"
INTENT_CHAT = "CHAT"
DEFAULT_INTENT = INTENT_RAG # Choisir RAG par défaut pour privilégier la recherche


def classify_query_intent(query: str, client: MistralClient, model: str = "mistral-large-latest") -> str:
    """
    Classifie l'intention de la requête utilisateur en utilisant l'API Mistral.


    Args:
        query: La question posée par l'utilisateur.
        client: Le client Mistral initialisé.
        model: Le modèle Mistral à utiliser pour la classification.


    Returns:
        L'intention détectée ("RAG" ou "CHAT").
    """
    classification_system_prompt = f"""
    Votre rôle est de classifier l'intention de la question de l'utilisateur pour un chatbot de mairie.
    Répondez uniquement par "RAG" ou "CHAT". Ne fournissez aucune autre explication.


    - Répondez "RAG" si la question cherche des informations spécifiques qui pourraient se trouver dans les documents de la mairie (procédures administratives, horaires, documents nécessaires, règlements, services municipaux, informations locales spécifiques).
    - Répondez "CHAT" si la question est une salutation, une formule de politesse, une conversation générale, une question hors sujet pour la mairie, ou une simple interaction sociale.


    Exemples:
    - "Quels papiers faut-il pour un passeport ?" -> RAG
    - "Bonjour comment allez-vous ?" -> CHAT
    - "Quels sont les horaires de la piscine municipale ?" -> RAG
    - "Merci !" -> CHAT
    - "Parlez-moi de la météo demain" -> CHAT
    - "Comment inscrire mon enfant à l'école ?" -> RAG


    Question à classifier :
    """


    messages = [
        ChatMessage(role="system", content=classification_system_prompt),
        ChatMessage(role="user", content=query)
    ]


    try:
        logging.info(f"Classification de la requête: '{query[:50]}...'")
        response = client.chat(
            model=model,
            messages=messages,
            temperature=0.1, # Basse température pour une réponse plus déterministe
            max_tokens=5     # Très court, on attend juste RAG ou CHAT
        )
        intent = response.choices[0].message.content.strip().upper()


        if intent == INTENT_RAG:
            logging.info(f"Intention détectée: {INTENT_RAG}")
            return INTENT_RAG
        elif intent == INTENT_CHAT:
            logging.info(f"Intention détectée: {INTENT_CHAT}")
            return INTENT_CHAT
        else:
            logging.warning(f"Classification non claire reçue: '{intent}'. Utilisation de l'intention par défaut: {DEFAULT_INTENT}")
            return DEFAULT_INTENT # Retourne l'intention par défaut si la réponse n'est pas claire


    except Exception as e:
        logging.error(f"Erreur lors de la classification de la requête: {e}")
        return DEFAULT_INTENT # Retourne l'intention par défaut en cas d'erreur
