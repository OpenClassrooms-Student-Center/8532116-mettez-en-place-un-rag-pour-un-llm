"""
Module de classification des requêtes pour déterminer si une question nécessite RAG
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from utils.config import MISTRAL_API_KEY, CHAT_MODEL, COMMUNE_NAME

class QueryClassifier:
    """
    Classe pour classifier les requêtes et déterminer si elles nécessitent RAG
    """
    
    def __init__(self):
        """
        Initialise le classificateur de requêtes
        """
        self.mistral_client = MistralClient(api_key=MISTRAL_API_KEY) if MISTRAL_API_KEY else None
        
        # Mots-clés liés à la commune qui suggèrent un besoin de RAG
        self.commune_keywords = [
            COMMUNE_NAME.lower(),
            "mairie", "commune", "ville", "municipal", "municipalité",
            "conseil", "maire", "adjoint", "élu", "service",
            "horaire", "ouverture", "fermeture", "adresse", "contact",
            "document", "formulaire", "démarche", "administrative",
            "urbanisme", "permis", "construction", "travaux",
            "école", "crèche", "garderie", "cantine", "scolaire",
            "association", "sport", "culture", "loisir", "bibliothèque",
            "événement", "manifestation", "fête", "marché",
            "transport", "bus", "circulation", "stationnement", "parking",
            "déchet", "poubelle", "recyclage", "environnement",
            "impôt", "taxe", "budget", "finance"
        ]
        
        # Questions générales qui ne nécessitent pas de RAG
        self.general_patterns = [
            r"^(bonjour|salut|hello|coucou|hey|bonsoir)[\s\.,!]*$",
            r"^(merci|thanks|thank you|je te remercie)[\s\.,!]*$",
            r"^(comment ça va|ça va|comment vas-tu|comment allez-vous)[\s\.,!?]*$",
            r"^(au revoir|bye|à bientôt|à plus tard|à la prochaine)[\s\.,!]*$",
            r"^(qui es[- ]tu|qu'es[- ]tu|que fais[- ]tu|comment fonctionnes[- ]tu|tu es quoi)[\s\?]*$",
            r"^(aide|help|sos|besoin d'aide)[\s\.,!?]*$"
        ]
    
    def needs_rag(self, query: str) -> Tuple[bool, float, str]:
        """
        Détermine si une requête nécessite RAG
        
        Args:
            query: Requête de l'utilisateur
            
        Returns:
            Tuple (besoin_rag, confiance, raison)
        """
        # Convertir la requête en minuscules pour la comparaison
        query_lower = query.lower()
        
        # 1. Vérifier les patterns de questions générales (salutations, remerciements, etc.)
        for pattern in self.general_patterns:
            if re.match(pattern, query_lower):
                return False, 0.95, "Question générale ou salutation"
        
        # 2. Vérifier la présence de mots-clés liés à la commune
        commune_keywords_found = [kw for kw in self.commune_keywords if kw in query_lower]
        if commune_keywords_found:
            keywords_str = ", ".join(commune_keywords_found)
            return True, 0.9, f"Contient des mots-clés liés à la commune: {keywords_str}"
        
        # 3. Utiliser le LLM pour les cas ambigus
        if self.mistral_client:
            return self._classify_with_llm(query)
        
        # Par défaut, utiliser RAG pour les questions longues (plus de 5 mots)
        words = query.split()
        if len(words) > 5:
            return True, 0.6, "Question complexe (plus de 5 mots)"
        
        # Par défaut, ne pas utiliser RAG
        return False, 0.5, "Aucun critère spécifique détecté"
    
    def _classify_with_llm(self, query: str) -> Tuple[bool, float, str]:
        """
        Utilise le LLM pour classifier la requête
        
        Args:
            query: Requête de l'utilisateur
            
        Returns:
            Tuple (besoin_rag, confiance, raison)
        """
        try:
            system_prompt = f"""Vous êtes un classificateur de requêtes pour un assistant virtuel de la commune de {COMMUNE_NAME}.
Votre tâche est de déterminer si une question nécessite une recherche dans une base de connaissances spécifique à la commune.

Répondez UNIQUEMENT par "RAG" ou "DIRECT" suivi d'une brève explication:
- "RAG" si la question porte sur des informations spécifiques à {COMMUNE_NAME} (services municipaux, événements, adresses, horaires, etc.)
- "DIRECT" si c'est une question générale, une salutation, ou une question qui ne nécessite pas d'informations spécifiques à la commune.

Exemples:
Question: "Bonjour, comment ça va?"
Réponse: DIRECT - Simple salutation

Question: "Quels sont les horaires de la mairie?"
Réponse: RAG - Demande d'informations spécifiques à la commune

Question: "Qui est le maire actuel?"
Réponse: RAG - Demande d'informations spécifiques à la commune

Question: "Qu'est-ce que l'intelligence artificielle?"
Réponse: DIRECT - Question générale de connaissance
"""
            
            messages = [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=query)
            ]
            
            response = self.mistral_client.chat(
                model=CHAT_MODEL,
                messages=messages,
                temperature=0.1,  # Température basse pour des réponses cohérentes
                max_tokens=50  # Réponse courte suffisante
            )
            
            result = response.choices[0].message.content.strip()
            logging.info(f"Classification LLM pour '{query}': {result}")
            
            # Analyser la réponse
            if result.startswith("RAG"):
                confidence = 0.85  # Confiance élevée dans la décision du LLM
                reason = result.replace("RAG - ", "").replace("RAG-", "").replace("RAG:", "").strip()
                return True, confidence, reason
            elif result.startswith("DIRECT"):
                confidence = 0.85
                reason = result.replace("DIRECT - ", "").replace("DIRECT-", "").replace("DIRECT:", "").strip()
                return False, confidence, reason
            else:
                # Réponse ambiguë, utiliser RAG par défaut
                return True, 0.6, "Classification ambiguë, utilisation de RAG par précaution"
                
        except Exception as e:
            logging.error(f"Erreur lors de la classification avec LLM: {e}")
            # En cas d'erreur, utiliser RAG par défaut
            return True, 0.5, f"Erreur de classification: {str(e)}"
