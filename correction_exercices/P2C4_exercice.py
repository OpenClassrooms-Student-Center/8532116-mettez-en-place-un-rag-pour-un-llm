# utils/vector_store.py
import numpy as np
import faiss
import logging
import pickle
from mistralai.client import MistralClient
# Assurez-vous que la fonction embed_texts_mistral est définie ou importée ici
 
# ... (autres fonctions comme load_faiss_index, load_chunks, embed_texts_mistral) ...
 
def search_similar_documents(
	query: str,
	index: faiss.Index,
	chunks: list[str],
	client: MistralClient,
	k: int = 3,
	model_embed: str = "mistral-embed",
	max_distance_threshold: float | None = None # Nouveau paramètre
	) -> list[str]:
	"""
	Recherche les k chunks les plus pertinents pour une requête donnée,
	avec un filtrage optionnel par seuil de distance max.
 
	Args:
    	query: La question de l'utilisateur.
    	index: L'index Faiss chargé.
    	chunks: La liste complète des chunks de texte.
    	client: Le client Mistral pour l'embedding.
    	k: Nombre maximum de documents à retourner avant filtrage par seuil.
    	model_embed: Modèle d'embedding Mistral à utiliser.
    	max_distance_threshold: Seuil de distance L2 max. Si None, pas de filtrage.
 
	Returns:
    	Liste des chunks pertinents filtrés.
	"""
	if index is None:
    	logging.warning("Index non disponible pour la recherche.")
    	return []
	try:
    	# Obtenir l'embedding de la requête
    	query_embedding = embed_texts_mistral([query], client, model_embed)[0]
    	query_embedding_np = np.array([query_embedding]).astype('float32')
 
    	# Recherche initiale des k plus proches voisins
    	distances, indices = index.search(query_embedding_np, k)
 
    	# Extraire les résultats de la première (et unique) requête
    	result_distances = distances[0]
    	result_indices = indices[0]
 
    	# Filtrage par seuil de distance
    	relevant_chunks = []
    	filtered_indices_distances = [] # Pour le logging/debug
 
    	for i in range(len(result_indices)):
        	idx = result_indices[i]
        	dist = result_distances[i]
 
        	# Ignorer les résultats invalides de Faiss (parfois -1)
        	if idx == -1:
            	continue
 
        	# Appliquer le filtre de seuil si défini
        	if max_distance_threshold is None or dist <= max_distance_threshold:
            	relevant_chunks.append(chunks[idx])
                filtered_indices_distances.append((idx, dist))
        	else:
            	# Si on atteint un document qui ne passe pas le seuil,
            	# on peut potentiellement arrêter car les suivants sont plus loin.
            	# Mais Faiss ne garantit pas toujours un ordre parfait pour certains index,
            	# donc il est plus sûr de vérifier tous les k résultats.
            	logging.debug(f"Document index {idx} écarté (distance {dist:.4f} > seuil {max_distance_threshold})")
 
 
    	logging.info(f"Recherche RAG: {len(relevant_chunks)}/{k} chunks trouvés après filtrage (seuil={max_distance_threshold}). Indices/Distances retenus: {filtered_indices_distances}")
    	return relevant_chunks
 
	except Exception as e:
    	logging.error(f"Erreur pendant la recherche de similarité: {e}")
    	return []
2. Appel depuis MistralChat.py ou le script d'évaluation
Assurez-vous de lire les valeurs de k et du seuil depuis votre configuration et de les passer à la fonction.
Python
# Extrait de MistralChat.py ou generer_donnees_evaluation
 
# Charger les paramètres depuis la configuration
num_relevant_docs = config.get('rag_num_docs', 3) # k
distance_threshold = config.get('rag_distance_threshold', None) # Seuil (peut être None)
 
# ... (après avoir obtenu le 'prompt' ou la 'question') ...
 
# --- Étape 2: Récupération (si RAG) ---
if intent == INTENT_RAG:
	logging.info(f"Intention RAG: Recherche de documents (k={num_relevant_docs}, seuil={distance_threshold}).")
	model_embed = config.get('mistral_model_embed', 'mistral-embed')
 
	contexts = search_similar_documents(
    	prompt, # ou 'q' dans la fonction d'évaluation
    	index,
    	chunks,
    	client,
    	k=num_relevant_docs, # Passer k
    	model_embed=model_embed,
        max_distance_threshold=distance_threshold # Passer le seuil
	)
else:
	contexts = []
 # ... (la suite du code reste identique : construction du prompt, génération) ...
