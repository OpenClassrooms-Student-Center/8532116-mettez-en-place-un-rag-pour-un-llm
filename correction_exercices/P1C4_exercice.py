import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# Textes du corpus municipal
corpus = [
    "Le projet de rénovation de la mairie avance.",
    "Une nouvelle école sera inaugurée le mois prochain.",
    "Les travaux de la route principale sont terminés.",
    "La bibliothèque municipale organise une exposition.",
    "Un plan de développement durable a été adopté."
]


# 1. Obtention des embeddings avec Sentence-BERT
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
sbert_embeddings = sbert_model.encode(corpus)


# Réduction de dimension pour la visualisation avec t-SNE
sbert_tsne = TSNE(n_components=2, random_state=0)
sbert_embeddings_2d = sbert_tsne.fit_transform(sbert_embeddings)


# Visualisation des embeddings Sentence-BERT
plt.figure(figsize=(8, 6))
plt.scatter(sbert_embeddings_2d[:, 0], sbert_embeddings_2d[:, 1], color='blue')
for i, txt in enumerate(corpus):
    plt.annotate(txt, (sbert_embeddings_2d[i, 0], sbert_embeddings_2d[i, 1]), fontsize=9)
plt.title("Visualisation des embeddings avec Sentence-BERT")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()


### Visualisation avec Mistral embeddings
Pour inclure Mistral, voici comment générer les embeddings et les visualiser :


from mistral.embedding import EmbeddingClient


# Initialisation du client Mistral
client = EmbeddingClient(api_key='YOUR_API_KEY')


# Obtention des embeddings avec Mistral
mistral_embeddings = [client.get_embedding(text) for text in corpus]


# Réduction de dimension pour les embeddings Mistral
mistral_tsne = TSNE(n_components=2, random_state=0).fit_transform(mistral_embeddings)


# Visualisation des embeddings Mistral
plt.figure(figsize=(8, 6))
plt.scatter(mistral_tsne[:, 0], mistral_tsne[:, 1], color='green')
for i, txt in enumerate(corpus):
    plt.annotate(txt, (mistral_tsne[i, 0], mistral_tsne[i, 1]), fontsize=9)
plt.title("Visualisation des embeddings avec Mistral")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()


### Comparaison avec fastText
Nous pouvons comparer les embeddings générés par SBERT et Mistral avec ceux produits par fastText.


import fasttext.util
import numpy as np
import seaborn as sns


# Téléchargement et chargement du modèle fastText
fasttext.util.download_model('fr', if_exists='ignore')  # Modèle français
ft = fasttext.load_model('cc.fr.300.bin')


# Obtention des embeddings fastText
fasttext_embeddings = [ft.get_sentence_vector(text) for text in corpus]


# Réduction de dimension pour les embeddings fastText
fasttext_tsne = TSNE(n_components=2, random_state=0).fit_transform(np.array(fasttext_embeddings))


# Visualisation des embeddings fastText
plt.figure(figsize=(8, 6))
ax = sns.scatterplot(x=fasttext_tsne[:, 0], y=fasttext_tsne[:, 1], hue=corpus, palette="deep")
sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1))
plt.title("Visualisation des embeddings avec fastText")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()
