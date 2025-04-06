# pages/1_Feedback_Viewer.py
import streamlit as st
import pandas as pd
import numpy as np
import logging
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Ajouter le dossier parent au chemin de recherche des modules Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Maintenant, nous pouvons importer les modules du dossier parent
from utils.database import get_all_interactions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(
    page_title="Visionneur de Feedbacks",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Visionneur des Interactions et Feedbacks")
st.caption("Affiche les dernières interactions enregistrées dans la base de données.")

# Bouton pour rafraîchir les données
if st.button("🔄 Rafraîchir les données"):
    st.cache_data.clear() # Invalide le cache de get_all_interactions si utilisé

# Récupérer les données (utilisation de st.cache_data pour la mise en cache)
@st.cache_data(ttl=60) # Cache les données pendant 60 secondes
def load_data():
    logging.info("Chargement des interactions depuis la base de données pour le viewer...")
    interactions_list = get_all_interactions(limit=200) # Augmenter la limite si besoin
    if not interactions_list:
        # Retourne deux DataFrames vides si pas de données
        empty_df = pd.DataFrame()
        return empty_df, empty_df

    # Convertir la liste de dictionnaires en DataFrame Pandas pour un affichage facile
    df = pd.DataFrame(interactions_list)

    # Optionnel: Améliorer la présentation du DataFrame
    # Convertir le timestamp en type datetime si ce n'est pas déjà fait
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Trier par timestamp le plus récent en premier
    df = df.sort_values(by='timestamp', ascending=False)
    # Sélectionner et renommer les colonnes pour plus de clarté
    df_display = df[[
        'timestamp',
        'query',
        'response',
        'feedback',
        'feedback_comment',
        'id', # Garder l'ID pour référence
        'sources', # Garder les sources pour inspection si nécessaire
        'metadata' # Informations sur le mode utilisé
    ]].rename(columns={
        'timestamp': 'Date & Heure (UTC)',
        'query': 'Question Utilisateur',
        'response': 'Réponse Assistant',
        'feedback': 'Feedback',
        'feedback_comment': 'Commentaire',
        'id': 'ID Interaction',
        'metadata': 'Mode'
    })
    return df_display, df # Retourne aussi le df original si besoin d'accéder aux sources

# Charger et afficher les données
try:
    df_display, df_original = load_data()

    if df_display.empty:
        st.warning("Aucune interaction enregistrée dans la base de données pour le moment.")
    else:
        st.info(f"{len(df_display)} interactions trouvées.")

        # Créer un onglet pour les statistiques et un pour les données brutes
        tab1, tab2 = st.tabs(["Statistiques", "Données brutes"])

        with tab1:
            st.subheader("📊 Statistiques des feedbacks")

            # Utiliser les valeurs numériques de feedback si disponibles

            # Ajouter une colonne pour la valeur numérique du feedback si elle existe
            if 'feedback_value' in df_original.columns:
                feedback_values = df_original['feedback_value'].dropna()
            else:
                # Convertir les textes en valeurs numériques si la colonne n'existe pas
                feedback_values = df_original['feedback'].apply(
                    lambda x: 1 if x == "positif" else 0 if x == "négatif" else None
                ).dropna()

            # Compter les feedbacks positifs et négatifs
            if len(feedback_values) > 0:
                positive_count = sum(feedback_values == 1)
                negative_count = sum(feedback_values == 0)
                total_count = len(feedback_values)
                positive_percent = (positive_count / total_count * 100) if total_count > 0 else 0
                negative_percent = (negative_count / total_count * 100) if total_count > 0 else 0

                # Afficher les statistiques
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total des feedbacks", total_count)
                with col2:
                    st.metric("Feedbacks positifs", positive_count, f"{positive_percent:.1f}%")
                with col3:
                    st.metric("Feedbacks négatifs", negative_count, f"{negative_percent:.1f}%")

                # Créer un graphique en barres
                feedback_data = pd.DataFrame({
                    'Type': ['Positif', 'Négatif'],
                    'Nombre': [positive_count, negative_count]
                })

                fig = px.bar(
                    feedback_data,
                    x='Type',
                    y='Nombre',
                    color='Type',
                    color_discrete_map={'Positif': '#00CC96', 'Négatif': '#EF553B'},
                    title="Répartition des feedbacks"
                )

                # Ajouter les pourcentages sur les barres
                fig.update_traces(texttemplate='%{y} (%{y/sum:.1%})', textposition='outside')

                # Afficher le graphique
                st.plotly_chart(fig, use_container_width=True)

                # Ajouter un graphique d'évolution des feedbacks dans le temps si assez de données
                if len(df_original) >= 5:
                    st.subheader("📈 Évolution des feedbacks dans le temps")

                    # Convertir le timestamp en datetime si ce n'est pas déjà fait
                    df_original['timestamp'] = pd.to_datetime(df_original['timestamp'])

                    # Créer une colonne pour la date (sans l'heure)
                    df_original['date'] = df_original['timestamp'].dt.date

                    # Grouper par date et compter les feedbacks positifs et négatifs
                    if 'feedback_value' in df_original.columns:
                        # Utiliser la colonne feedback_value si disponible
                        feedback_by_date = df_original.groupby('date').apply(
                            lambda x: pd.Series({
                                'positif': sum(x['feedback_value'] == 1),
                                'négatif': sum(x['feedback_value'] == 0),
                                'total': len(x)
                            })
                        ).reset_index()
                    else:
                        # Sinon, utiliser la colonne feedback
                        feedback_by_date = df_original.groupby('date').apply(
                            lambda x: pd.Series({
                                'positif': sum(x['feedback'] == "positif"),
                                'négatif': sum(x['feedback'] == "négatif"),
                                'total': len(x)
                            })
                        ).reset_index()

                    # Créer un graphique d'évolution
                    fig2 = go.Figure()

                    # Ajouter les lignes pour les feedbacks positifs et négatifs
                    fig2.add_trace(go.Scatter(
                        x=feedback_by_date['date'],
                        y=feedback_by_date['positif'],
                        mode='lines+markers',
                        name='Positifs',
                        line=dict(color='#00CC96', width=2),
                        marker=dict(size=8)
                    ))

                    fig2.add_trace(go.Scatter(
                        x=feedback_by_date['date'],
                        y=feedback_by_date['négatif'],
                        mode='lines+markers',
                        name='Négatifs',
                        line=dict(color='#EF553B', width=2),
                        marker=dict(size=8)
                    ))

                    # Configurer le graphique
                    fig2.update_layout(
                        title="Évolution des feedbacks par jour",
                        xaxis_title="Date",
                        yaxis_title="Nombre de feedbacks",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )

                    # Afficher le graphique
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Aucun feedback n'a encore été donné.")

        with tab2:
            st.subheader("📃 Données brutes")
            st.dataframe(
            df_display,
            use_container_width=True,
            # Configuration des colonnes pour ajuster la largeur et le formatage
            column_config={
                "Date & Heure (UTC)": st.column_config.DatetimeColumn(
                    format="YYYY-MM-DD HH:mm:ss",
                    width="small"
                ),
                "Question Utilisateur": st.column_config.TextColumn(width="medium"),
                "Réponse Assistant": st.column_config.TextColumn(width="large"),
                "Feedback": st.column_config.TextColumn(width="small"),
                "Commentaire": st.column_config.TextColumn(width="medium"),
                "ID Interaction": st.column_config.NumberColumn(width="small"),
                "sources": st.column_config.ListColumn(width="medium"), # Affiche comme une liste
                "Mode": st.column_config.JsonColumn(width="medium") # Affiche les métadonnées comme JSON
            },
            hide_index=True # Cache l'index du DataFrame
        )

        # Optionnel: Permettre de voir les détails d'une interaction (y compris les sources)
        st.subheader("🔍 Examiner une interaction spécifique")
        selected_id = st.selectbox("Sélectionnez l'ID de l'interaction:", options=df_original['id'].tolist())

        if selected_id:
            selected_interaction = df_original[df_original['id'] == selected_id].iloc[0]
            st.write(f"**Question:** {selected_interaction['query']}")
            st.write(f"**Réponse:** {selected_interaction['response']}")
            st.write(f"**Feedback:** {selected_interaction['feedback']} {selected_interaction['feedback_comment'] or ''}")

            # Afficher les métadonnées (mode, confiance, etc.)
            metadata = selected_interaction['metadata']
            if metadata and isinstance(metadata, dict):
                mode = metadata.get('mode', 'N/A')
                confidence = metadata.get('confidence', 0.0)
                reason = metadata.get('reason', 'N/A')
                st.write(f"**Mode:** {mode} (confiance: {confidence:.2f})")
                st.write(f"**Raison:** {reason}")
            elif metadata:
                st.write("**Métadonnées:**")
                st.json(metadata)
            st.write("**Sources utilisées lors de la génération:**")
            sources = selected_interaction['sources']
            if sources and isinstance(sources, list):
                 for i, src in enumerate(sources):
                     meta = src.get("metadata", {})
                     with st.expander(f"Source {i+1}: `{meta.get('source', 'N/A')}` (Score: {src.get('score', 0.0):.4f})"):
                         st.text(src.get('text', 'N/A'))
            elif sources:
                 st.json(sources) # Affiche le JSON brut si ce n'est pas une liste
            else:
                 st.write("Aucune source enregistrée pour cette interaction.")


except Exception as e:
    logging.error(f"Erreur lors du chargement ou de l'affichage des données: {e}", exc_info=True)
    st.error(f"Une erreur est survenue lors de l'affichage des feedbacks: {e}")