import sys
import pandas as pd
import streamlit as st
from anime_recommender.utils.ml_utils.models.content_filtering_models import ContentBasedRecommender
from anime_recommender.utils.ml_utils.models.collaborative_filtering_models import CollaborativeAnimeRecommender 
from anime_recommender.utils.main_utils.utils import load_object
from anime_recommender.loggers.logging import logging
# File paths
anime_file_path = "datasets/Animes.csv"
userratings_file_path = "datasets/UserRatings.csv"
merged_file_path = "datasets/Anime_UserRatings.csv"
userbasedknn = "models/userbasedknn.pkl"
itembasedknn = "models/itembasedknn.pkl"
svd = "models/svd.pkl"
cosine_tfv = "models/cosine_similarity.pkl"

    anime_file_path = data_ingestion_artifact.feature_store_anime_file_path
    userratings_file_path = data_ingestion_artifact.feature_store_userrating_file_path

    merged_file_path = "datasets/Anime_UserRatings.csv"  # This can also be obtained from Data Transformation artifact 

    anime_data = pd.read_csv(anime_file_path)
    userratings_data = pd.read_csv(userratings_file_path)

st.set_page_config(page_title="Anime Recommendation System", layout="wide")

# Streamlit UI
app_selector = st.sidebar.radio(
    "Select App", ("Content-Based Recommender", "Collaborative Recommender", "Top Anime Recommender")
)

if app_selector == "Content-Based Recommender":
    st.title("Content-Based Recommender System") 
    try:
        anime_data = pd.read_csv(anime_file_path)
        anime_list = anime_data["name"].tolist()
        anime_name = st.selectbox("Select an Anime", anime_list)

        # Set number of recommendations
        max_recommendations = min(len(anime_data), 100)
        n_recommendations = st.slider("Number of Recommendations", 1, max_recommendations, 10)

        # Inject custom CSS for anime name font size
        st.markdown(
            """
            <style>
            .anime-title {
                font-size: 14px !important;
                font-weight: bold;
                text-align: center;
                margin-top: 5px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        ) 
        # Get Recommendations
        if st.button("Get Recommendations"):
            try:
                recommender = ContentBasedRecommender(anime_data)
                recommendations = recommender.get_rec_cosine(anime_name, n_recommendations=n_recommendations,model_path=cosine_tfv)

                if isinstance(recommendations, str):
                    st.warning(recommendations)
                elif recommendations.empty:
                    st.warning("No recommendations found.")
                else:
                    st.write(f"Here are the Content-based Recommendations for {anime_name}:") 
                    cols = st.columns(5)
                    for i, row in enumerate(recommendations.iterrows()):
                        col = cols[i % 5]
                        with col:
                            st.image(row[1]['Image URL'], use_container_width=True)
                            st.markdown(
                                f"<div class='anime-title'>{row[1]['Anime name']}</div>",
                                unsafe_allow_html=True,
                            )
                            st.caption(f"Genres: {row[1]['Genres']} | Rating: {row[1]['Rating']}")

            except FileNotFoundError as e:
                st.error(f"File not found: {str(e)}")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

    except FileNotFoundError:
        st.error(f"File {anime_file_path} not found.")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")

elif app_selector == "Collaborative Recommender":
    st.title("Collaborative Recommender System")
    
    try: 
        animerating_data = pd.read_csv(merged_file_path)  
        # Sidebar for choosing the collaborative filtering method
        collaborative_method = st.sidebar.selectbox(
            "Choose a collaborative filtering method:", 
            ["SVD Collaborative Filtering", "User-Based Collaborative Filtering", "Anime-Based KNN Collaborative Filtering"]
        )

        # User input
        if collaborative_method == "SVD Collaborative Filtering" or collaborative_method == "User-Based Collaborative Filtering": 
            user_ids = animerating_data['user_id'].unique()  # Get unique user IDs
            user_id = st.selectbox("Select a user ID", user_ids) 
            n_recommendations = st.slider("Number of Recommendations:", min_value=1, max_value=50, value=10)
        elif collaborative_method == "Anime-Based KNN Collaborative Filtering": 
            anime_list = animerating_data["name"].dropna().unique().tolist()  # Ensure no NaN values in anime names
            anime_name = st.selectbox("Select an Anime", anime_list)
            n_recommendations = st.slider("Number of Recommendations:", min_value=1, max_value=50, value=10)
   
        # Get recommendations
        if st.button("Get Recommendations"):
            # Load the recommender
            recommender = CollaborativeAnimeRecommender(animerating_data) 
            if collaborative_method == "SVD Collaborative Filtering":
                svd_model = load_object(svd)
                recommendations = recommender.get_svd_recommendations(user_id, n=n_recommendations, svd_model=svd_model) 
                # st.write(recommendations.head())
            elif collaborative_method == "User-Based Collaborative Filtering":
                user_knn_model = load_object(userbasedknn)
                recommendations = recommender.get_user_based_recommendations(user_id, n_recommendations=n_recommendations, knn_user_model=user_knn_model)
            elif collaborative_method == "Anime-Based KNN Collaborative Filtering":
                if anime_name:
                    item_knn_model = load_object(itembasedknn)
                    recommendations = recommender.get_item_based_recommendations(anime_name, n_recommendations=n_recommendations, knn_item_model=item_knn_model)
                else:
                    st.error("Invalid Anime Name. Please enter a valid anime title.")
            
            if isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
                if len(recommendations) < n_recommendations:
                    st.warning(f"Only {len(recommendations)} recommendations available, fewer than the requested {n_recommendations}.")
                st.write(f"Here are the Collaborative Recommendations:")
                cols = st.columns(5)
                for i, row in enumerate(recommendations.iterrows()):
                    col = cols[i % 5]
                    with col:
                        st.image(row[1]['Image URL'], use_container_width=True)
                        st.markdown(
                            f"<div class='anime-title'>{row[1]['Anime Name']}</div>",
                            unsafe_allow_html=True,
                        ) 
                        st.caption(f"Genres: {row[1]['Genres']} | Rating: {row[1]['Rating']}")
            else:
                st.error("No recommendations found.")
    except Exception as e:
        st.error(f"An error occurred: {e}")  