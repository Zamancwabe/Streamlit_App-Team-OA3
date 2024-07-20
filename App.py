import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import numpy as np

# Load data with caching
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# Content-Based Filtering
def content_based_recommendations(user_input, data):
    data['genre'] = data['genre'].fillna('')
    data['combined_features'] = data['genre'] + ' ' + data['name']
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['combined_features'])
    cosine_sim = cosine_similarity(count_matrix)

    recommendations = []
    for anime in user_input:
        try:
            anime_index = data[data['name'] == anime].index[0]
            similar_animes = list(enumerate(cosine_sim[anime_index]))
            sorted_similar_animes = sorted(similar_animes, key=lambda x: x[1], reverse=True)
            recommendations.extend([data.iloc[i[0]]['name'] for i in sorted_similar_animes[1:11]]) 
        except IndexError:
            recommendations.append("Anime not found in dataset")
    return recommendations

# Collaborative-Based Filtering
def collaborative_based_recommendations(user_input, data):
    data['rating'] = data['rating'].fillna(0)
    data = data.astype({'anime_id': 'int64'})
    data = data.drop_duplicates(['anime_id', 'rating'])
    data_pivot = data.pivot_table(index='anime_id', columns='name', values='rating').fillna(0)

    user_ratings = pd.Series(0, index=data_pivot.columns)
    for anime in user_input:
        try:
            user_ratings[anime] = 10 
        except KeyError:
            pass # Ignore if anime not in dataset
    
    cos_sim = cosine_similarity([user_ratings], data_pivot)
    df_cos_sim = pd.DataFrame(cos_sim[0], index=data_pivot.index)
    df_cos_sim.columns = ['Cosine Similarity']
    df_cos_sim = df_cos_sim.sort_values(by='Cosine Similarity', ascending=False)
    
    if df_cos_sim.iloc[1, 0] < 0.4:  
        return ["No suitable recommendations found!"]
    
    top_anime_id = df_cos_sim.index[1]
    recommendations = data_pivot.columns[(data_pivot.loc[top_anime_id] == 10) & (data_pivot.columns != user_input[0])][:10].tolist()

    return recommendations

# Main Streamlit App
def main():
    st.title('Anime Recommendation App')

    # Image display (replace with your actual path)
    image_path = r'C:\Users\Zaman\Downloads\the-top-25-greatest-anime-characters-of-all-time_6uv2.jpg'
    image = Image.open(image_path)
    st.image(image, caption='Anime Recommendations', use_column_width=True)

    # File upload
    uploaded_file = st.file_uploader("Upload your anime dataset CSV file", type="csv")

    if uploaded_file is not None:
        anime_data = load_data(uploaded_file)
        
        # Algorithm selection
        st.header('Select an algorithm:')
        algorithm = st.radio('', ('Content Based Filtering', 'Collaborative Based Filtering'))

        # User input
        st.header('Enter your three favorite anime:')
        user_input = [
            st.text_input('First anime:'),
            st.text_input('Second anime:'),
            st.text_input('Third anime:')
        ]

        # Recommend button
        if st.button('Recommend'):
            if algorithm == 'Content Based Filtering':
                recommendations = content_based_recommendations(user_input, anime_data)
            else:
                recommendations = collaborative_based_recommendations(user_input, anime_data)

            # Display recommendations
            st.subheader('Recommendations:')
            for i, rec in enumerate(recommendations):
                st.write(f"{i+1}. {rec}")

# Run the app
main() 

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Home", "About", "Recommendations", "EDA"])