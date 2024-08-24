import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# Relative paths to the files
ANIME_FILE_PATH = 'anime.xlsx'
TRAIN_FILE_PATH = 'train.csv'

# Load data
@st.cache_data
def load_anime_data():
    return pd.read_excel(ANIME_FILE_PATH)

@st.cache_data
def load_train_data():
    return pd.read_csv(TRAIN_FILE_PATH)

# Content-Based Filtering
def content_based_recommendations(user_input, data):
    recommendations = []
    
    # Ensure that genre and name are strings before concatenating
    data['genre'] = data['genre'].fillna('').astype(str)
    data['name'] = data['name'].fillna('').astype(str)
    data['combined_features'] = data['genre'] + ' ' + data['name']

    # Create the count matrix and compute the cosine similarity
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['combined_features'])
    cosine_sim = cosine_similarity(count_matrix)
    
    def get_title_from_index(index):
        return data.iloc[index]['name']

    def get_index_from_title(title):
        try:
            return data[data['name'].str.lower() == title.lower()].index[0]
        except IndexError:
            return None

    for anime in user_input:
        if anime:
            anime_index = get_index_from_title(anime)
            if anime_index is not None:
                similar_animes = list(enumerate(cosine_sim[anime_index]))
                sorted_similar_animes = sorted(similar_animes, key=lambda x: x[1], reverse=True)
                recommendations.extend([get_title_from_index(element[0]) for element in sorted_similar_animes[1:11]])
            else:
                recommendations.append(f"Anime '{anime}' not found in dataset")

    return recommendations

# Collaborative Filtering
def collaborative_filtering_recommendations(user_input, train_data, anime_data):
    recommendations = []
    
    # Create the user-item interaction matrix
    user_item_matrix = train_data.pivot_table(index='user_id', columns='anime_id', values='rating')
    
    def get_anime_id(anime_name):
        anime_id = anime_data[anime_data['name'].str.lower() == anime_name.lower()]['anime_id']
        return anime_id.values[0] if not anime_id.empty else None

    def get_anime_title(anime_id):
        title = anime_data[anime_data['anime_id'] == anime_id]['name']
        return title.values[0] if not title.empty else None
    
    for anime in user_input:
        if anime:
            anime_id = get_anime_id(anime)
            if anime_id is not None:
                # Calculate similarity for the given anime_id
                anime_ratings = user_item_matrix[anime_id]
                similar_animes = user_item_matrix.corrwith(anime_ratings)
                similar_animes = similar_animes.dropna().sort_values(ascending=False).head(10)
                
                recommendations.extend([get_anime_title(anime_id) for anime_id in similar_animes.index if anime_id != anime_id])
            else:
                recommendations.append(f"Anime '{anime}' not found in dataset")
    
    return recommendations

# Main Streamlit App
def main():
    st.set_page_config(page_title="AniMatch - Your Anime Discovery Companion", page_icon="🎥", layout="wide")

    st.markdown(
        """
        <style>
        .stApp {
            background-color: #eaeaea;
            color: #333;
        }
        .sidebar .sidebar-content {
            background-color: #404040;
            color: #f5f5f5;
        }
        .sidebar .sidebar-content .stRadio {
            color: #f5f5f5;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio(
        "Go to", 
        ["Home", "Team", "About", "Guidelines", "Recommendations", "Community", "Feedback"]
    )

    if selection == "Home":
        st.title('Welcome to AniMatch - Your Personalized Anime Discovery Companion')

        # Update the image path to be relative
        image_path = 'Picture.jpg'
        
        try:
            # Open and display the image
            image = Image.open(image_path)
            st.image(image, caption='Anime Recommendations', use_column_width=True)
        except FileNotFoundError:
            # Display an error message if the image file is not found
            st.error("Image file not found. Please make sure 'Picture.jpg' is in the correct directory.")

        st.write("AniMatch helps you find the perfect anime to watch based on your preferences and viewing history.")
    
    elif selection == "Team":
        st.title("Our Team")
        st.write("Meet the amazing team behind AniMatch:")
        st.write("Zamancwabe Makhathini - Data Scientist")
        st.write("Asanda Gambu - Github Manager")
        st.write("Cleragy Kanuni - Project Leader")
        st.write("Phumzile Sibiya - Data Scientist")
        st.write("Coceka Keto - Data Scientist")
        st.write("Keamomegetshwe Mothoa - Project Manager")
        

    elif selection == "About":
        st.title("About AniMatch")
        st.write("""
        At AniMatch, we're more than just a recommendation engine—we're your gateway to discovering the world of anime like never before. 
        Our platform was born in 2024, the result of six passionate tech enthusiasts coming together to blend their expertise in data science, 
        machine learning, and software development. With prior experience in machine learning, regression analysis, Python, and Power BI, 
        we set out on a mission to create something extraordinary.

        *Our Mission:* To create the most personalized and efficient anime recommendation system, helping users discover new anime that suits their tastes. 
        We believe that every anime fan deserves a curated experience that caters to their unique preferences.

        *Our Vision:* To be the leading platform for anime recommendations, enhancing the viewing experience for anime enthusiasts worldwide. 
        We aim to become the go-to platform where every anime lover finds their next favorite show.

        *What We Offer:*
        - *Personalized Recommendations:* Our cutting-edge algorithms combine content-based and collaborative filtering to suggest anime tailored specifically to your tastes.
        - *Community Engagement:* Beyond recommendations, AniMatch fosters a vibrant community where users can discuss, share, and explore anime together.
        - *User-Friendly Interface:* Designed with you in mind, our platform is intuitive and easy to navigate, ensuring a seamless experience.

        *Our Story:* We started as a small team with big dreams. United by our love for anime and technology, we recognized the need for a more personalized way to explore the vast world of anime. 
        Since our inception, we've been driven by a simple yet powerful idea: to make anime discovery as enjoyable and effortless as watching your favorite show.

        Join us on our journey as we continue to innovate, bringing you the best anime recommendations and a community of like-minded enthusiasts.
        """)

    elif selection == "Guidelines":
        st.title("Guidelines")
        st.write("""
        1. Use the pre-loaded dataset: Ensure your dataset contains anime titles, genres, and ratings.
        2. Choose your favorite anime: Input up to three anime titles you like.
        3. Get Recommendations: Based on your input, AniMatch will suggest new anime for you.
        4. Explore Community Features: Share your recommendations and see what others are watching.
        """)

    elif selection == "Recommendations":
        st.title('Anime Recommendation App')

        anime_data = load_anime_data()
        train_data = load_train_data()

        st.header('Select an algorithm:')
        algorithm = st.radio('', ('Content Based Filtering', 'Collaborative Based Filtering'))

        st.header('Enter your three favorite anime:')
        user_input = [
            st.text_input('First anime:'),
            st.text_input('Second anime:'),
            st.text_input('Third anime:')
        ]

        if st.button('Recommend'):
            recommendations = []
            if algorithm == 'Content Based Filtering':
                recommendations = content_based_recommendations(user_input, anime_data)
            elif algorithm == 'Collaborative Based Filtering':
                recommendations = collaborative_filtering_recommendations(user_input, train_data, anime_data)

            st.subheader('Recommendations:')
            if recommendations:
                for i, rec in enumerate(recommendations):
                    st.write(f"{i+1}. {rec}")
            else:
                st.write("No recommendations found.")

        st.write("Understanding RMSE: Root Mean Square Error (RMSE) measures the accuracy of the recommendations. The lower the RMSE, the better the prediction.")

    elif selection == "Community":
        st.title("Community Features")
        st.write("Join discussions, share your watchlist, and see what others are watching.")

        st.header("Discussion Forum")
        st.text_area("What's on your mind? Share your thoughts about the latest anime you watched!")

        st.header("Community Recommendations")
        st.write("Top picks from other AniMatch users:")
        community_recs = ["One Piece", "Naruto", "Bleach", "Attack on Titan", "My Hero Academia"]
        for i, rec in enumerate(community_recs):
            st.write(f"{i+1}. {rec}")

    elif selection == "Feedback":
        st.title("Feedback")
        st.write("We value your feedback! Please rate your experience and provide comments on the recommendations.")
        rating = st.slider("Rate the recommendations:", 1, 5, 3)
        comments = st.text_area("Additional Comments:")
        
        if st.button("Submit Feedback"):
            st.success("Thank you for your feedback!")

if __name__ == '__main__':
    main()
