import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# 1. LOAD DATA
# ==========================================

movies = pd.read_csv("data/movies.csv")

movies["genres"] = movies["genres"].fillna("")

print(f"Loaded {len(movies)} movies.")


# ==========================================
# 2. CREATE TF-IDF MATRIX
# ==========================================

vectorizer = TfidfVectorizer(
    token_pattern=r"[^|]+"
)

tfidf_matrix = vectorizer.fit_transform(
    movies["genres"]
)


# ==========================================
# 3. PERSONALIZED RECOMMENDATION FUNCTION
# ==========================================

def personalized_recommend(
    favorite_movies,
    number_of_recommendations=10
):

    favorite_indices = []

    # Find each favorite movie
    for movie_title in favorite_movies:

        matches = movies[
            movies["title"].str.lower() == movie_title.lower()
        ]

        if not matches.empty:

            favorite_indices.append(
                matches.index[0]
            )

    # Check if movies were found
    if not favorite_indices:

        return []


    # ======================================
    # CREATE USER PREFERENCE PROFILE
    # ======================================

    user_profile = np.asarray(
        tfidf_matrix[favorite_indices].mean(axis=0)
    )


    # ======================================
    # COMPARE USER PROFILE WITH ALL MOVIES
    # ======================================

    similarity_scores = cosine_similarity(
        user_profile,
        tfidf_matrix
    ).flatten()


    # ======================================
    # DON'T RECOMMEND ALREADY SELECTED MOVIES
    # ======================================

    for index in favorite_indices:

        similarity_scores[index] = -1


    # ======================================
    # GET TOP RECOMMENDATIONS
    # ======================================

    recommended_indices = np.argsort(
        similarity_scores
    )[::-1][:number_of_recommendations]


    recommendations = []

    for index in recommended_indices:

        recommendations.append({

            "title": movies.iloc[index]["title"],

            "genres": movies.iloc[index]["genres"],

            "similarity": round(
                float(similarity_scores[index]),
                3
            )

        })


    return recommendations


# ==========================================
# 4. TEST THE RECOMMENDATION ENGINE
# ==========================================

if __name__ == "__main__":

    favorites = [
        "Toy Story (1995)",
        "Jumanji (1995)",
        "The Lion King (1994)"
    ]

    print()
    print("==========================================")
    print(" PERSONALIZED RECOMMENDATION ENGINE")
    print("==========================================")

    print()
    print("Your favorite movies:")

    for movie in favorites:

        print(" -", movie)


    results = personalized_recommend(
        favorites,
        10
    )


    print()
    print("Recommended for you:")
    print("------------------------------------------")


    if not results:

        print("No recommendations found.")

    else:

        for i, item in enumerate(
            results,
            start=1
        ):

            print(
                f"{i}. {item['title']}"
            )

            print(
                f"   Genres: {item['genres']}"
            )

            print(
                f"   Match: {item['similarity']}"
            )

            print()