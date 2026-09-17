import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# 1. LOAD DATA
# ==========================================

ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")

print(f"Loaded {len(ratings)} ratings.")


# ==========================================
# 2. CREATE USER-MOVIE MATRIX
# ==========================================

user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)


print(
    f"User-movie matrix: "
    f"{user_movie_matrix.shape[0]} users x "
    f"{user_movie_matrix.shape[1]} movies"
)


# ==========================================
# 3. CALCULATE USER SIMILARITY
# ==========================================

user_similarity = cosine_similarity(
    user_movie_matrix
)

user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)


# ==========================================
# 4. RECOMMEND FOR A USER
# ==========================================

def recommend_for_user(
    user_id,
    number_of_recommendations=10
):

    if user_id not in user_movie_matrix.index:

        return []


    # Get similarity scores for this user
    similar_users = user_similarity_df[
        user_id
    ].sort_values(
        ascending=False
    )


    # Remove the user themselves
    similar_users = similar_users[
        similar_users.index != user_id
    ]


    # Take top similar users
    top_users = similar_users.head(20)


    # Movies already rated by the target user
    user_ratings = user_movie_matrix.loc[user_id]

    rated_movies = user_ratings[
        user_ratings > 0
    ].index


    # ======================================
    # CALCULATE RECOMMENDATION SCORES
    # ======================================

    recommendation_scores = {}


    for similar_user_id, similarity in top_users.items():

        if similarity <= 0:
            continue


        similar_user_ratings = (
            user_movie_matrix.loc[similar_user_id]
        )


        # Movies liked by similar user
        liked_movies = similar_user_ratings[
            similar_user_ratings >= 4
        ]


        for movie_id, rating in liked_movies.items():

            # Don't recommend movies already rated
            if movie_id in rated_movies:
                continue


            if movie_id not in recommendation_scores:

                recommendation_scores[movie_id] = 0


            recommendation_scores[movie_id] += (
                similarity * rating
            )


    # ======================================
    # SORT RECOMMENDATIONS
    # ======================================

    sorted_recommendations = sorted(
        recommendation_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )


    # ======================================
    # CONVERT MOVIE IDs TO TITLES
    # ======================================

    recommendations = []


    for movie_id, score in sorted_recommendations[
        :number_of_recommendations
    ]:

        movie_info = movies[
            movies["movieId"] == movie_id
        ]


        if movie_info.empty:
            continue


        recommendations.append({

            "title": movie_info.iloc[0]["title"],

            "genres": movie_info.iloc[0]["genres"],

            "score": round(
                float(score),
                3
            )

        })


    return recommendations


# ==========================================
# 5. TEST
# ==========================================

if __name__ == "__main__":

    user_id = 1

    print()
    print("==========================================")
    print(" COLLABORATIVE RECOMMENDATION ENGINE")
    print("==========================================")

    print()
    print("Recommendations for User:", user_id)

    results = recommend_for_user(
        user_id,
        10
    )

    print()
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
                f"   Score: {item['score']}"
            )

            print()
