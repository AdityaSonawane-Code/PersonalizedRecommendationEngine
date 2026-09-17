import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# 1. LOAD DATA
# ==========================================

movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

movies["genres"] = movies["genres"].fillna("")


# ==========================================
# 2. CONTENT-BASED MODEL
# ==========================================

vectorizer = TfidfVectorizer(
    token_pattern=r"[^|]+"
)

tfidf_matrix = vectorizer.fit_transform(
    movies["genres"]
)


# ==========================================
# 3. USER-MOVIE MATRIX
# ==========================================

user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)


# ==========================================
# 4. USER SIMILARITY
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
# 5. CONTENT SCORE
# ==========================================

def get_content_scores(user_id):

    if user_id not in user_movie_matrix.index:

        return np.zeros(len(movies))


    user_ratings = user_movie_matrix.loc[user_id]

    liked_movie_ids = user_ratings[
        user_ratings >= 4
    ].index.tolist()


    liked_indices = []

    for movie_id in liked_movie_ids:

        matches = movies.index[
            movies["movieId"] == movie_id
        ]

        if len(matches) > 0:

            liked_indices.append(
                matches[0]
            )


    if not liked_indices:

        return np.zeros(len(movies))


    # Create user's preference profile
    user_profile = np.asarray(
        tfidf_matrix[liked_indices].mean(axis=0)
    )


    scores = cosine_similarity(
        user_profile,
        tfidf_matrix
    ).flatten()


    return scores


# ==========================================
# 6. COLLABORATIVE SCORE
# ==========================================

def get_collaborative_scores(user_id):

    scores = np.zeros(len(movies))


    if user_id not in user_movie_matrix.index:

        return scores


    similar_users = user_similarity_df[
        user_id
    ].sort_values(
        ascending=False
    )


    similar_users = similar_users[
        similar_users.index != user_id
    ]


    top_users = similar_users.head(20)


    for similar_user_id, similarity in top_users.items():

        if similarity <= 0:

            continue


        similar_ratings = (
            user_movie_matrix.loc[
                similar_user_id
            ]
        )


        liked_movies = similar_ratings[
            similar_ratings >= 4
        ]


        for movie_id, rating in liked_movies.items():

            movie_indices = movies.index[
                movies["movieId"] == movie_id
            ]


            if len(movie_indices) == 0:

                continue


            movie_index = movie_indices[0]


            scores[movie_index] += (
                similarity * rating
            )


    return scores


# ==========================================
# 7. HYBRID RECOMMENDER
# ==========================================

def recommend(
    user_id,
    number_of_recommendations=10,
    content_weight=0.5,
    collaborative_weight=0.5
):

    print("\nCalculating recommendations...")


    # Get both scores
    content_scores = get_content_scores(
        user_id
    )

    collaborative_scores = (
        get_collaborative_scores(user_id)
    )


    # ======================================
    # NORMALIZE SCORES
    # ======================================

    def normalize(scores):

        minimum = scores.min()
        maximum = scores.max()

        if maximum == minimum:

            return np.zeros_like(scores)

        return (
            (scores - minimum)
            / (maximum - minimum)
        )


    content_scores = normalize(
        content_scores
    )

    collaborative_scores = normalize(
        collaborative_scores
    )


    # ======================================
    # COMBINE BOTH MODELS
    # ======================================

    hybrid_scores = (
        content_weight * content_scores
        +
        collaborative_weight
        * collaborative_scores
    )


    # ======================================
    # REMOVE ALREADY RATED MOVIES
    # ======================================

    if user_id in user_movie_matrix.index:

        rated_movie_ids = user_movie_matrix.loc[
            user_id
        ]

        rated_movie_ids = rated_movie_ids[
            rated_movie_ids > 0
        ].index


        for movie_id in rated_movie_ids:

            movie_indices = movies.index[
                movies["movieId"] == movie_id
            ]

            if len(movie_indices) > 0:

                hybrid_scores[
                    movie_indices[0]
                ] = -1


    # ======================================
    # GET TOP MOVIES
    # ======================================

    top_indices = np.argsort(
        hybrid_scores
    )[::-1][
        :number_of_recommendations
    ]


    recommendations = []


    for index in top_indices:

        recommendations.append({

            "title": movies.iloc[index]["title"],

            "genres": movies.iloc[index]["genres"],

            "score": round(
                float(hybrid_scores[index]),
                4
            )

        })


    return recommendations


# ==========================================
# 8. TEST
# ==========================================

if __name__ == "__main__":

    user_id = 1


    print()
    print("==========================================")
    print("       HYBRID RECOMMENDATION ENGINE")
    print("==========================================")

    print()
    print("User:", user_id)

    results = recommend(
        user_id,
        number_of_recommendations=10
    )


    print()
    print("Recommended for you:")
    print("------------------------------------------")


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
            f"   Hybrid Score: {item['score']}"
        )

        print()