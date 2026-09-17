import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==============================
# 1. LOAD MOVIE DATA
# ==============================

movies = pd.read_csv("data/movies.csv")

print(f"Loaded {len(movies)} movies.")


# ==============================
# 2. PREPARE GENRE DATA
# ==============================

movies["genres"] = movies["genres"].fillna("")


# ==============================
# 3. CONVERT GENRES INTO NUMBERS
# ==============================

vectorizer = TfidfVectorizer(token_pattern=r"[^|]+")

tfidf_matrix = vectorizer.fit_transform(movies["genres"])


# ==============================
# 4. CALCULATE MOVIE SIMILARITY
# ==============================

similarity_matrix = cosine_similarity(tfidf_matrix)


# ==============================
# 5. RECOMMENDATION FUNCTION
# ==============================

def recommend(movie_title, number_of_recommendations=10):

    matches = movies[
        movies["title"].str.lower() == movie_title.lower()
    ]

    if matches.empty:
        return []

    movie_index = matches.index[0]

    similarity_scores = list(
        enumerate(similarity_matrix[movie_index])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Remove the movie itself
    similarity_scores = similarity_scores[1:]

    recommendations = []

    for index, score in similarity_scores[:number_of_recommendations]:

        recommendations.append({
            "title": movies.iloc[index]["title"],
            "genres": movies.iloc[index]["genres"],
            "similarity": round(float(score), 3)
        })

    return recommendations


# ==============================
# 6. TEST THE ENGINE
# ==============================

if __name__ == "__main__":

    movie = "Toy Story (1995)"

    results = recommend(movie, 10)

    print()
    print("========================================")
    print(" PERSONALIZED RECOMMENDATION ENGINE")
    print("========================================")
    print()
    print("Because you liked:", movie)
    print()

    if not results:

        print("Movie not found.")

    else:

        for i, item in enumerate(results, start=1):

            print(
                f"{i}. {item['title']}"
            )

            print(
                f"   Genres: {item['genres']}"
            )

            print(
                f"   Similarity: {item['similarity']}"
            )

            print()