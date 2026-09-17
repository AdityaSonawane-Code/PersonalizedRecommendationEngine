from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from model.hybrid_recommender import recommend


app = FastAPI(
    title="MovieMind",
    description="Personalized Movie Recommendation Engine",
    version="1.0"
)


@app.get("/api/recommend/{user_id}")
def get_recommendations(user_id: int, limit: int = 10):

    results = recommend(
        user_id=user_id,
        number_of_recommendations=limit
    )

    return {
        "user_id": user_id,
        "recommendations": results
    }


# Frontend must be mounted LAST
app.mount(
    "/",
    StaticFiles(
        directory="frontend",
        html=True
    ),
    name="frontend"
)