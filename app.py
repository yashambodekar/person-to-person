import pickle
import ast
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# Load model artifacts
try:
    with open("user_cf_model.pkl", "rb") as f:
        artifacts = pickle.load(f)
except FileNotFoundError:
    raise Exception("Model file 'user_cf_model.pkl' not found in D:\\Hack Wave")
except Exception as e:
    raise Exception(f"Error loading model: {str(e)}")

vectorizer = artifacts["vectorizer"]
scaler = artifacts["scaler"]
user_matrix = artifacts["user_matrix"]
df = artifacts["df"]

app = FastAPI()

# Define allowed attributes (with defaults)
class NewUser(BaseModel):
    gender: str = Field(..., description="User gender", pattern="^(M|F|O)$")
    age: Optional[int] = Field(30, description="User age")
    weight: Optional[float] = Field(70.0, description="User weight")
    height: Optional[float] = Field(170.0, description="User height")
    style_preferences: Optional[List[str]] = Field(default_factory=list)
    preferred_colors: Optional[List[str]] = Field(default_factory=list)
    preferred_fabrics: Optional[List[str]] = Field(default_factory=list)
    interested_in: Optional[List[str]] = Field(default_factory=list)

    class Config:
        extra = "ignore"  # Ignore extra attributes not defined above

@app.post("/recommend")
def recommend(user_input: Dict[str, Any], top_k: int = 3):
    """
    Flexible recommend endpoint:
    - Requires gender; other attributes optional (fills defaults)
    - Ignores extra attributes
    - Works with partial JSON
    """
    try:
        new_user = NewUser(**user_input)
    except Exception as e:
        print(f"Warning: Invalid input, using defaults: {str(e)}")
        new_user = NewUser(gender="other")  # Default gender if parsing fails

    # Build user profile text
    profile = (
        new_user.gender + " " +
        " ".join(new_user.style_preferences) + " " +
        " ".join(new_user.preferred_colors) + " " +
        " ".join(new_user.preferred_fabrics) + " " +
        " ".join(new_user.interested_in)
    )
    pref_vec = vectorizer.transform([profile])
    num_vec = scaler.transform([[new_user.age, new_user.weight, new_user.height]])
    new_user_vec = hstack([pref_vec, num_vec])

    # Compute similarity
    sim_scores = cosine_similarity(new_user_vec, user_matrix).flatten()
    top_users = sim_scores.argsort()[-top_k:][::-1]

    # Collect recommendations
    recommended = []
    for u in top_users:
        try:
            recommended.extend(ast.literal_eval(df.iloc[u]["interested_in"]))
        except Exception as e:
            print(f"Warning: Failed to parse interested_in for user {u}: {str(e)}")
            continue
    recs = Counter(recommended).most_common(top_k)

    return {
        "parsed_user": new_user.dict(),
        "similar_users": top_users.tolist(),
        "recommendations": recs
    }

   if __name__ == "__main__":
    # Get the PORT from the environment variable provided by the platform (e.g., Render)
    port = int(os.environ.get("PORT", 8000))
    # '0.0.0.0' is the host that makes the app accessible from outside the container
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
