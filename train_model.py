import pickle
from sklearn.neighbors import KNeighborsClassifier

EMBEDDINGS_FILE = "models/embeddings.pkl"
MODEL_FILE = "models/knn_model.pkl"

# Load face embeddings
with open(EMBEDDINGS_FILE, "rb") as file:
    data = pickle.load(file)

X = data["encodings"]
y = data["names"]

print("Number of training samples:", len(X))
print("People:", sorted(set(y)))

if len(X) < 2:
    raise ValueError("You need at least 2 face samples.")

# Use a small KNN model
k = min(3, len(X))

model = KNeighborsClassifier(
    n_neighbors=k,
    weights="distance"
)

# Train
model.fit(X, y)

# Save model
with open(MODEL_FILE, "wb") as file:
    pickle.dump(model, file)

print("KNN model trained successfully!")
print("Model saved to:", MODEL_FILE)