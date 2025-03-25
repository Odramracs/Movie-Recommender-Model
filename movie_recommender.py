import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time

# Load datasets
print("Loading dataset...")
ratings = pd.read_csv("rating.csv")
movies = pd.read_csv("movie.csv")

# Create movie ID to title mapping
movie_id_to_title = movies.set_index("movieId")["title"].to_dict()

# Generate user and movie mappings
user_to_index = {user: i for i, user in enumerate(ratings["userId"].unique())}
movie_to_index = {movie: i for i, movie in enumerate(ratings["movieId"].unique())}

# Map user and movie IDs
ratings["userId"] = ratings["userId"].map(user_to_index)
ratings["movieId"] = ratings["movieId"].map(movie_to_index)

# Convert to tensors
user_tensor = torch.tensor(ratings["userId"].values, dtype=torch.long)
movie_tensor = torch.tensor(ratings["movieId"].values, dtype=torch.long)
rating_tensor = torch.tensor(ratings["rating"].values, dtype=torch.float32)

# Hyperparameters
num_epochs = 5  
batch_size = 512  
embed_size = 20  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

# Custom Dataset
class MovieLensDataset(Dataset):
    def __init__(self, users, movies, ratings):
        self.users, self.movies, self.ratings = users, movies, ratings

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

# Create dataset and loader
dataset = MovieLensDataset(user_tensor, movie_tensor, rating_tensor)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Neural Collaborative Filtering (NCF) Model
class NeuralCF(nn.Module):
    def __init__(self, num_users, num_movies, embed_size=20):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_size)
        self.movie_embed = nn.Embedding(num_movies, embed_size)
        self.layers = nn.Sequential(
            nn.Linear(embed_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, users, movies):
        x = torch.cat([self.user_embed(users), self.movie_embed(movies)], dim=1)
        return self.layers(x).squeeze()

# Initialize model
model = NeuralCF(len(user_to_index), len(movie_to_index), embed_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print(f"Using device: {device}")  # Should print "cuda" if using GPU

# Training loop
print("Training model...\n")
start_time = time.time()

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0

    for batch_idx, (users, movies, ratings) in enumerate(train_loader, 1):
        users, movies, ratings = users.to(device), movies.to(device), ratings.to(device)
        
        optimizer.zero_grad()
        loss = criterion(model(users, movies), ratings)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        print(f"\rEpoch {epoch}/{num_epochs} [{batch_idx}/{len(train_loader)}] Loss: {total_loss/batch_idx:.4f}", end="")

    print(f"\nEpoch {epoch} completed. Avg Loss: {total_loss / len(train_loader):.4f}")

print(f"\n Training completed in {time.time() - start_time:.2f} seconds.")

# Movie recommendation function
def recommend_movies(user_id, top_k=5):
    model.eval()
    user_idx = torch.tensor([user_to_index.get(user_id)], dtype=torch.long).to(device)
    movie_indices = torch.arange(len(movie_to_index), dtype=torch.long).to(device)

    with torch.no_grad():
        predictions = model(user_idx.expand(len(movie_to_index)), movie_indices)

    top_movie_ids = [list(movie_to_index.keys())[i] for i in predictions.argsort(descending=True)[:top_k].cpu().numpy()]
    
    return [movie_id_to_title.get(mid, "Unknown Movie") for mid in top_movie_ids]

# Example Recommendation
print(f" Recommended movies for user 1: {recommend_movies(1)}")