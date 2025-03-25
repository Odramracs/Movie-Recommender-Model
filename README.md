# Movie Recommendation System 
This project implements a **Neural Collaborative Filtering (NCF) model** using **PyTorch** to recommend movies based on user ratings. 
The model is trained on the **MovieLens dataset** and utilizes **GPU acceleration** if available.

---

## Features
**Collaborative Filtering**: Uses user-movie interactions to recommend movies.  
**Neural Network Model**: Deep learning-based recommendation system.  
**GPU Acceleration**: Automatically detects and uses CUDA-enabled GPU.  
**Movie Title Mapping**: Outputs readable movie recommendations instead of just IDs.  

---

##  Install Dependencies
- Ensure you have **Python 3.8+** installed
- pip install torch torchvision torchaudio pandas numpy

To enable GPU support: 
- install CUDA-enabled PyTorch
  - pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

## Download the MovieLens Dataset
https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset?select=rating.csv

## Run the model 
python movie_recommender.py

## Hyperparameters
- change based on needs
   
num_epochs = 3         
batch_size = 256       
embed_size = 20        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
