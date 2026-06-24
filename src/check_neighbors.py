import numpy as np

# Load vectors
data = np.load("results/word_vectors.npz", allow_pickle=True)
embeddings = data["embeddings"]
raw_word2idx = data["word2idx"]

word2idx = raw_word2idx.item() if hasattr(raw_word2idx, 'item') else raw_word2idx.tolist()
idx2word = {i: w for w, i in word2idx.items()}

# Normalize
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norms[norms == 0] = 1e-9
normed = embeddings / norms

def get_neighbors(word, top_k=5):
    if word not in word2idx:
        print(f"'{word}' not in vocabulary.")
        return
    idx = word2idx[word]
    vec = normed[idx]
    sims = normed @ vec
    
    # Sort descending
    nearest = np.argsort(-sims)[:top_k+1]
    print(f"\nClosest words to '{word}':")
    for n in nearest:
        if n == idx: continue
        print(f"  {idx2word[n]}: {sims[n]:.4f}")

# Test some common words
get_neighbors("king")
get_neighbors("france")
get_neighbors("apple")