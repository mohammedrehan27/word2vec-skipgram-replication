"""
Train Skip-gram with Negative Sampling on the text8 corpus.

Usage:
    python src/train.py --epochs 5 --embedding_dim 100 --window 5 --neg_samples 5
"""

import argparse
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data import (
    download_text8,
    load_corpus,
    build_vocab,
    subsample_frequent_words,
    build_negative_sampling_distribution,
    generate_skipgram_pairs,
)
from model import SkipGramNegSampling


class SkipGramDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        return center, context


def sample_negatives(neg_dist_tensor, batch_size, k):
    """Sample (batch_size, k) negative word indices using torch.multinomial,
    which is much faster than per-example numpy sampling for large batches."""
    return torch.multinomial(neg_dist_tensor, batch_size * k, replacement=True).view(batch_size, k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--neg_samples", type=int, default=5)
    parser.add_argument("--min_count", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--max_tokens", type=int, default=None,
                         help="Optionally cap corpus size for quick debug runs")
    parser.add_argument("--out", default="results/word_vectors.npz")
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    corpus_path = download_text8(args.data_dir)
    print("Loading corpus...")
    tokens = load_corpus(corpus_path)
    if args.max_tokens:
        tokens = tokens[: args.max_tokens]
    print(f"Total raw tokens: {len(tokens)}")

    word2idx, idx2word, counts, indexed = build_vocab(tokens, min_count=args.min_count)
    vocab_size = len(word2idx)
    print(f"Vocab size (min_count={args.min_count}): {vocab_size}")

    print("Subsampling frequent words...")
    indexed = subsample_frequent_words(indexed, counts)
    print(f"Tokens after subsampling: {len(indexed)}")

    print("Generating skip-gram pairs...")
    pairs = generate_skipgram_pairs(indexed, window_size=args.window)
    print(f"Total training pairs: {len(pairs)}")

    neg_dist = build_negative_sampling_distribution(counts, vocab_size)
    neg_dist_tensor = torch.tensor(neg_dist, dtype=torch.float)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = SkipGramDataset(pairs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = SkipGramNegSampling(vocab_size, args.embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    neg_dist_tensor = neg_dist_tensor.to(device)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(args.epochs):
        start = time.time()
        total_loss = 0.0
        n_batches = 0

        for centers, contexts in loader:
            centers = centers.to(device)
            contexts = contexts.to(device)
            negs = sample_negatives(neg_dist_tensor, centers.size(0), args.neg_samples)

            optimizer.zero_grad()
            loss = model(centers, contexts, negs)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if n_batches % 2000 == 0:
                print(f"  epoch {epoch+1} batch {n_batches}/{len(loader)} "
                      f"avg loss so far: {total_loss / n_batches:.4f}")

        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{args.epochs} done. "
              f"avg loss: {total_loss / max(n_batches,1):.4f}  ({elapsed:.1f}s)")

    embeddings = model.get_embeddings()
    np.savez(args.out, embeddings=embeddings,
             word2idx=word2idx, idx2word=idx2word)
    print(f"Saved embeddings to {args.out}")


if __name__ == "__main__":
    main()
