"""
Data pipeline for Skip-gram word2vec training.

Handles:
- Downloading/loading the text8 corpus
- Building a vocabulary (with min-count filtering, like the paper)
- Subsampling of frequent words (Mikolov et al., 2013b, section 2.3)
- Generating (center, context) skip-gram pairs from a window
- Building the unigram^0.75 distribution used for negative sampling
"""

import os
import random
import urllib.request
import zipfile
from collections import Counter

TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"


def download_text8(data_dir="data"):
    """Download and extract the text8 corpus if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "text8.zip")
    txt_path = os.path.join(data_dir, "text8")

    if os.path.exists(txt_path):
        print(f"Found existing corpus at {txt_path}")
        return txt_path

    print("Downloading text8 corpus...")
    urllib.request.urlretrieve(TEXT8_URL, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    return txt_path


def load_corpus(path):
    """text8 is one long line of lowercase, space-separated tokens."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text.split()


def build_vocab(tokens, min_count=5):
    """
    Build word <-> index mappings, filtering out rare words.
    Returns: word2idx, idx2word, word_counts (Counter), filtered token list as indices.
    """
    counts = Counter(tokens)
    # keep only words occurring at least min_count times
    vocab_words = [w for w, c in counts.items() if c >= min_count]
    # sort by frequency descending for readability/debugging
    vocab_words.sort(key=lambda w: -counts[w])

    word2idx = {w: i for i, w in enumerate(vocab_words)}
    idx2word = {i: w for w, i in word2idx.items()}

    # convert corpus to indices, dropping OOV (filtered) words entirely
    indexed = [word2idx[w] for w in tokens if w in word2idx]

    filtered_counts = {word2idx[w]: c for w, c in counts.items() if w in word2idx}

    return word2idx, idx2word, filtered_counts, indexed


def subsample_frequent_words(indexed_tokens, word_counts, threshold=1e-5):
    """
    Mikolov et al. 2013b, eq. (5): randomly discard frequent words.
    P(discard word w) = 1 - sqrt(threshold / freq(w))
    where freq(w) is the word's relative frequency in the corpus.
    """
    total_count = sum(word_counts.values())
    freqs = {idx: c / total_count for idx, c in word_counts.items()}

    def keep_prob(idx):
        f = freqs[idx]
        # guard against f < threshold producing negative under sqrt
        ratio = threshold / f if f > 0 else 1.0
        p_discard = max(0.0, 1.0 - (ratio ** 0.5))
        return 1.0 - p_discard

    subsampled = [idx for idx in indexed_tokens if random.random() < keep_prob(idx)]
    return subsampled


def build_negative_sampling_distribution(word_counts, vocab_size, power=0.75):
    """
    Returns an array of sampling probabilities over the vocabulary,
    proportional to count(w)^0.75, as specified in Mikolov et al. 2013b.
    """
    counts_arr = [word_counts.get(i, 0) for i in range(vocab_size)]
    weighted = [c ** power for c in counts_arr]
    total = sum(weighted)
    probs = [w / total for w in weighted]
    return probs


def generate_skipgram_pairs(indexed_tokens, window_size=5):
    """
    For each position i, pair the center word with each context word
    within a random window size in [1, window_size] (the paper uses a
    dynamic window, sampling window size uniformly each time -- this
    gives nearby words more training weight than far ones).

    Returns a list of (center_idx, context_idx) tuples.
    """
    pairs = []
    n = len(indexed_tokens)
    for i, center in enumerate(indexed_tokens):
        win = random.randint(1, window_size)
        start = max(0, i - win)
        end = min(n, i + win + 1)
        for j in range(start, end):
            if j == i:
                continue
            pairs.append((center, indexed_tokens[j]))
    return pairs


if __name__ == "__main__":
    # quick smoke test on a tiny synthetic corpus (no internet needed)
    tokens = ("the quick brown fox jumps over the lazy dog the fox runs " * 50).split()
    word2idx, idx2word, counts, indexed = build_vocab(tokens, min_count=1)
    print("vocab size:", len(word2idx))

    sub = subsample_frequent_words(indexed, counts)
    print("tokens before/after subsampling:", len(indexed), len(sub))

    pairs = generate_skipgram_pairs(indexed[:20], window_size=2)
    print("example pairs:", pairs[:10])

    neg_dist = build_negative_sampling_distribution(counts, len(word2idx))
    print("neg sampling dist sums to:", sum(neg_dist))
