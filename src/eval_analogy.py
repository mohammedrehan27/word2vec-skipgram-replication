"""
Evaluate trained word vectors on the word analogy task, the same metric
used in Mikolov et al. (2013).

For each analogy "a : b :: c : d" (e.g. "Athens : Greece :: Oslo : Norway"),
we compute:
    predicted_vector = vec(b) - vec(a) + vec(c)
and find the nearest neighbor to predicted_vector by cosine similarity,
excluding a, b, and c themselves from the candidates. If the nearest
neighbor is d, it's counted correct.

Download the standard analogy question file from:
    http://download.tensorflow.org/data/questions-words.txt
(this is the same file commonly used to reproduce word2vec's reported
analogy accuracy numbers)

Usage:
    python src/eval_analogy.py --vectors results/word_vectors.npz \
        --questions data/questions-words.txt
"""

import argparse
import urllib.request
import os
import numpy as np

QUESTIONS_URL = "http://download.tensorflow.org/data/questions-words.txt"


def download_questions(path):
    if os.path.exists(path):
        return path
    print("Downloading analogy question set...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve(QUESTIONS_URL, path)
    return path


def load_questions(path):
    """
    File format: most lines are "wordA wordB wordC wordD".
    Lines starting with ':' mark a new category, e.g. ": capital-common-countries"
    Returns list of (category, (a, b, c, d)) tuples, all lowercased.
    """
    questions = []
    category = "unknown"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(":"):
                category = line[1:].strip()
                continue
            parts = line.lower().split()
            if len(parts) == 4:
                questions.append((category, tuple(parts)))
    return questions


def evaluate_analogies(embeddings, word2idx, idx2word, questions, max_questions=None):
    """
    embeddings: (vocab_size, dim) numpy array
    word2idx: dict word -> index
    Returns overall accuracy and per-category breakdown.
    """
    # normalize embeddings for cosine similarity via dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    normed = embeddings / norms

    correct = 0
    total = 0
    skipped_oov = 0
    category_stats = {}

    if max_questions:
        questions = questions[:max_questions]

    for category, (a, b, c, d) in questions:
        if any(w not in word2idx for w in (a, b, c, d)):
            skipped_oov += 1
            continue

        ia, ib, ic, id_ = (word2idx[a], word2idx[b], word2idx[c], word2idx[d])
        target_vec = normed[ib] - normed[ia] + normed[ic]
        target_vec = target_vec / (np.linalg.norm(target_vec) + 1e-9)

        sims = normed @ target_vec  # cosine sim to every word in vocab
        
        # exclude the three input words from candidates
        sims[ia] = -np.inf
        sims[ib] = -np.inf
        sims[ic] = -np.inf

        best_idx = int(np.argmax(sims))
        is_correct = (best_idx == id_)

        total += 1
        correct += int(is_correct)

        cat = category_stats.setdefault(category, {"correct": 0, "total": 0})
        cat["total"] += 1
        cat["correct"] += int(is_correct)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, total, skipped_oov, category_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", default="results/word_vectors.npz")
    parser.add_argument("--questions", default="data/questions-words.txt")
    parser.add_argument("--max_questions", type=int, default=None,
                        help="Cap number of questions for a quick sanity check")
    args = parser.parse_args()

    download_questions(args.questions)

    # Load saved vectors
    data = np.load(args.vectors, allow_pickle=True)
    embeddings = data["embeddings"]
    
    raw_word2idx = data["word2idx"]
    raw_idx2word = data["idx2word"]

    # Safe extraction depending on how numpy packed the dictionaries
    word2idx = raw_word2idx.item() if hasattr(raw_word2idx, 'item') else raw_word2idx.tolist()
    idx2word = raw_idx2word.item() if hasattr(raw_idx2word, 'item') else raw_idx2word.tolist()

    questions = load_questions(args.questions)
    print(f"Loaded {len(questions)} analogy questions.")

    accuracy, total, skipped, category_stats = evaluate_analogies(
        embeddings, word2idx, idx2word, questions, max_questions=args.max_questions
    )

    print(f"\nEvaluated on {total} questions (skipped {skipped} due to OOV words)")
    print(f"Overall analogy accuracy: {accuracy * 100:.2f}%")

    print("\nPer-category breakdown:")
    for cat, stats in sorted(category_stats.items()):
        if stats["total"] == 0:
            continue
        cat_acc = stats["correct"] / stats["total"] * 100
        print(f"  {cat:30s} {cat_acc:6.2f}%  ({stats['correct']}/{stats['total']})")


if __name__ == "__main__":
    main()