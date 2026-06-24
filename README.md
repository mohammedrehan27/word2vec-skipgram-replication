# Word2Vec (Skip-gram with Negative Sampling) — From-Scratch Reproduction

Implementation of the Skip-gram model with Negative Sampling from:
- Mikolov et al., "Efficient Estimation of Word Representations in Vector Space" (2013)
- Mikolov et al., "Distributed Representations of Words and Phrases and their Compositionality" (2013)

See `PAPER_NOTES.md` for the paper's central claim, method, and reported metrics.

## Setup

```bash
pip install torch numpy
```

No GPU required — this runs fine on CPU, just slower. (`device` is auto-detected.)

## Run

### 1. Train

```bash
cd src
python train.py --epochs 5 --embedding_dim 100 --window 5 --neg_samples 5
```

This will:
- Auto-download the `text8` corpus (~31MB zipped) into `../data/` on first run
- Build vocabulary, subsample frequent words, generate skip-gram training pairs
- Train Skip-gram with Negative Sampling
- Save trained word vectors to `../results/word_vectors.npz`

For a quick smoke test before committing to a full run:
```bash
python train.py --epochs 1 --max_tokens 200000
```

### 2. Evaluate (word analogy accuracy — the paper's metric)

```bash
python eval_analogy.py --vectors ../results/word_vectors.npz
```

This auto-downloads the standard Google analogy question set and reports:
- Overall accuracy
- Per-category breakdown (semantic vs. syntactic analogy types)

## Expected Results

The original paper trained on a ~6-100B word Google News corpus and reported
analogy accuracies in the 50-70%+ range depending on embedding dimension and
training data size (see Table 2/3/4 in the 2013 papers).

We train on `text8` (~17M words after cleaning), which is roughly 3 orders of
magnitude smaller. Expect substantially lower accuracy as a result — this is
a known, explainable gap due to corpus size, not a bug. The goal is to verify
the *architecture and training procedure are correct* and that the qualitative
behavior (meaningful nearest neighbors, some correct analogies) holds, not to
match the paper's absolute numbers.

See `results/` for our actual run logs and accuracy numbers.

## Implementation Notes

- The model (`src/model.py`) is implemented from scratch using only
  `torch.nn.Embedding` and basic tensor ops — no prebuilt word2vec library
  (e.g. `gensim`) is used for the model itself.
- `torchvision`/`HuggingFace datasets` etc. are not used; data loading
  (`src/data.py`) is also from scratch, using only `urllib`, standard Python,
  and `numpy`.
- Negative sampling distribution uses unigram frequency ^0.75, as specified
  in the paper.
- Frequent-word subsampling follows the paper's formula
  (Mikolov et al. 2013b, eq. 5).
