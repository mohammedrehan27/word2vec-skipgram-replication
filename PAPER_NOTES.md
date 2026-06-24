# Paper Notes: Word2Vec (Mikolov et al., 2013)

Papers referenced:
- "Efficient Estimation of Word Representations in Vector Space" (Mikolov et al., 2013) — introduces CBOW and Skip-gram architectures
- "Distributed Representations of Words and Phrases and their Compositionality" (Mikolov et al., 2013) — introduces Negative Sampling and Subsampling of frequent words, which make Skip-gram trainable at scale

## 1. Central Claim

Prior neural language models (e.g. feedforward NNLM, RNNLM) learned word vectors as a side effect of language modeling, but were computationally expensive because of a nonlinear hidden layer and a full softmax over the vocabulary. The paper's claim is that you don't need a deep nonlinear model to get high-quality word vectors — a much simpler, shallow log-linear model trained on a very large corpus produces embeddings that are both cheaper to train and capture richer linear semantic/syntactic structure than prior, more complex models. The most famous evidence for "richer structure" is that simple vector arithmetic on these embeddings recovers semantic relationships, e.g. vector("king") - vector("man") + vector("woman") ≈ vector("queen").

## 2. Core Architecture / Algorithm

Two architecture variants are proposed; I'm implementing **Skip-gram**, since it's the more widely used and benchmarked variant and works better on smaller datasets/rarer words (relevant given I'm training on a much smaller corpus than the original).

**Skip-gram model:**
- For each word in the corpus (the "center" word), the model tries to predict the words in a small window around it (the "context" words).
- Unlike CBOW (which averages context vectors to predict one center word), skip-gram goes the other direction: one center word → predict each nearby context word independently.
- Each word has two vector representations during training: an "input" embedding (used when the word is the center word) and an "output" embedding (used when the word is being predicted as a context word). Only the input embeddings are kept at the end as the final word vectors.
- Architecture is shallow and log-linear: input embedding lookup → dot product with output embeddings → probability distribution over vocabulary (in the naive form, via softmax).

**Why naive softmax doesn't scale:**
- Computing softmax over the full vocabulary (could be hundreds of thousands of words) for every single training example is far too slow at corpus sizes of billions of words.

**Negative Sampling (the key trick, from the companion paper):**
- Reframes prediction as binary classification instead of multi-class softmax.
- For each true (center, context) pair observed in the corpus, sample k "negative" context words at random from the vocabulary (weighted by a smoothed unigram frequency distribution, typically raised to the 3/4 power).
- Train a binary logistic classifier to output high probability for true pairs and low probability for the k negative (fake) pairs.
- This turns an O(vocab size) softmax computation into an O(k) computation per training example, making training on huge corpora feasible.

**Subsampling of frequent words (secondary trick):**
- Very frequent words (e.g. "the", "a", "is") provide less useful signal than rare words and are randomly downsampled during training based on their frequency, which both speeds up training and improves the quality of vectors for rarer words.

## 3. Dataset, Metric, Baseline

**Dataset (original paper):** Trained on a large Google News corpus (~6B words for some experiments, up to ~100B words in places), vocabulary restricted to the most frequent ~1M words.

**Metric:** Word analogy accuracy — a curated test set of analogy questions (e.g. "Athens : Greece :: Oslo : ?") spanning both semantic relationships (capital-country, currency) and syntactic relationships (plural forms, verb tense). For each question, the model computes vector(b) - vector(a) + vector(c) and checks via nearest-neighbor search (cosine similarity, excluding the input words themselves) whether the closest vector matches the expected answer word. Reported as overall % accuracy, broken down into semantic and syntactic subsets.

**Baseline:** Prior word vector approaches, primarily NNLM (Bengio et al.) and RNNLM (Mikolov et al.'s own earlier recurrent model), as well as simpler approaches like Latent Semantic Analysis (LSA). Skip-gram and CBOW are shown to dramatically outperform these on the analogy task while training much faster.

## My Scoped-Down Reproduction Plan

- **Dataset:** `text8` corpus (~17M words, first 100MB of cleaned English Wikipedia) instead of the full ~100B word Google News corpus — the standard small benchmark for word2vec reproductions.
- **Model:** Skip-gram with Negative Sampling, implemented from scratch in PyTorch (embedding layers + manual negative sampling loss, no use of `nn.Embedding`-based prebuilt word2vec libraries like `gensim`).
- **Metric:** Same word analogy task, using the publicly available Google analogy question set, reporting overall accuracy.
- **Expected outcome:** Given the much smaller corpus and vocabulary versus the original paper, I expect noticeably lower accuracy than the paper's reported numbers (which were in the 50-70%+ range depending on model/dimensionality), but the qualitative behavior (nearest neighbors making semantic sense, some analogies resolving correctly) should still be observable.
