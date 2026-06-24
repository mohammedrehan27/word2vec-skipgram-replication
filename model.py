"""
Skip-gram with Negative Sampling (SGNS).

Each word gets two embeddings:
  - "input" embedding (used when word is the center/target word)
  - "output" embedding (used when word is a context word being predicted)

Training objective (Mikolov et al. 2013b, eq. 4), for one (center, true_context)
pair with k sampled negative context words:

    log sigmoid(v_out(context)  . v_in(center))
  + sum_{neg in negatives} log sigmoid(-v_out(neg) . v_in(center))

We maximize this (i.e. minimize its negative) instead of doing a full softmax
over the vocabulary.
"""

import torch
import torch.nn as nn


class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

        # paper-style initialization: small uniform range for input embeddings,
        # zero init for output embeddings is also common; we use a small
        # uniform init for both to keep training stable.
        init_range = 0.5 / embedding_dim
        nn.init.uniform_(self.in_embed.weight, -init_range, init_range)
        nn.init.uniform_(self.out_embed.weight, -init_range, init_range)

    def forward(self, center_words, pos_context_words, neg_context_words):
        """
        center_words:      (batch,)        long
        pos_context_words: (batch,)        long
        neg_context_words: (batch, k)      long

        Returns scalar loss (mean negative log-likelihood over the batch).
        """
        v_center = self.in_embed(center_words)              # (B, D)
        v_pos = self.out_embed(pos_context_words)            # (B, D)
        v_neg = self.out_embed(neg_context_words)            # (B, K, D)

        # positive term: dot(v_center, v_pos) per example
        pos_score = torch.sum(v_center * v_pos, dim=1)       # (B,)
        pos_loss = torch.nn.functional.logsigmoid(pos_score)  # (B,)

        # negative term: dot(v_center, v_neg_k) for each of K negatives
        neg_score = torch.bmm(v_neg, v_center.unsqueeze(2)).squeeze(2)  # (B, K)
        neg_loss = torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)  # (B,)

        loss = -(pos_loss + neg_loss).mean()
        return loss

    def get_embeddings(self):
        """Final word vectors = input embeddings (standard convention)."""
        return self.in_embed.weight.detach().cpu().numpy()


if __name__ == "__main__":
    # smoke test with random data
    vocab_size, dim, batch, k = 50, 16, 8, 5
    model = SkipGramNegSampling(vocab_size, dim)

    centers = torch.randint(0, vocab_size, (batch,))
    pos_ctx = torch.randint(0, vocab_size, (batch,))
    neg_ctx = torch.randint(0, vocab_size, (batch, k))

    loss = model(centers, pos_ctx, neg_ctx)
    print("loss:", loss.item())
    loss.backward()
    print("grad ok, in_embed grad norm:", model.in_embed.weight.grad.norm().item())
