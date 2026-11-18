"""Advanced N-Gram Language Model with Kneser-Ney Smoothing and Prefix Drafting.
Supports N=2, N=3, Kneser-Ney, Laplace/Add-k, MLE, and interactive word suggestions.
"""

from collections import Counter, defaultdict
import math
import random
import re


def to_sentences(text: str) -> list[str]:
    """Split text into sentences using standard punctuation."""
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def to_tokens(sentence: str) -> list[str]:
    """Extract normalized lowercase tokens from a sentence."""
    return re.findall(r"[a-z0-9']+", sentence.lower())


class LegalNGramLM:
    """N-Gram Language Model tailored for legal contract provisions.

    Features:
    - Configurable order N (default n=3 for trigram or n=2 for bigram)
    - Kneser-Ney smoothing (interpolated with absolute discounting)
    - Laplace / Add-k smoothing
    - Prefix-conditioned autoregressive clause generation
    - Next-word candidate ranking with probabilities
    - Cross-entropy perplexity scoring
    """

    def __init__(self, n: int = 3, discount: float = 0.75, add_k: float = 0.5):
        self.n = n
        self.discount = discount
        self.add_k = add_k
        self.vocab: set[str] = set()
        self.foll: dict[tuple[str, ...], Counter] = defaultdict(Counter)
        self.context_counts: Counter = Counter()
        self.unigram_counts: Counter = Counter()
        self.word_continuations: Counter = Counter()  # |{v : C(v, w) > 0}|
        self.total_bigram_types: int = 0
        self.sentences: list[str] = []

    def fit(self, text: str) -> "LegalNGramLM":
        """Trains n-gram counts and Kneser-Ney continuation statistics on text corpus."""
        self.sentences = to_sentences(text)
        self.foll.clear()
        self.context_counts.clear()
        self.unigram_counts.clear()
        self.word_continuations.clear()
        self.vocab.clear()

        # Track bigram predecessor counts for continuation probability
        bigram_seen = set()

        for s in self.sentences:
            tokens = to_tokens(s)
            if not tokens:
                continue

            for t in tokens:
                self.vocab.add(t)
                self.unigram_counts[t] += 1

            # Pad sentence with start and end tokens
            padded = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
            for tok in padded:
                self.vocab.add(tok)

            # Accumulate n-gram counts
            for i in range(len(padded) - self.n + 1):
                ctx = tuple(padded[i : i + self.n - 1])
                target = padded[i + self.n - 1]
                self.foll[ctx][target] += 1
                self.context_counts[ctx] += 1

                # Accumulate for bigram continuation
                if len(padded) >= 2 and i < len(padded) - 1:
                    w1, w2 = padded[i], padded[i + 1]
                    if (w1, w2) not in bigram_seen:
                        bigram_seen.add((w1, w2))
                        self.word_continuations[w2] += 1

        self.total_bigram_types = max(1, len(bigram_seen))
        return self

    def continuation_prob(self, word: str) -> float:
        """P_continuation(w) = |{v : C(v, w) > 0}| / sum_w' |{v : C(v, w') > 0}|"""
        return self.word_continuations[word] / self.total_bigram_types

    def prob_kneser_ney(self, context: tuple[str, ...], word: str) -> float:
        """Interpolated Kneser-Ney probability for P(word | context)."""
        ctx_count = self.context_counts.get(context, 0)
        p_cont = self.continuation_prob(word)

        if ctx_count == 0:
            return p_cont if p_cont > 0 else 1.0 / max(1, len(self.vocab))

        c = self.foll[context]
        word_count = c.get(word, 0)

        # Higher-order discounted term
        discounted = max(word_count - self.discount, 0.0) / ctx_count

        # Interpolation weight lambda
        n_unique_following = len(c)
        lambda_ctx = (self.discount * n_unique_following) / ctx_count

        return discounted + (lambda_ctx * p_cont)

    def prob_addk(self, context: tuple[str, ...], word: str) -> float:
        """Laplace / Add-k smoothed probability."""
        c = self.foll.get(context, Counter())
        total = sum(c.values())
        return (c[word] + self.add_k) / (total + self.add_k * max(1, len(self.vocab)))

    def prob_mle(self, context: tuple[str, ...], word: str) -> float:
        """Maximum Likelihood Estimation probability."""
        c = self.foll.get(context, Counter())
        total = sum(c.values())
        return c[word] / total if total > 0 else 0.0

    def suggest_next(
        self, context_str: str, k: int = 5, method: str = "kneser_ney"
    ) -> list[tuple[str, float]]:
        """Returns top-k candidate next words with probabilities given context string."""
        tokens = to_tokens(context_str)
        needed = self.n - 1
        if len(tokens) < needed:
            ctx = tuple(["<s>"] * (needed - len(tokens)) + tokens)
        else:
            ctx = tuple(tokens[-needed:])

        candidates = list(self.foll.get(ctx, Counter()).keys())
        if not candidates:
            # Fallback to most frequent legal words
            candidates = [w for w, _ in self.unigram_counts.most_common(20) if w != "<s>"]

        scored = []
        for w in candidates:
            if w in ("<s>", "</s>"):
                continue
            if method == "kneser_ney":
                p = self.prob_kneser_ney(ctx, w)
            elif method == "add_k":
                p = self.prob_addk(ctx, w)
            else:
                p = self.prob_mle(ctx, w)
            scored.append((w, p))

        scored.sort(key=lambda x: x[1], reverse=True)
        # Normalize top-k probabilities to sum to 1 for intuitive display
        top = scored[:k]
        s = sum(p for _, p in top) or 1.0
        return [(w, round(p / s, 4)) for w, p in top]

    def draft_clause(
        self,
        seed_phrase: str = "this agreement shall be governed",
        max_words: int = 25,
        temperature: float = 0.7,
        seed: int | None = 42,
    ) -> str:
        """Autoregressively drafts a legal clause starting from a seed phrase."""
        rng = random.Random(seed)
        seed_tokens = to_tokens(seed_phrase)
        if not seed_tokens:
            seed_tokens = ["this", "agreement"]

        needed = self.n - 1
        output_tokens = list(seed_tokens)

        for _ in range(max_words):
            current_context = tuple(
                (["<s>"] * (needed - len(output_tokens)) + output_tokens)[-needed:]
            )
            c = self.foll.get(current_context)
            if not c:
                # Fallback to bigram/unigram
                fallback_words = [w for w, _ in self.unigram_counts.most_common(10) if w not in ("<s>", "</s>")]
                if not fallback_words:
                    break
                nxt = rng.choice(fallback_words)
            else:
                words, counts = zip(*c.items())
                # Filter out start token
                valid_pairs = [(w, cnt) for w, cnt in zip(words, counts) if w != "<s>"]
                if not valid_pairs:
                    break
                words, counts = zip(*valid_pairs)

                if temperature <= 0.05:
                    # Greedy
                    nxt = words[counts.index(max(counts))]
                else:
                    # Temperature scaling
                    weights = [math.pow(cnt, 1.0 / max(0.1, temperature)) for cnt in counts]
                    nxt = rng.choices(words, weights=weights)[0]

            if nxt == "</s>":
                break
            output_tokens.append(nxt)

        clause = " ".join(output_tokens)
        # Capitalize first letter and add period
        if clause:
            clause = clause[0].upper() + clause[1:]
            if not clause.endswith("."):
                clause += "."
        return clause

    def perplexity(self, test_text: str, method: str = "kneser_ney") -> float:
        """Calculates cross-entropy perplexity on unseen test text."""
        test_sents = to_sentences(test_text)
        log_prob_sum = 0.0
        token_count = 0

        for s in test_sents:
            toks = to_tokens(s)
            if not toks:
                continue
            padded = ["<s>"] * (self.n - 1) + toks + ["</s>"]
            for i in range(len(padded) - self.n + 1):
                ctx = tuple(padded[i : i + self.n - 1])
                target = padded[i + self.n - 1]
                if method == "kneser_ney":
                    p = self.prob_kneser_ney(ctx, target)
                else:
                    p = self.prob_addk(ctx, target)
                log_prob_sum += math.log(max(p, 1e-12))
                token_count += 1

        if token_count == 0:
            return float("inf")
        return math.exp(-log_prob_sum / token_count)
