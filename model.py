"""
model.py — N-Gram Language Model
Handles corpus loading, bigram/trigram model building, and prediction.
"""

import nltk
from nltk.corpus import brown
from nltk.util import ngrams
from collections import defaultdict, Counter


def load_corpus():
    """
    Downloads and returns lowercased alphabetic tokens from the Brown corpus.
    Returns:
        list[str]: A list of cleaned lowercase word tokens.
    """
    nltk.download('brown', quiet=True)
    return [w.lower() for w in brown.words() if w.isalpha()]


def build_bigram_model(tokens: list[str]) -> defaultdict:
    """
    Builds a bigram frequency model from a list of tokens.
    Args:
        tokens: List of word tokens.
    Returns:
        defaultdict[str, Counter]: Maps each word to a Counter of following words.
    """
    model = defaultdict(Counter)
    for w1, w2 in ngrams(tokens, 2):
        model[w1][w2] += 1
    return model


def build_trigram_model(tokens: list[str]) -> defaultdict:
    """
    Builds a trigram frequency model from a list of tokens.
    Args:
        tokens: List of word tokens.
    Returns:
        defaultdict[tuple, Counter]: Maps (w1, w2) pairs to a Counter of following words.
    """
    model = defaultdict(Counter)
    for w1, w2, w3 in ngrams(tokens, 3):
        model[(w1, w2)][w3] += 1
    return model


def predict_next_words(
    bigram_model: defaultdict,
    trigram_model: defaultdict,
    text: str,
    top_n: int = 5
) -> list[tuple[str, float]]:
    """
    Predicts the next word(s) given input text.
    Tries trigram first, falls back to bigram.

    Args:
        bigram_model:  The bigram frequency model.
        trigram_model: The trigram frequency model.
        text:          The current user input string.
        top_n:         Number of suggestions to return.

    Returns:
        List of (word, confidence_pct) tuples sorted by frequency.
    """
    words = text.lower().split()
    if not words:
        return []

    # --- Try trigram first (smarter, uses last 2 words) ---
    if len(words) >= 2:
        key = (words[-2], words[-1])
        counter = trigram_model.get(key)
        if counter:
            return _rank(counter, top_n)

    # --- Fallback: bigram (uses last 1 word) ---
    last_word = words[-1]
    counter = bigram_model.get(last_word)
    if counter:
        return _rank(counter, top_n)

    return []


def predict_with_prefix(
    bigram_model: defaultdict,
    trigram_model: defaultdict,
    text: str,
    top_n: int = 5
) -> list[tuple[str, float]]:
    """
    Predicts completions, filtering by prefix if the user is mid-word.

    E.g., if text = "he sa", it completes "sa..." using bigram on "he".
    If text = "he said", it predicts the next word after "said".

    Args:
        bigram_model:  The bigram model.
        trigram_model: The trigram model.
        text:          Raw user input.
        top_n:         Max suggestions to return.

    Returns:
        List of (suggested_completion, confidence_pct) tuples.
    """
    words = text.lower().split()
    if not words:
        return []

    # If text ends with a space → predict next full word
    if text.endswith(" "):
        return predict_next_words(bigram_model, trigram_model, text.strip(), top_n)

    # Otherwise → user is mid-word, filter suggestions by prefix
    prefix = words[-1]
    context = " ".join(words[:-1]) if len(words) > 1 else ""

    candidates = predict_next_words(
        bigram_model, trigram_model,
        context if context else prefix,
        top_n=30
    )
    filtered = [(w, conf) for w, conf in candidates if w.startswith(prefix)]
    return filtered[:top_n]


# ---- Internal helper ----

def _rank(counter: Counter, top_n: int) -> list[tuple[str, float]]:
    """Converts a Counter into a ranked list with confidence percentages."""
    total = sum(counter.values())
    return [
        (word, round(count / total * 100, 1))
        for word, count in counter.most_common(top_n)
    ]
