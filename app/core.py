"""Core engine implementations for all 5 Milestones of the Legal Drafting Assistant.
Builds the complete pipeline from corpus preparation to N-gram drafting and GPT-2 clause tagging.
"""

from pathlib import Path
from app.classifier import LEGAL_CLAUSE_TYPES, LegalClauseClassifier
from app.default_corpus import CURATED_LEGAL_CLAUSES, get_default_corpus_text
from app.ngram import LegalNGramLM, to_sentences, to_tokens


# --- Milestone 1: Prepare a specialized legal text corpus for n-gram modeling.
def milestone_1(corpus_path: str | Path | None = None) -> dict:
    """Prepares and validates the legal text corpus for N-gram modeling.

    Returns:
        dict containing 'text', 'sentences', 'num_sentences', and 'num_words'.
    """
    path = Path(corpus_path or "data/legal_corpus.txt")
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        text = get_default_corpus_text()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        except OSError:
            pass

    sentences = to_sentences(text)
    words = to_tokens(text)

    return {
        "text": text,
        "sentences": sentences,
        "num_sentences": len(sentences),
        "num_words": len(words),
    }


# --- Milestone 2: Implement the legal n-gram next-word predictor with Kneser-Ney smoothing.
def milestone_2(corpus_text: str | None = None, n: int = 3) -> LegalNGramLM:
    """Initializes and trains the legal N-gram language model with Kneser-Ney smoothing.

    Args:
        corpus_text: raw text corpus (defaults to milestone_1 corpus).
        n: order of n-gram model (default 3 for trigram).

    Returns:
        Fitted LegalNGramLM instance.
    """
    if corpus_text is None:
        data = milestone_1()
        corpus_text = data["text"]

    lm = LegalNGramLM(n=n, discount=0.75, add_k=0.5)
    lm.fit(corpus_text)
    return lm


# --- Milestone 3: Initialize the GPT-2 legal clause tagger with atomic classification tokens.
def milestone_3(model_path: str | Path | None = None) -> LegalClauseClassifier:
    """Loads the pre-trained/fine-tuned GPT-2 model with atomic legal label tokens.

    Args:
        model_path: optional path to saved fine-tuned model checkpoint.

    Returns:
        Initialized LegalClauseClassifier instance.
    """
    return LegalClauseClassifier(model_path=model_path)


# --- Milestone 4: Dual-engine pipeline combining n-gram drafting and GPT-2 clause tagging.
def milestone_4(
    seed_phrase: str = "this agreement shall be governed",
    max_words: int = 25,
    lm: LegalNGramLM | None = None,
    classifier: LegalClauseClassifier | None = None,
) -> dict:
    """Generates a legal sentence using the N-gram predictor and immediately classifies it with GPT-2.

    Returns:
        dict containing 'seed', 'drafted_clause', 'classification', 'confidence', and 'distribution'.
    """
    if lm is None:
        lm = milestone_2()
    if classifier is None:
        classifier = milestone_3()

    drafted_clause = lm.draft_clause(seed_phrase=seed_phrase, max_words=max_words, temperature=0.7)
    classification_result = classifier.classify(drafted_clause)

    return {
        "seed": seed_phrase,
        "drafted_clause": drafted_clause,
        "classification": classification_result["label"],
        "confidence": classification_result["confidence"],
        "distribution": classification_result["distribution"],
        "margin": classification_result["margin"],
    }


# --- Milestone 5: Empirical evaluation suite across benchmark contract clauses.
def milestone_5(
    classifier: LegalClauseClassifier | None = None,
    test_clauses: list[dict] | None = None,
) -> dict:
    """Evaluates the legal clause classifier across benchmark test clauses.

    Args:
        classifier: LegalClauseClassifier instance.
        test_clauses: optional list of dicts with 'text' and 'expected_label'.

    Returns:
        dict containing 'accuracy', 'total', 'correct', 'results', and 'per_class_accuracy'.
    """
    if classifier is None:
        classifier = milestone_3()

    if test_clauses is None:
        # Benchmark suite spanning the 6 primary categories
        test_clauses = [
            {"text": "This Agreement shall be governed by and construed in accordance with the laws of the State of New York.", "expected_label": "Governing Laws"},
            {"text": "All notices hereunder shall be in writing and sent by certified mail to the addresses set forth above.", "expected_label": "Notices"},
            {"text": "This Agreement may be executed in any number of counterparts, each of which shall be deemed an original.", "expected_label": "Counterparts"},
            {"text": "This Agreement constitutes the entire agreement between the parties with respect to the subject matter hereof.", "expected_label": "Entire Agreements"},
            {"text": "If any provision of this Agreement is held to be invalid or unenforceable, the remaining provisions shall continue in full force.", "expected_label": "Severability"},
            {"text": "The representations, warranties, covenants and indemnification obligations contained herein shall survive the closing.", "expected_label": "Survival"},
            {"text": "This contract shall be construed in accordance with the domestic laws of the State of Delaware.", "expected_label": "Governing Laws"},
            {"text": "Any notice or other communication required to be given shall be deemed given when delivered in person.", "expected_label": "Notices"},
            {"text": "This instrument may be signed in several counterparts, each of which will constitute an original.", "expected_label": "Counterparts"},
            {"text": "This document embodies the complete and final understanding between both contracting parties.", "expected_label": "Entire Agreements"},
            {"text": "Should any clause of this contract be declared unlawful, all other provisions shall remain valid and binding.", "expected_label": "Severability"},
            {"text": "All obligations of confidentiality and non-disclosure shall survive termination of this contract.", "expected_label": "Survival"},
        ]

    total = len(test_clauses)
    correct = 0
    results = []
    class_totals = {c: 0 for c in LEGAL_CLAUSE_TYPES}
    class_correct = {c: 0 for c in LEGAL_CLAUSE_TYPES}

    for item in test_clauses:
        text = item["text"]
        expected = item.get("expected_label") or item.get("label")
        prediction = classifier.classify(text)
        predicted_label = prediction["label"]

        is_match = predicted_label.lower().replace(" ", "_") == expected.lower().replace(" ", "_") or predicted_label == expected
        if is_match:
            correct += 1

        if expected in class_totals:
            class_totals[expected] += 1
            if is_match:
                class_correct[expected] += 1

        results.append({
            "text": text[:60] + "...",
            "expected": expected,
            "predicted": predicted_label,
            "confidence": prediction["confidence"],
            "correct": is_match,
        })

    accuracy = correct / total if total > 0 else 0.0

    per_class_accuracy = {}
    for c in LEGAL_CLAUSE_TYPES:
        if class_totals[c] > 0:
            per_class_accuracy[c] = round(class_correct[c] / class_totals[c], 4)

    return {
        "accuracy": round(accuracy, 4),
        "total": total,
        "correct": correct,
        "results": results,
        "per_class_accuracy": per_class_accuracy,
    }
