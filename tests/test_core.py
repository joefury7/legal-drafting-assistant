"""Unit and Integration Tests for Milestones 1 through 5.
Verifies corpus preparation, n-gram modeling, GPT-2 clause tagging, CLI drafting, and evaluation.
"""

from app.classifier import LEGAL_CLAUSE_TYPES, LegalClauseClassifier
from app.core import milestone_1, milestone_2, milestone_3, milestone_4, milestone_5
from app.ngram import LegalNGramLM


# Milestone 1: Prepare a small legal text corpus for n-gram training.
def test_milestone_1():
    corpus_info = milestone_1()
    assert isinstance(corpus_info, dict)
    assert "text" in corpus_info
    assert "sentences" in corpus_info
    assert "num_sentences" in corpus_info
    assert "num_words" in corpus_info

    assert corpus_info["num_sentences"] >= 10
    assert corpus_info["num_words"] >= 100
    assert isinstance(corpus_info["text"], str)

    # Validate essential legal terms in corpus
    lower_text = corpus_info["text"].lower()
    for term in ["agreement", "laws", "notices", "counterparts"]:
        assert term in lower_text, f"Expected '{term}' in legal corpus"


# Milestone 2: Implement the n-gram next-word predictor.
def test_milestone_2():
    lm = milestone_2(n=3)
    assert isinstance(lm, LegalNGramLM)
    assert lm.n == 3
    assert len(lm.vocab) > 50

    # Test next-word suggestions
    suggestions = lm.suggest_next("this agreement shall be", k=5, method="kneser_ney")
    assert isinstance(suggestions, list)
    assert len(suggestions) > 0
    word, prob = suggestions[0]
    assert isinstance(word, str)
    assert 0.0 <= prob <= 1.0

    # Test Add-k suggestions
    addk_sugg = lm.suggest_next("this agreement shall be", k=3, method="add_k")
    assert len(addk_sugg) > 0

    # Test clause drafting from seed
    drafted = lm.draft_clause(seed_phrase="this agreement shall be governed", max_words=15)
    assert isinstance(drafted, str)
    assert drafted.lower().startswith("this agreement shall be governed")
    assert len(drafted.split()) >= 5

    # Test perplexity calculation
    test_phrase = "This Agreement shall be governed by the laws of New York."
    ppl = lm.perplexity(test_phrase, method="kneser_ney")
    assert ppl > 0.0 and ppl != float("inf")


# Milestone 3: Load the pre-trained GPT-2 model and fine-tuned adapter for legal clause tagging.
def test_milestone_3():
    classifier = milestone_3()
    assert isinstance(classifier, LegalClauseClassifier)
    assert classifier.classes == LEGAL_CLAUSE_TYPES
    assert len(classifier.label_ids) == 6

    # Test single clause classification
    sample_clause = "This Agreement shall be governed by and construed in accordance with the laws of Delaware."
    res = classifier.classify(sample_clause)

    assert "label" in res
    assert res["label"] in LEGAL_CLAUSE_TYPES
    assert "confidence" in res
    assert 0.0 <= res["confidence"] <= 1.0
    assert "distribution" in res
    assert len(res["distribution"]) == len(LEGAL_CLAUSE_TYPES)
    assert "margin" in res
    assert res["margin"] >= 0.0


# Milestone 4: Create a CLI interface to generate text and then classify the generated text.
def test_milestone_4(capsys):
    from app.cli import main

    # 1. Test core programmatic pipeline
    result = milestone_4(seed_phrase="all notices under this agreement", max_words=15)
    assert isinstance(result, dict)
    assert "seed" in result
    assert "drafted_clause" in result
    assert "classification" in result
    assert "confidence" in result
    assert result["classification"] in LEGAL_CLAUSE_TYPES

    # 2. Test CLI execution of draft command
    main(["draft", "--seed", "this agreement may be executed", "--words", "10", "--method", "kneser_ney"])
    captured = capsys.readouterr().out
    assert "DRAFTED LEGAL CLAUSE" in captured
    assert "PREDICTED CLAUSE TYPE" in captured

    # 3. Test CLI classify command
    main(["classify", "--clause", "Should any provision be invalid, remainder remains in effect."])
    captured_classify = capsys.readouterr().out
    assert "PREDICTED TYPE" in captured_classify


# Milestone 5: Evaluate classification accuracy on example clauses.
def test_milestone_5(capsys):
    from app.cli import main

    # 1. Test core evaluation function
    eval_clauses = [
        {"text": "This Agreement shall be governed by the laws of New York.", "expected_label": "Governing Laws"},
        {"text": "All notices hereunder shall be in writing.", "expected_label": "Notices"},
        {"text": "This Agreement may be executed in counterparts.", "expected_label": "Counterparts"},
    ]
    report = milestone_5(test_clauses=eval_clauses)
    assert isinstance(report, dict)
    assert "accuracy" in report
    assert "total" in report
    assert report["total"] == 3
    assert 0.0 <= report["accuracy"] <= 1.0
    assert "per_class_accuracy" in report

    # 2. Test CLI evaluate command
    main(["evaluate"])
    captured = capsys.readouterr().out
    assert "EVALUATION SUITE" in captured
    assert "Overall Accuracy" in captured
