"""Command-line entry point: `python -m app.cli --help`.
Full CLI for paralegals to draft clauses with N-gram models and classify with fine-tuned GPT-2.
"""

import argparse
import sys
from app.classifier import LegalClauseClassifier
from app.core import milestone_1, milestone_2, milestone_3, milestone_4, milestone_5
from app.ngram import LegalNGramLM


def format_bar(prob: float, length: int = 20) -> str:
    """Renders a simple ASCII probability bar."""
    filled = int(round(prob * length))
    return f"[{'#' * filled}{'-' * (length - filled)}] {prob * 100:5.1f}%"


def cmd_hello():
    print("Legal Drafting Assistant is ready. Open README.md for your milestones.")


def cmd_draft(args):
    print("=" * 75)
    print("  LEGAL DRAFTING ASSISTANT -- CLAUSE DRAFTER & TAGGER")
    print("=" * 75)
    print(f"Seed Prompt    : \"{args.seed}\"")
    print(f"Max Words      : {args.words}")
    print(f"Smoothing      : {args.method}")
    print(f"Temperature    : {args.temperature}")
    print("-" * 75)

    corpus_data = milestone_1()
    lm = LegalNGramLM(n=args.order)
    lm.fit(corpus_data["text"])
    classifier = milestone_3()

    result = milestone_4(
        seed_phrase=args.seed,
        max_words=args.words,
        lm=lm,
        classifier=classifier,
    )

    print("\n📝 DRAFTED LEGAL CLAUSE:")
    print(f"   {result['drafted_clause']}\n")
    print("-" * 75)
    print(f"🏷️  PREDICTED CLAUSE TYPE: {result['classification']}")
    print(f"📊 CONFIDENCE SCORE      : {result['confidence'] * 100:.1f}%")
    print(f"📈 CERTAINTY MARGIN      : {result['margin'] * 100:.1f}%")
    print("\nCandidate Probabilities:")
    for label, prob in sorted(result["distribution"].items(), key=lambda x: x[1], reverse=True):
        print(f"   {label:20s}: {format_bar(prob)}")
    print("=" * 75)


def cmd_classify(args):
    print("=" * 75)
    print("  LEGAL CLAUSE CLASSIFIER")
    print("=" * 75)
    print(f"Input Clause:\n\"{args.clause}\"\n")
    print("-" * 75)

    classifier = milestone_3()
    res = classifier.classify(args.clause)

    print(f"🏷️  PREDICTED TYPE : {res['label']}")
    print(f"📊 CONFIDENCE     : {res['confidence'] * 100:.1f}%")
    print(f"📈 MARGIN         : {res['margin'] * 100:.1f}%\n")
    print("Full Category Distribution:")
    for label, prob in sorted(res["distribution"].items(), key=lambda x: x[1], reverse=True):
        print(f"   {label:20s}: {format_bar(prob)}")
    print("=" * 75)


def cmd_suggest(args):
    corpus_data = milestone_1()
    lm = LegalNGramLM(n=args.order)
    lm.fit(corpus_data["text"])

    suggestions = lm.suggest_next(args.context, k=args.top, method=args.method)
    print(f"\nContext: \"{args.context}\"")
    print(f"Top-{args.top} Next Word Suggestions ({args.method}):")
    for i, (word, prob) in enumerate(suggestions, 1):
        print(f"  {i}. {word:16s} {format_bar(prob)}")


def cmd_evaluate(args):
    print("=" * 75)
    print("  EVALUATION SUITE: LEGAL CLAUSE CLASSIFIER")
    print("=" * 75)
    classifier = milestone_3()
    eval_res = milestone_5(classifier=classifier)

    print(f"Total Test Clauses : {eval_res['total']}")
    print(f"Correctly Tagged   : {eval_res['correct']}")
    print(f"Overall Accuracy   : {eval_res['accuracy'] * 100:.1f}%  (Baseline random: 16.7%)")
    print("\nPer-Class Breakdown:")
    for c, acc in eval_res["per_class_accuracy"].items():
        print(f"   {c:20s}: {acc * 100:5.1f}%")

    print("\nSample Evaluations:")
    for r in eval_res["results"][:6]:
        mark = "✓" if r["correct"] else "✗"
        print(f"  [{mark}] Expected: {r['expected']:18s} | Predicted: {r['predicted']:18s} ({r['confidence']*100:.1f}%)")
    print("=" * 75)


def cmd_interactive(args):
    print("=" * 75)
    print("  LEGAL DRAFTING ASSISTANT -- INTERACTIVE PARALEGAL CONSOLE")
    print("=" * 75)
    print("Commands: type text to expand, or 'exit' / 'quit' to end session.")
    print("Loading language model and classifier...")

    corpus_data = milestone_1()
    lm = LegalNGramLM(n=3)
    lm.fit(corpus_data["text"])
    classifier = milestone_3()

    current_text = args.seed or "This Agreement shall be governed by"

    while True:
        print("\n" + "-" * 75)
        print(f"Current Draft: \"{current_text}\"")

        # Classify current text if long enough
        if len(current_text.split()) >= 3:
            res = classifier.classify(current_text)
            print(f"Tag: {res['label']} ({res['confidence'] * 100:.1f}% conf)")

        # Suggestions
        suggestions = lm.suggest_next(current_text, k=4, method="kneser_ney")
        print("Suggestions: " + " | ".join(f"[{i+1}] {w} ({p*100:.0f}%)" for i, (w, p) in enumerate(suggestions)))

        try:
            user_input = input("Enter word / number [1-4] to append (or 'q' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in ("q", "quit", "exit"):
            break

        if user_input.isdigit() and 1 <= int(user_input) <= len(suggestions):
            chosen = suggestions[int(user_input) - 1][0]
            current_text += " " + chosen
        else:
            current_text += " " + user_input

    print("\nFinal Clause Draft:\n" + current_text)
    final_res = classifier.classify(current_text)
    print(f"\nFinal Clause Tag: {final_res['label']} ({final_res['confidence'] * 100:.1f}% confidence)")
    print("Session ended.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="legal-drafting-assistant",
        description="Paralegals get an assistant to draft legal sentences using n-gram prediction and instantly classify them with fine-tuned GPT-2.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Diagnostic healthcheck
    sub.add_parser("hello", help="verify assistant environment and wiring")

    # 2. draft
    p_draft = sub.add_parser("draft", help="generate a legal clause from seed and tag its type")
    p_draft.add_argument("--seed", default="This Agreement shall be governed by", help="starting legal seed phrase")
    p_draft.add_argument("--words", type=int, default=25, help="maximum words to generate")
    p_draft.add_argument("--order", type=int, default=3, help="n-gram order (2 for bigram, 3 for trigram)")
    p_draft.add_argument("--method", choices=["kneser_ney", "add_k", "mle"], default="kneser_ney", help="smoothing method")
    p_draft.add_argument("--temperature", type=float, default=0.7, help="sampling temperature (lower = more deterministic)")

    # 3. classify
    p_classify = sub.add_parser("classify", help="classify any legal clause into contract categories")
    p_classify.add_argument("--clause", required=True, help="contract clause text to classify")

    # 4. suggest
    p_suggest = sub.add_parser("suggest", help="suggest next legal words given context")
    p_suggest.add_argument("--context", required=True, help="preceding context words")
    p_suggest.add_argument("--top", type=int, default=5, help="number of suggestions to return")
    p_suggest.add_argument("--order", type=int, default=3, help="n-gram order")
    p_suggest.add_argument("--method", choices=["kneser_ney", "add_k", "mle"], default="kneser_ney", help="smoothing method")

    # 5. evaluate
    sub.add_parser("evaluate", help="evaluate classification performance across benchmark clauses")

    # 6. interactive
    p_interactive = sub.add_parser("interactive", help="launch interactive paralegal drafting terminal")
    p_interactive.add_argument("--seed", default="This Agreement shall be governed by", help="initial seed phrase")

    args = parser.parse_args(argv)

    if args.command == "hello":
        cmd_hello()
    elif args.command == "draft":
        cmd_draft(args)
    elif args.command == "classify":
        cmd_classify(args)
    elif args.command == "suggest":
        cmd_suggest(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "interactive":
        cmd_interactive(args)


if __name__ == "__main__":
    main()
