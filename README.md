# Legal Drafting Assistant ⚖️🤖

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/drv7ew4bnz/badge.svg)](https://ask.thili.ai/b/drv7ew4bnz)
![Tests](https://github.com/joefury7/legal-drafting-assistant/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A specialized AI drafting co-pilot for paralegals and legal teams. It fuses an **interpolated Kneser-Ney N-gram language model** for fast, deterministic clause drafting with a **fine-tuned DistilGPT-2 model** for instant, constrained legal clause classification with confidence scoring.

---

## 🎯 What You Are Building

* **Goal:** A functional legal drafting assistant that uses n-gram prediction to draft standard contractual provisions and immediately tags them into clause categories with fine-tuned GPT-2.
* **Domain:** Banking & Finance / LegalTech
* **Architecture:** Dual-engine architecture combining statistical subword/token sequence completion with autoregressive generative classification.

---

## 🗺️ Completed Milestones

- [x] **1. Prepare a small legal text corpus for n-gram training.**  
  *Curated and preprocessed 600+ real legal contract provisions (50k+ words, 1,400+ sentences) covering the 6 primary commercial contract clause types from the LEDGAR dataset.*
- [x] **2. Implement the n-gram next-word predictor.**  
  *Built an advanced N-gram language model from scratch supporting N=2/3, Maximum Likelihood Estimation, Add-k (Laplace) smoothing, and interpolated **Kneser-Ney smoothing** with absolute discounting.*
- [x] **3. Load the pre-trained GPT-2 model and fine-tuned adapter for legal clause tagging.**  
  *Loaded DistilGPT-2 with extended vocabulary (+6 atomic `<Label>` tokens), resized embedding layers, and single-step constrained decoding over legal label logits.*
- [x] **4. Create a CLI interface to generate text and then classify the generated text.**  
  *Developed a modular command-line interface with subcommands for drafting, classifying, live word suggestions, and an interactive paralegal console.*
- [x] **5. Evaluate classification accuracy on example clauses.**  
  *Benchmarked the classifier across held-out legal clauses, calculating per-class accuracy, probability distributions, certainty margins, and confusion patterns.*

---

## 🌟 Hackathon & Stretch Features

1. **Interpolated Kneser-Ney Smoothing:**  
   Implements absolute discounting ($d = 0.75$) and continuation probability ($P_{\text{continuation}}$) to generate fluent, contextually sound legal text even for rare n-gram contexts.
2. **Prefix-Conditioned Autoregressive Drafting:**  
   Enables paralegals to provide seed prefixes (e.g., `"This Agreement shall be governed by"`) and generate coherent contractual continuations with configurable sampling temperature.
3. **Softmax Confidence Scoring & Margin Metrics:**  
   Calculates normalized softmax probability distributions across all candidate legal classes, reporting top-1 confidence scores and certainty margins to flag ambiguous clauses.
4. **Interactive Paralegal Console (`interactive` mode):**  
   A live terminal assistant where paralegals can type clauses, see real-time numbered autocompletion suggestions, and watch the predicted clause tag update dynamically.
5. **Self-Contained Offline Fallback:**  
   Ships with an embedded curated corpus and graceful model loading so tests and CLI run reliably in CI pipelines and air-gapped environments.

---

## 🛠️ Quickstart & Installation

### 1. Setup Environment
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

### 2. Run the Test Suite
All 8 smoke and core milestone tests pass out-of-the-box:
```bash
pytest -v
```

---

## 💻 CLI Usage Guide

The assistant is accessible via `python -m app.cli <command>`:

### 1. Draft a Clause from a Seed Prompt
Draft an entire legal provision and immediately classify its category:
```bash
python -m app.cli draft --seed "This Agreement shall be governed by" --words 20 --method kneser_ney
```
*Output:*
```text
===========================================================================
  LEGAL DRAFTING ASSISTANT -- CLAUSE DRAFTER & TAGGER
===========================================================================
Seed Prompt    : "This Agreement shall be governed by"
Max Words      : 20
Smoothing      : kneser_ney
Temperature    : 0.7
---------------------------------------------------------------------------
📝 DRAFTED LEGAL CLAUSE:
   This agreement shall be governed by and construed in accordance with the laws of the state of new york.

---------------------------------------------------------------------------
🏷️  PREDICTED CLAUSE TYPE: Governing Laws
📊 CONFIDENCE SCORE      : 94.2%
📈 CERTAINTY MARGIN      : 88.5%

Candidate Probabilities:
   Governing Laws      : [####################]  94.2%
   Entire Agreements   : [--------------------]   2.4%
   Counterparts        : [--------------------]   1.5%
   Severability        : [--------------------]   0.9%
   Survival            : [--------------------]   0.6%
   Notices             : [--------------------]   0.4%
===========================================================================
```

### 2. Classify an Existing Contract Clause
Classify any contract clause with confidence metrics:
```bash
python -m app.cli classify --clause "All notices under this Agreement shall be in writing and sent by certified mail to the addresses set forth above."
```

### 3. Real-Time Next-Word Suggestions
Inspect candidate next words and completion probabilities:
```bash
python -m app.cli suggest --context "shall be governed by" --top 5
```

### 4. Interactive Paralegal Terminal
Launch the live interactive terminal drafting session:
```bash
python -m app.cli interactive --seed "This Agreement may be executed in"
```

### 5. Evaluate Performance
Run the evaluation suite over benchmark clauses:
```bash
python -m app.cli evaluate
```

---

## 📐 Project Structure

```text
legal-drafting-assistant/
├── .github/workflows/ci.yml       # Automated GitHub Actions test pipeline
├── .env.example                   # Configuration settings template
├── .gitignore                     # Git tracking exclusions
├── pytest.ini                     # Test runner configuration
├── requirements.txt               # Dependencies
├── README.md                      # Project documentation
├── app/
│   ├── __init__.py                # Package root
│   ├── cli.py                     # Rich CLI with draft, classify, suggest, evaluate, interactive
│   ├── core.py                    # Milestone 1-5 implementations
│   ├── ngram.py                   # N-gram LM with Kneser-Ney, Laplace, and prefix generation
│   ├── classifier.py              # Fine-tuned DistilGPT-2 clause tagger with confidence scoring
│   ├── default_corpus.py          # Embedded legal provisions for offline reproducibility
│   ├── llm.py                     # Multi-provider chat client (Ollama/Groq/Gemini)
│   └── storage.py                 # File-backed key-value persistence store
└── tests/
    ├── test_core.py               # Unit & integration tests for Milestones 1-5
    └── test_smoke.py              # Base plumbing tests
```

---

## 📄 License

This project is licensed under the MIT License.
