# Legal Drafting Assistant

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/drv7ew4bnz/badge.svg)](https://ask.thili.ai/b/drv7ew4bnz)

This project develops a legal drafting assistant for paralegals. It leverages n-gram prediction to suggest simple legal sentences and then uses a fine-tuned GPT-2 model to instantly classify these sentences into specific clause types, streamlining the drafting process.

## What you are building

**Goal.** You will have built a legal drafting assistant that uses n-gram prediction to draft simple legal sentences and instantly classifies them into clause types with a fine-tuned GPT-2 model.

**Done when.** You will have a functional command-line interface that can generate legal sentences using your n-gram predictor and then classify those generated sentences into clause types using the fine-tuned GPT-2 model. All five milestones will be complete, including an evaluation of the classification accuracy.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Prepare a small legal text corpus for n-gram training.**  
      Hint: Corpus preparation from the 'Build a Next-Word Predictor from Scratch' drill.
- [ ] **2. Implement the n-gram next-word predictor.**  
      Hint: Implement the n-gram logic from the 'Build a Next-Word Predictor from Scratch' drill.
- [ ] **3. Load the pre-trained GPT-2 model and fine-tuned adapter for legal clause tagging.**  
      Hint: Load the model and adapter as shown in the 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' drill.
- [ ] **4. Create a CLI interface to generate text and then classify the generated text.**  
      Hint: Combine the n-gram generation and GPT-2 classification into a command-line interface.
- [ ] **5. Evaluate classification accuracy on example clauses.**  
      Hint: Apply evaluation techniques from the 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' drill.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Enhance the n-gram predictor with techniques like Kneser-Ney smoothing for better word suggestions.
- Expand the CLI to allow users to specify a starting word or phrase for text generation.
- Implement a feature to display the confidence score for the GPT-2 clause classification.
