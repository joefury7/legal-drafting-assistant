"""Legal Clause Classifier powered by fine-tuned DistilGPT-2.
Implements constrained next-token generation with confidence scoring and probability distributions.
"""

from collections import Counter
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

LEGAL_CLAUSE_TYPES = [
    "Governing Laws",
    "Notices",
    "Counterparts",
    "Entire Agreements",
    "Severability",
    "Survival",
]


class LegalClauseClassifier:
    """Classifies contract provisions using constrained next-token generation.

    Features:
    - Constrained decoding over target <Label> tokens
    - Softmax probability distributions across candidate legal categories
    - Confidence scoring and certainty margins
    - Automatic fallback for environments without saved checkpoints
    """

    def __init__(self, model_path: str | Path | None = None):
        self.classes = list(LEGAL_CLAUSE_TYPES)
        self.model_path = self._resolve_model_path(model_path)
        self._init_model_and_tokenizer()

    def _resolve_model_path(self, user_path: str | Path | None) -> Path | None:
        if user_path:
            p = Path(user_path)
            if p.exists() and (p / "config.json").exists():
                return p

        # Check candidate locations
        candidates = [
            Path("data/legalgpt-ft"),
            Path(__file__).parent.parent / "data" / "legalgpt-ft",
            Path(__file__).parent.parent.parent / "04-finetune-gpt2-legal" / "legalgpt-ft",
        ]
        for c in candidates:
            if c.exists() and (c / "config.json").exists():
                return c
        return None

    def _init_model_and_tokenizer(self):
        if self.model_path and self.model_path.exists():
            self.tok = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForCausalLM.from_pretrained(str(self.model_path)).eval()
        else:
            # Fallback to base distilgpt2 with custom added label tokens
            self.tok = AutoTokenizer.from_pretrained("distilgpt2")
            self.tok.pad_token = self.tok.eos_token
            label_tokens = [f"<{name}>" for name in self.classes]
            self.tok.add_tokens(label_tokens)
            self.model = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
            self.model.resize_token_embeddings(len(self.tok), mean_resizing=False)

        self.label_tokens = [f"<{name}>" for name in self.classes]
        self.label_ids = [self.tok.convert_tokens_to_ids(tok) for tok in self.label_tokens]
        self.id_to_name = {tid: name for tid, name in zip(self.label_ids, self.classes)}

    @torch.no_grad()
    def classify(self, clause: str) -> dict:
        """Classifies a clause using constrained next-token decoding.

        Returns:
            dict containing:
            - 'label': predicted clause type (str)
            - 'confidence': confidence probability (0.0 to 1.0)
            - 'distribution': dict mapping each category to its probability
            - 'margin': certainty margin between top-1 and top-2 classes
        """
        prompt = clause.strip() + " "
        inputs = self.tok(prompt, return_tensors="pt")
        outputs = self.model(**inputs)
        next_token_logits = outputs.logits[0, -1]

        # Restrict logits to only the allowed legal label tokens
        candidate_logits = torch.tensor([next_token_logits[tid].item() for tid in self.label_ids])
        probs = F.softmax(candidate_logits, dim=-1)

        best_idx = torch.argmax(probs).item()
        best_label = self.classes[best_idx]
        best_conf = probs[best_idx].item()

        # Distribution dictionary
        distribution = {
            self.classes[i]: round(probs[i].item(), 4) for i in range(len(self.classes))
        }

        # Calculate certainty margin (top 1 - top 2)
        sorted_probs = sorted(probs.tolist(), reverse=True)
        margin = sorted_probs[0] - (sorted_probs[1] if len(sorted_probs) > 1 else 0.0)

        return {
            "label": best_label,
            "confidence": round(best_conf, 4),
            "distribution": distribution,
            "margin": round(margin, 4),
        }

    def classify_batch(self, clauses: list[str]) -> list[dict]:
        """Classifies a list of legal clauses."""
        return [self.classify(c) for c in clauses]
