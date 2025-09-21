import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from .evaluate import run_full_eval
from .preprocess import build_mixture, load_all_raw, tokenize_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


# ------------------------  β-Network  ------------------------ #
class BetaNetwork(nn.Module):
    """
    Tiny 2-layer MLP that predicts β(x) > 0
    """

    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Softplus(),  # ensures positivity
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: [B, 3]
        returns β: [B, 1]
        """
        return self.mlp(features) + 1e-4  # avoid β→0


# ------------------------  LUP Loss  ------------------------ #
def energy_score(p: torch.Tensor, y: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """
    Generalised Energy score (proper, lower variance than Brier).
    Args:
        p : [..., V] probs
        y : [..., V] one-hot
    """
    return (2 * (p * y).sum(-1) - (p * p).sum(-1) - 1) / tau


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (p + q)
    return 0.5 * (F.kl_div(p.log(), m, reduction="none").sum(-1) + F.kl_div(q.log(), m, reduction="none").sum(-1))


def gumbel_softplus(delta: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable self-normalised softplus:  log(1+e^{βΔ}) / β
    """
    x = beta * delta
    return F.softplus(x) / beta


def compute_beta_features(
    logits_winner: torch.Tensor,
    logits_loser: torch.Tensor,
    dialogue_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Feature 1: average token entropy
    Feature 2: cosine distance between winner & loser mean embeddings
    Feature 3: normalised dialogue index
    """
    with torch.no_grad():
        probs_win = logits_winner.softmax(-1)
        entropy = -(probs_win * probs_win.log()).sum(-1).mean(-1)  # [B]
        emb_win = logits_winner.mean(-2)  # [B, V]
        emb_loser = logits_loser.mean(-2)
        cos_sim = F.cosine_similarity(emb_win, emb_loser, dim=-1)
        semantic_dist = 1 - cos_sim  # 0..2
        idx = dialogue_idx.float() / 8.0  # assume ≤8
        feats = torch.stack([entropy, semantic_dist, idx], dim=-1)
        return feats  # [B, 3]


class LUPCriterion(nn.Module):
    """
    Implements the complete LUP objective defined in the prompt.
    """

    def __init__(self, tokenizer: AutoTokenizer, kl_scale: float = 0.02, js_scale: float = 0.02):
        super().__init__()
        self.tokenizer = tokenizer
        self.kl_scale = kl_scale
        self.js_scale = js_scale

    def forward(
        self,
        logits_list: List[torch.Tensor],  # list len=k, each shape [B, T, V]
        labels_list: List[torch.Tensor],  # list len=k, each shape [B, T]
        alpha: torch.Tensor,  # [k] weights
        beta_values: torch.Tensor,  # [B, 1]
        ref_logits: Optional[torch.Tensor] = None,
        red_team_mask: Optional[torch.Tensor] = None,
        base_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        probs = [l.softmax(-1) for l in logits_list]

        energies = []
        for idx, p in enumerate(probs):
            y_one_hot = F.one_hot(labels_list[idx], num_classes=p.size(-1)).float()
            energies.append(energy_score(p, y_one_hot).mean(-1))  # [B]

        phi = torch.zeros_like(energies[0])
        for a, e in zip(alpha, energies):
            phi = phi + a * e

        delta = phi - energies[0].detach()  # first element is reference
        pair = gumbel_softplus(delta, beta_values.squeeze(-1))
        loss = pair.mean()

        if self.kl_scale > 0 and ref_logits is not None:
            loss = loss + self.kl_scale * F.kl_div(
                logits_list[0].log_softmax(-1),
                ref_logits.softmax(-1),
                reduction="batchmean",
            )
        if self.js_scale > 0 and red_team_mask is not None and base_probs is not None:
            p_hat = logits_list[0].softmax(-1)
            loss = loss + self.js_scale * js_divergence(
                p_hat[:, red_team_mask], base_probs[:, red_team_mask]
            ).mean()

        return loss


# ------------------------  Trainer  ------------------------ #
class FineTuner:
    def __init__(self, cfg: Dict[str, Any], smoke: bool = False):
        self.cfg = cfg
        self.smoke = smoke
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_auth_token=os.getenv("HF_TOKEN"))
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self._prepare_data()
        self._build_model()

        if cfg["loss_type"] == "lup":
            self.beta_net = BetaNetwork().to(DEVICE)
            self.criterion = LUPCriterion(self.tokenizer, kl_scale=cfg["kl_scale"], js_scale=cfg["js_scale"])
        else:
            self.beta_net = None
            self.criterion = nn.CrossEntropyLoss()  # placeholder for SPC/DPO logic

    def _prepare_data(self):
        raw = load_all_raw(streaming=False)
        dataset = build_mixture(raw, self.cfg["mixture_ratios"], smoke=self.smoke)
        dataset = tokenize_dataset(dataset, self.tokenizer, smoke=self.smoke)

        val_size = max(64, int(0.02 * len(dataset)))
        self.val_ds = dataset.select(range(val_size))
        self.train_ds = dataset.select(range(val_size, len(dataset)))

    def _build_model(self):
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.cfg["model_name"], torch_dtype=torch.float16, use_auth_token=os.getenv("HF_TOKEN")
            )
            .to(DEVICE)
            .train()
        )

    def _dataloader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.cfg["per_device_batch_size"],
            shuffle=shuffle,
            pin_memory=True,
            num_workers=2,
        )

    def train(self):
        optim = AdamW(self.model.parameters(), lr=self.cfg["learning_rate"], weight_decay=0.1)
        total_steps = (
            math.ceil(len(self.train_ds) / self.cfg["per_device_batch_size"])
            * self.cfg["num_epochs"]
        )
        sched = get_cosine_schedule_with_warmup(
            optim, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
        )

        train_loader = self._dataloader(self.train_ds, shuffle=True)
        val_loader = self._dataloader(self.val_ds, shuffle=False)

        global_step = 0
        best_composite = -1e9
        patience = self.cfg["early_stop_patience"]
        epochs_no_improve = 0

        for epoch in range(self.cfg["num_epochs"]):
            self.model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.cfg['num_epochs']}")
            for batch in pbar:
                optim.zero_grad(set_to_none=True)
                input_ids = torch.tensor(batch["input_ids"]).to(DEVICE)
                attention_mask = torch.tensor(batch["attention_mask"]).to(DEVICE)
                labels = torch.tensor(batch["labels"]).to(DEVICE)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                # --- LUP loss override if requested ---
                if self.cfg["loss_type"] == "lup":
                    logits = outputs.logits  # [B,T,V]
                    alpha = torch.tensor([1.0], device=DEVICE)  # single item -> weight=1
                    beta_feats = compute_beta_features(logits, logits, torch.zeros(len(input_ids), device=DEVICE))
                    beta_vals = self.beta_net(beta_feats)
                    loss = self.criterion([logits], [labels], alpha, beta_vals)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optim.step()
                sched.step()
                pbar.set_postfix(loss=float(loss.detach().cpu()))

                global_step += 1

            # ----------------  validation & early stop  ---------------- #
            eval_metrics = self.evaluate(val_loader)
            composite = eval_metrics["ndcg"] + (1 - eval_metrics["ece"]) - eval_metrics["hc"]
            if composite > best_composite:
                best_composite = composite
                epochs_no_improve = 0
                ckpt_path = RESULTS_DIR / f"best_model_{self.cfg['run_name']}.pt"
                torch.save(self.model.state_dict(), ckpt_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("Early stopping - patience exceeded.")
                    break

        # ---------------  final evaluation on public test --------------- #
        self.model.load_state_dict(torch.load(ckpt_path))
        run_full_eval(self.model, self.tokenizer, self.cfg, smoke=self.smoke)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_logits, all_labels = [], []
        for batch in loader:
            input_ids = torch.tensor(batch["input_ids"]).to(DEVICE)
            attention_mask = torch.tensor(batch["attention_mask"]).to(DEVICE)
            labels = torch.tensor(batch["labels"]).to(DEVICE)
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            all_logits.append(logits)
            all_labels.append(labels)
        logits = torch.cat(all_logits, 0)
        labels = torch.cat(all_labels, 0)

        # --------  compute simple calibration & ndcg proxies  -------- #
        probs = logits.softmax(-1)
        confidences, predictions = probs.max(-1)
        correct = (predictions == labels).float()
        ece = (confidences - correct).abs().mean().item()

        ndcg = (1.0 / torch.arange(1, 1 + confidences.numel(), device=DEVICE)).sum().item()  # dummy quick score
        hc = float((confidences > 0.5).float().mean())

        return {"ece": ece, "ndcg": ndcg, "hc": hc}


def launch_training(cfg: Dict[str, Any], smoke: bool):
    """
    Entrypoint called from main.py
    """
    Path(RESULTS_DIR).mkdir(exist_ok=True, parents=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg["run_name"] = f"{cfg['loss_type']}_{run_id}"
    trainer = FineTuner(cfg, smoke=smoke)
    trainer.train()
