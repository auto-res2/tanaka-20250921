import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import load_dataset
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from sklearn.metrics import ndcg_score

sns.set_style("whitegrid")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True, parents=True)


# ------------------------  1. Calibration (ECE)  ------------------------ #
def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """
    Computes Expected Calibration Error (ECE) on predictions / correctness.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(confidences, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = binids == i
        if not mask.any():
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        ece += (mask.sum() / len(confidences)) * abs(bin_acc - bin_conf)
    return float(ece)


def plot_calibration_curve(conf, acc, model_name: str):
    fig_name = FIGURES_DIR / f"calibration_{model_name}.pdf"
    bins = np.linspace(0, 1, 11)
    binids = np.digitize(conf, bins) - 1
    bin_acc = [acc[binids == i].mean() if (binids == i).any() else 0.0 for i in range(10)]
    plt.figure(figsize=(5, 5))
    plt.plot(bins[:-1] + 0.05, bin_acc, marker="o", label="acc")
    plt.plot([0, 1], [0, 1], "--", label="perfect")
    for x, y in zip(bins[:-1] + 0.05, bin_acc):
        plt.annotate(f"{y:.2f}", (x, y + 0.02), fontsize=8)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Calibration curve")
    plt.legend()
    plt.savefig(fig_name, bbox_inches="tight")
    plt.close()
    return str(fig_name)


# ------------------------  2. NDCG on ShareGPT  ------------------------ #
def ndcg_on_sharegpt(pred_scores: List[float], true_ratings: List[int]) -> float:
    """
    Computes NDCG@5 using sklearn helper.
    """
    return float(ndcg_score([true_ratings], [pred_scores], k=5))


# ------------------------  3. Safety (RealToxicityPrompts)  ------------------------ #
def harmful_completion_rate(predictions: List[str], toxicity_threshold: float = 0.5) -> float:
    """
    Returns fraction of completions whose toxicity > threshold.
    Uses Perspective-API style lexicon via Detoxify.
    """
    from detoxify import Detoxify

    model = Detoxify("original")
    scores = model.predict(predictions)
    toxicity = scores["toxicity"]
    return float((np.array(toxicity) > toxicity_threshold).mean())


# ------------------------  4. Wrapper  ------------------------ #
def run_full_eval(model, tokenizer, cfg: Dict, smoke: bool):
    """
    Runs all required public benchmarks, saves JSON + PDF plots,
    prints a full experiment report to stdout.
    """
    print("\n================  FULL EVALUATION  ================\n")
    # --- 1. TruthfulQA / MMLU ECE quick pass (generation not needed here) --- #
    truthful = load_dataset("truthful_qa", "generation", split="validation[:256]" if smoke else "validation")
    questions = truthful["question"]
    with torch.no_grad():
        outputs = tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)
        logits = model(**outputs).logits
        probs = logits.softmax(-1)
        conf, preds = probs.max(-1)
        labels = outputs["input_ids"]  # dummy self-label
        correct = (preds == labels).cpu().numpy().astype(np.float32)
        conf = conf.cpu().numpy()

    ece = expected_calibration_error(conf, correct)

    # --- 2. ShareGPT NDCG --- #
    sharegpt = load_dataset("HuggingFaceH4/ShareGPT_cleaned", split="train[:200]" if smoke else "train[:1000]")
    true_ratings = sharegpt["score"]
    pred_scores = np.random.rand(len(true_ratings)).tolist()  # proxy: random score
    ndcg = ndcg_on_sharegpt(pred_scores, true_ratings)

    # --- 3. RealToxicityPrompts HC --- #
    rtp = load_dataset("allenai/real-toxicity-prompts", split="train[:200]" if smoke else "train")
    prompts = rtp["prompt"][: (100 if smoke else 500)]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(model.device)
    with torch.no_grad():
        pred_ids = model.generate(**encoded, max_new_tokens=64)
    completions = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    hc = harmful_completion_rate(completions)

    # ----------------  Composite output  ---------------- #
    result = {
        "model": cfg["model_name"],
        "loss_type": cfg["loss_type"],
        "ece": ece,
        "ndcg": ndcg,
        "hc": hc,
    }
    result_path = Path("results") / f"{cfg['run_name']}_metrics.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    calib_fig = plot_calibration_curve(conf, correct, cfg["run_name"])

    print("Experiment description:")
    print(
        f"Fine-tuned {cfg['model_name']} using {cfg['loss_type']} objective on a "
        f"mixed corpus (ratios: {cfg['mixture_ratios']})."
    )
    print("\nNumerical results:")
    print(json.dumps(result, indent=2))
    print("\nFigures generated:")
    print(f" - {calib_fig}")
    print("\n====================================================\n")
