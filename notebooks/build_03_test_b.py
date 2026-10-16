"""Build notebooks/03_test_b_choice.ipynb from the cell list below."""
import json
import sys

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.strip("\n")})


md(r"""
# 03 — Test B: prompted choice (and Test F, part 1: confidence)

**Question.** When we show the model both sentences and ask "Which sentence is correct?", does it pick the correct one? Does it know when it is wrong?

**Method.**
- Each minimal pair is shown **two times**: once with the correct sentence as A, and once with the correct sentence as B. This removes position bias.
- The model answers `A` or `B` and gives a **confidence** from 0 to 100.
- The notebook also reads the model's probability for the letters `A` and `B` directly (the **forced-choice probability**). This is a second answer that does not depend on the text format.

**Metrics.**

| Metric | Formula |
|---|---|
| Accuracy | right answers ÷ all answers (both orders) |
| Position bias | answers "A" ÷ all valid answers (no bias = 50%) |
| Consistency | pairs right in both orders ÷ all pairs |
| Invalid rate | answers that the notebook cannot read ÷ all answers |
| Gap A − B | Test A accuracy − Test B accuracy, by error type |
| Calibration (Test F) | accuracy in each confidence group, ECE, Brier score |
| Overconfidence (Test F) | mean confidence of the wrong answers |
| AUROC (Test F) | how well a confidence signal separates right from wrong answers (0.5 = no information) |

**Input.** The same workbook as Test A (rows with `STATUS` = `OK` or `FIXED`). For the gap and for the margin signal, the notebook also reads the Test A results from Drive, if they exist.

**Model.** `Qwen/Qwen3.5-4B`, with thinking turned off.
""")

md("## 1. Install packages")
code(r"""
%pip install -q "transformers==5.19.0" "huggingface_hub>=2.2" accelerate pandas openpyxl scipy torchvision pillow
""")

md("## 2. Settings")
code(r"""
MODEL_ID = "Qwen/Qwen3.5-4B"
DRIVE_ROOT = "/content/drive/MyDrive/fatima technical challenge"
ITEMS_PATH = f"{DRIVE_ROOT}/luganda_eval_items.xlsx"
USE_STATUS = {"OK", "FIXED"}
INCLUDE_TODO = False     # True = test run with unvalidated rows (results not valid)
RUN_NAME = "run1"
OUTPUT_DIR = f"{DRIVE_ROOT}/outputs/03_test_b/{RUN_NAME}"
TEST_A_DIR = f"{DRIVE_ROOT}/outputs/02_test_a/run1"   # for the A − B gap and the margin signal
MAX_NEW_TOKENS = 40
SEED = 42
N_BOOT = 5000

# The instruction is in English, so that the model understands the task. Only the two sentences are Luganda.
SYSTEM_PROMPT = "You are an expert in Luganda grammar."
USER_TEMPLATE = (
    "Here are two Luganda sentences. One sentence is correct. The other sentence has one grammatical error.\n\n"
    "A: {a}\n"
    "B: {b}\n\n"
    "Which sentence is correct? Also give your confidence, from 0 to 100, that your answer is correct.\n"
    "Reply in exactly this format:\n"
    "Answer: <A or B>\n"
    "Confidence: <number from 0 to 100>"
)
""")

md("## 3. Mount Google Drive, log in to Hugging Face, and check the GPU")
code(r"""
import os, json, math, random, re
from pathlib import Path

from google.colab import drive
drive.mount("/content/drive")

import torch
import transformers
from huggingface_hub import login, whoami

try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except Exception as e:
    print("Colab secret HF_TOKEN not read:", e)
login(token=os.environ.get("HF_TOKEN"))
print("Logged in as:", whoami()["name"])

assert torch.cuda.is_available(), "No GPU. Select Runtime > Change runtime type > GPU."
GPU_NAME = torch.cuda.get_device_name(0)
DTYPE = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
print(f"GPU: {GPU_NAME} | dtype: {DTYPE} | transformers {transformers.__version__} | torch {torch.__version__}")

out = Path(OUTPUT_DIR)
out.mkdir(parents=True, exist_ok=True)
print("Outputs go to:", out)
""")

md(r"""
## 4. Load the validated minimal pairs

The same rules as Test A: the notebook uses `OK` and `FIXED` rows, and the `fixed_*` text when you wrote one.
""")
code(r"""
import pandas as pd

assert Path(ITEMS_PATH).exists(), f"Workbook not found: {ITEMS_PATH}"
df = pd.read_excel(ITEMS_PATH, sheet_name="minimal_pairs", dtype=str).fillna("")
df["STATUS"] = df["STATUS"].str.strip().str.upper()
for col, fixed in [("correct", "fixed_correct"), ("wrong", "fixed_wrong")]:
    df[col + "_final"] = [f.strip() if f.strip() else o.strip() for o, f in zip(df[col], df[fixed])]
keep = USE_STATUS | ({"TODO"} if INCLUDE_TODO else set())
print("Status counts:", df["STATUS"].value_counts().to_dict())
pairs_df = df[df["STATUS"].isin(keep) & (df.correct_final != df.wrong_final)].reset_index(drop=True)
assert len(pairs_df) > 0, "No usable minimal pairs. Run 00b_set_status_ok, or set INCLUDE_TODO = True for a test run."
if INCLUDE_TODO:
    print("TEST RUN: unvalidated rows are included. Do not report these results.")
print(f"Usable pairs: {len(pairs_df)}")
print(pairs_df.error_type.value_counts().to_string())

# Two trials for each pair: the correct sentence is A in order 1, and B in order 2.
trials = []
for r in pairs_df.itertuples():
    for order, (a, b, gold) in enumerate([(r.correct_final, r.wrong_final, "A"), (r.wrong_final, r.correct_final, "B")], start=1):
        trials.append({"id": r.id, "error_type": r.error_type, "order": order, "A": a, "B": b, "gold": gold})
print(f"Trials: {len(trials)}")
""")

md(r"""
## 5. Load the model

- `ask` sends the prompt and reads the text answer (greedy decoding, so the result is reproducible).
- `forced_choice` puts `Answer:` after the prompt and reads the probability of the next token: `A` or `B`. The two probabilities are normalized, so P(A) + P(B) = 1.
""")
code(r"""
import transformers as tf
from transformers import AutoTokenizer, AutoProcessor

MODEL_CLASSES = ["AutoModelForCausalLM", "AutoModelForMultimodalLM", "AutoModelForImageTextToText"]

def load_model(repo):
    tok = AutoTokenizer.from_pretrained(repo)
    if not getattr(tok, "chat_template", None):
        tok.chat_template = AutoProcessor.from_pretrained(repo).chat_template
    errors = []
    for cls_name in MODEL_CLASSES:
        cls = getattr(tf, cls_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(repo, dtype=DTYPE, device_map="cuda")
            print("loaded with", cls_name)
            return tok, model.eval()
        except Exception as e:
            errors.append(f"{cls_name}: {type(e).__name__}: {str(e)[:200]}")
    raise RuntimeError("Could not load " + repo + "\n" + "\n".join(errors))

tok, model = load_model(MODEL_ID)

def chat_text(user_text):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

THINK_RE = re.compile(r"<think>.*?</think>", flags=re.S)

@torch.no_grad()
def ask(prompt_text):
    enc = tok(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    out_ids = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                             temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    raw = tok.decode(out_ids[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return raw, THINK_RE.sub("", raw).strip()

def letter_ids(letter):
    ids = set()
    for variant in (letter, " " + letter):
        t = tok(variant, add_special_tokens=False)["input_ids"]
        if len(t) == 1:
            ids.add(t[0])
    assert ids, f"No single token for {letter!r}"
    return sorted(ids)

A_IDS, B_IDS = letter_ids("A"), letter_ids("B")
print("token ids  A:", A_IDS, " B:", B_IDS)

@torch.no_grad()
def forced_choice(prompt_text):
    enc = tok(prompt_text + "Answer:", return_tensors="pt", add_special_tokens=False).to(model.device)
    logp = torch.log_softmax(model(**enc).logits[0, -1].float(), dim=-1)
    la, lb = torch.logsumexp(logp[A_IDS], 0).item(), torch.logsumexp(logp[B_IDS], 0).item()
    pa = 1 / (1 + math.exp(lb - la))
    return pa, la, lb

ANS_RE = re.compile(r"answer\s*[:\-]?\s*\**\s*\(?([AB])\b", re.I)
CONF_RE = re.compile(r"confidence\s*[:\-]?\s*\**\s*(\d{1,3}(?:\.\d+)?)", re.I)

def parse(text):
    a, c = ANS_RE.search(text), CONF_RE.search(text)
    answer = a.group(1).upper() if a else None
    conf = min(100.0, max(0.0, float(c.group(1)))) if c else None
    return answer, conf

print(parse("Answer: B\nConfidence: 85"), parse("**Answer:** A, Confidence: 101"), parse("I think B."))
""")

md("## 6. Run all trials")
code(r"""
from tqdm.auto import tqdm

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

raw_rows = []
for t in tqdm(trials):
    prompt = chat_text(USER_TEMPLATE.format(a=t["A"], b=t["B"]))
    raw, text = ask(prompt)
    answer, conf = parse(text)
    pa, la, lb = forced_choice(prompt)
    raw_rows.append({**t, "prompt": prompt, "raw_output": raw, "output": text,
                     "answer": answer, "confidence": conf,
                     "p_A": pa, "logp_A": la, "logp_B": lb})
write_jsonl(out / "raw_trials.jsonl", raw_rows)

tb = pd.DataFrame(raw_rows)
tb["valid"] = tb.answer.notna()
tb["correct"] = tb.answer == tb.gold                    # an invalid answer counts as wrong
tb["fc_answer"] = (tb.p_A >= 0.5).map({True: "A", False: "B"})
tb["fc_correct"] = tb.fc_answer == tb.gold
tb["p_gold"] = tb.p_A.where(tb.gold == "A", 1 - tb.p_A)  # forced-choice probability of the correct letter
tb["fc_conf"] = tb.p_A.where(tb.fc_answer == "A", 1 - tb.p_A) * 100   # probability of the letter it chose
tb.drop(columns=["prompt", "raw_output"]).to_csv(out / "test_b_trials.csv", index=False)
print(f"Valid answers: {tb.valid.mean():.1%} | with a confidence value: {tb.confidence.notna().mean():.1%}")
tb[["id", "order", "gold", "output", "answer", "confidence", "p_A"]].head(6)
""")

md(r"""
## 7. Test B results by error type

- **acc**: accuracy of the text answers. **acc_fc**: accuracy of the forced-choice probability.
- **pct_A**: share of valid answers that are "A". 50% = no position bias.
- **consistent**: share of pairs that are right in both orders.
""")
code(r"""
from scipy.stats import binomtest

def boot_ci(x, n=N_BOOT, seed=SEED):
    rng = random.Random(seed)
    x = list(x)
    k = len(x)
    accs = sorted(sum(x[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return accs[int(0.025 * n)], accs[int(0.975 * n)]

per_pair = tb.groupby(["id", "error_type"]).agg(n_right=("correct", "sum")).reset_index()

def summarize(d):
    k, hits = len(d), int(d.correct.sum())
    lo, hi = boot_ci(d.correct.astype(int))
    v = d[d.valid]
    pp = per_pair[per_pair.id.isin(d.id)]
    return pd.Series({
        "n_trials": k, "acc": hits / k, "ci95_low": lo, "ci95_high": hi,
        "p_binom": binomtest(hits, k, 0.5).pvalue,
        "acc_fc": d.fc_correct.mean(),
        "pct_A": (v.answer == "A").mean() if len(v) else float("nan"),
        "consistent": (pp.n_right == 2).mean(),
        "invalid": 1 - d.valid.mean(),
    })

table_b = tb.groupby("error_type").apply(summarize)
table_b.loc["ALL"] = summarize(tb)
table_b.to_csv(out / "test_b_accuracy.csv")
table_b.round(3)
""")
code(r"""
# Accuracy for each order. A large difference between the orders shows position bias.
tb.pivot_table(index="error_type", columns="order", values="correct", aggfunc="mean").rename(
    columns={1: "order 1 (correct = A)", 2: "order 2 (correct = B)"}).round(3)
""")

md(r"""
## 8. Gap A − B

The notebook reads the Test A accuracy from `TEST_A_DIR`. Then it puts each error type in one row of the plan's results table. "High" means significantly better than chance (binomial test, p < 0.05).

| A | B | Likely cause | Direction (Question 3) |
|---|---|---|---|
| high | low | The knowledge exists, but the prompt does not reach it. | Instruction tuning with native-speaker judgments |
| low | low | The rule is not learned. | Minimal-pair data, morpheme-aware tokenization |
| high | high | The model knows and uses the rule. | — |
| low | high | Unusual: check the items and the prompt. | — |
""")
code(r"""
a_path = Path(TEST_A_DIR) / "test_a_accuracy.csv"
gap = None
if a_path.exists():
    ta = pd.read_csv(a_path, index_col=0)
    gap = pd.DataFrame({"acc_A": ta.acc_sum, "p_A": ta.p_binom, "acc_B": table_b.acc, "p_B": table_b.p_binom}).dropna()
    gap["gap_A_minus_B"] = gap.acc_A - gap.acc_B
    def label(r):
        a = "high" if (r.acc_A > 0.5 and r.p_A < 0.05) else "low"
        b = "high" if (r.acc_B > 0.5 and r.p_B < 0.05) else "low"
        return f"A {a}, B {b}"
    gap["pattern"] = gap.apply(label, axis=1)
    gap.to_csv(out / "gap_a_minus_b.csv")
    display(gap.round(3))
else:
    print("Test A results not found:", a_path, "\nRun notebook 02 first, then run this cell again.")
""")

md(r"""
## 9. Test F, part 1: confidence

Three confidence signals for each trial:
1. **stated**: the confidence number in the text answer (0–100).
2. **forced choice**: the probability of the letter that the model chose (50–100).
3. **margin**: the Test A margin of the same pair (correct score − wrong score). A larger margin should mean that the model is more likely to be right.

For each signal, the notebook computes **AUROC**: pick one right answer and one wrong answer at random. AUROC is the chance that the right answer has the higher signal. 0.5 = no information, 1.0 = perfect.

For the stated confidence, it also computes the **calibration table**, **ECE**, **Brier score** and **mean confidence on wrong answers**.
""")
code(r"""
def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))

BINS = [0, 20, 40, 60, 80, 100.0001]

def calibration(conf, correct):
    d = pd.DataFrame({"conf": conf, "correct": correct.astype(float)}).dropna()
    d["group"] = pd.cut(d.conf, BINS, right=False, labels=["0-20", "20-40", "40-60", "60-80", "80-100"])
    t = d.groupby("group", observed=False).agg(n=("conf", "size"), mean_conf=("conf", "mean"), accuracy=("correct", "mean"))
    t["mean_conf"] /= 100
    t["gap"] = (t.mean_conf - t.accuracy).abs()
    ece = (t.n / t.n.sum() * t.gap).sum()
    brier = ((d.conf / 100 - d.correct) ** 2).mean()
    return t, ece, brier, d

v = tb[tb.valid & tb.confidence.notna()]
cal, ece, brier, _ = calibration(v.confidence, v.correct)
cal.to_csv(out / "test_f_calibration_stated.csv")
print("Stated confidence: calibration table")
display(cal.round(3))

signals = {"stated": (v.confidence, v.correct), "forced choice": (tb.fc_conf, tb.fc_correct)}
m_path = Path(TEST_A_DIR) / "test_a_pairs.csv"
if m_path.exists():
    margins = pd.read_csv(m_path)[["id", "margin_sum"]]
    tbm = tb.merge(margins, on="id", how="inner")
    signals["Test A margin"] = (tbm.margin_sum, tbm.correct)
else:
    print("Test A margins not found; the margin signal is skipped.")

f_rows = []
for name, (s, c) in signals.items():
    f_rows.append({"signal": name, "n": len(s), "accuracy": c.mean(), "AUROC": auroc(list(s), list(c)),
                   "mean_signal_right": s[c].mean(), "mean_signal_wrong": s[~c].mean()})
test_f = pd.DataFrame(f_rows).set_index("signal")
test_f.to_csv(out / "test_f_signals.csv")
print(f"\nStated confidence: ECE = {ece:.3f} | Brier = {brier:.3f} | mean confidence: "
      f"right {v.confidence[v.correct].mean():.1f}, wrong {v.confidence[~v.correct].mean():.1f}")
display(test_f.round(3))
""")

md("## 10. Summary")
code(r"""
lines = [f"# Test B results ({MODEL_ID})", ""]
if INCLUDE_TODO:
    lines += ["**TEST RUN with unvalidated items. Do not report.**", ""]
lines += [f"Pairs: {len(pairs_df)} | trials (two orders): {len(tb)} | valid answers: {tb.valid.mean():.1%}", "",
          "| Error type | Accuracy | 95% CI | p (binomial) | Forced-choice accuracy | % answers A | Consistent pairs |",
          "|---|---|---|---|---|---|---|"]
for et, r in table_b.iterrows():
    lines.append(f"| {et} | {r.acc:.1%} | {r.ci95_low:.1%}–{r.ci95_high:.1%} | {r.p_binom:.4f} | {r.acc_fc:.1%} | {r.pct_A:.1%} | {r.consistent:.1%} |")
if gap is not None:
    lines += ["", "| Error type | Test A | Test B | Gap A − B | Pattern |", "|---|---|---|---|---|"]
    for et, r in gap.iterrows():
        lines.append(f"| {et} | {r.acc_A:.1%} | {r.acc_B:.1%} | {r.gap_A_minus_B:+.1%} | {r.pattern} |")
lines += ["", f"Test F (stated confidence): ECE {ece:.3f}, Brier {brier:.3f}.",
          "AUROC: " + ", ".join(f"{k} {r.AUROC:.2f}" for k, r in test_f.iterrows()), "", "Chance level is 50%."]
(out / "test_b_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
json.dump({"model": MODEL_ID, "gpu": GPU_NAME, "dtype": str(DTYPE), "transformers": transformers.__version__,
           "torch": torch.__version__, "system_prompt": SYSTEM_PROMPT, "user_template": USER_TEMPLATE,
           "include_todo": INCLUDE_TODO, "n_pairs": len(pairs_df), "n_trials": len(tb)},
          open(out / "run_info.json", "w"), indent=2)
print("\n".join(lines))
print("\nFiles:", sorted(p.name for p in out.iterdir()))
""")

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "L4", "name": "03_test_b_choice.ipynb"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
    lines = c["source"].split("\n")
    c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]

with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
print("wrote", sys.argv[1], len(cells), "cells")
