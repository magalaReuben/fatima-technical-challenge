"""Build notebooks/02_test_a_likelihood.ipynb from the cell list below."""
import json
import sys

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.strip("\n")})


md(r"""
# 02 — Test A: likelihood on minimal pairs

**Question.** Does the model give a higher probability to the correct Luganda sentence than to the sentence with one wrong morpheme?

**Method (direct inspection, no prompt).** The model gives a probability to each token. The score of a sentence is the sum of `log P(token | previous tokens)`. This is the training loss with the sign changed. No training happens here.

For each pair, the notebook computes:
- the **total score** (sum of log-probabilities), and
- the **mean score per token** (total / number of tokens). The mean controls for sentence length.

A pair is **correct** if the correct sentence gets the higher score. Chance level is 50%.

The notebook also scores:
- the **30 paragraphs** (consistent version against switched version). This is the likelihood part of Test E.
- the **20 variation items**. Both forms are correct, so a good model gives them similar scores.

**Input.** The workbook `luganda_eval_items.xlsx` in your Drive folder. The notebook uses only rows with `STATUS` = `OK` or `FIXED`. For `FIXED` rows, it uses your text in the `fixed_*` columns.

**Model.** `Qwen/Qwen3.5-4B`, selected in notebook 01.
""")

md("## 1. Install packages")
code(r"""
%pip install -q "transformers==5.19.0" "huggingface_hub>=2.2" accelerate pandas openpyxl scipy torchvision pillow
""")

md(r"""
## 2. Settings

- `INCLUDE_TODO = True` also uses rows that you did not validate yet. Use it only for a test run of the code. The results of such a run are not valid, because the items can contain errors.
""")
code(r"""
MODEL_ID = "Qwen/Qwen3.5-4B"
DRIVE_ROOT = "/content/drive/MyDrive/fatima technical challenge"
ITEMS_PATH = f"{DRIVE_ROOT}/luganda_eval_items.xlsx"
USE_STATUS = {"OK", "FIXED"}
INCLUDE_TODO = False     # True = test run with unvalidated rows (results not valid)
RUN_NAME = "run1"
OUTPUT_DIR = f"{DRIVE_ROOT}/outputs/02_test_a/{RUN_NAME}"
SEED = 42
N_BOOT = 5000
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
## 4. Load the validated items

The cell reads the three sheets. For each row it uses the `fixed_*` text if you wrote one, and the original text if not. It also checks each minimal pair: if the two sentences differ in more than one word, it prints a warning, because a minimal pair must differ in one morpheme only.
""")
code(r"""
import pandas as pd

assert Path(ITEMS_PATH).exists(), f"Workbook not found: {ITEMS_PATH}"

def load_sheet(sheet, pairs):
    df = pd.read_excel(ITEMS_PATH, sheet_name=sheet, dtype=str).fillna("")
    df["STATUS"] = df["STATUS"].str.strip().str.upper()
    for col, fixed in pairs:
        df[col + "_final"] = [f.strip() if f.strip() else o.strip() for o, f in zip(df[col], df[fixed])]
    keep = USE_STATUS | ({"TODO"} if INCLUDE_TODO else set())
    print(f"{sheet}: {len(df)} rows | status counts: {df['STATUS'].value_counts().to_dict()} | used: {df['STATUS'].isin(keep).sum()}")
    return df[df["STATUS"].isin(keep)].reset_index(drop=True)

pairs_df = load_sheet("minimal_pairs", [("correct", "fixed_correct"), ("wrong", "fixed_wrong")])
para_df = load_sheet("paragraphs", [("consistent", "fixed_consistent"), ("switched", "fixed_switched")])
var_df = load_sheet("variation", [("form_a", "fixed_form_a"), ("form_b", "fixed_form_b")])

same = pairs_df["correct_final"] == pairs_df["wrong_final"]
if same.any():
    print("WARNING: these pairs have identical sentences and are removed:", pairs_df.loc[same, "id"].tolist())
    pairs_df = pairs_df[~same].reset_index(drop=True)

def n_word_diff(a, b):
    wa, wb = a.split(), b.split()
    if len(wa) != len(wb):
        return max(len(wa), len(wb))
    return sum(x != y for x, y in zip(wa, wb))

pairs_df["n_word_diff"] = [n_word_diff(a, b) for a, b in zip(pairs_df.correct_final, pairs_df.wrong_final)]
multi = pairs_df[pairs_df.n_word_diff != 1]
if len(multi):
    print(f"WARNING: {len(multi)} pairs differ in more than one word. Check them:")
    display(multi[["id", "correct_final", "wrong_final"]])

if INCLUDE_TODO:
    print("\nTEST RUN: unvalidated rows are included. Do not report these results.")
assert len(pairs_df) > 0, "No usable minimal pairs. Validate rows in the workbook (STATUS = OK or FIXED), or set INCLUDE_TODO = True for a test run."
print("\nUsable minimal pairs per error type:")
print(pairs_df.error_type.value_counts().to_string())
""")

md(r"""
## 5. Load the model and define the score

The score uses the raw sentence, without a chat template, because Test A inspects the model directly. Each sentence starts after one start token, which is not scored. The function also keeps the log-probability of each token, so that you can see where the model "notices" the error.
""")
code(r"""
import transformers as tf
from transformers import AutoTokenizer

MODEL_CLASSES = ["AutoModelForCausalLM", "AutoModelForMultimodalLM", "AutoModelForImageTextToText"]

def load_model(repo):
    tok = AutoTokenizer.from_pretrained(repo)
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

def start_token(tok):
    if tok.bos_token_id is not None:
        return tok.bos_token_id, tok.bos_token
    vocab = tok.get_vocab()
    for t in ("<|endoftext|>", "<|begin_of_text|>"):
        if t in vocab:
            return vocab[t], t
    return tok.eos_token_id, tok.eos_token

START_ID, START_TOK = start_token(tok)
print("start token:", repr(START_TOK))

@torch.no_grad()
def score(text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([[START_ID] + ids], device=model.device)
    logits = model(input_ids=input_ids).logits[0, :-1].float()
    lp = torch.log_softmax(logits, dim=-1).gather(1, input_ids[0, 1:, None]).squeeze(1).tolist()
    total = sum(lp)
    return {"sum": total, "mean": total / len(ids), "n_tokens": len(ids),
            "tokens": tok.convert_ids_to_tokens(ids), "token_logprobs": lp}

print(score("Ekitabo ekinene kiri ku mmeeza.")["sum"])
""")

md("## 6. Score the minimal pairs")
code(r"""
from tqdm.auto import tqdm

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

raw = []
for r in tqdm(pairs_df.itertuples(), total=len(pairs_df)):
    c, w = score(r.correct_final), score(r.wrong_final)
    raw.append({"id": r.id, "error_type": r.error_type, "status": r.STATUS,
                "correct": r.correct_final, "wrong": r.wrong_final,
                "sum_correct": c["sum"], "sum_wrong": w["sum"],
                "mean_correct": c["mean"], "mean_wrong": w["mean"],
                "ntok_correct": c["n_tokens"], "ntok_wrong": w["n_tokens"],
                "tokens_correct": c["tokens"], "tokens_wrong": w["tokens"],
                "lp_correct": c["token_logprobs"], "lp_wrong": w["token_logprobs"]})
write_jsonl(out / "raw_minimal_pairs.jsonl", raw)

res = pd.DataFrame(raw)
res["margin_sum"] = res.sum_correct - res.sum_wrong        # > 0: model prefers the correct sentence
res["margin_mean"] = res.mean_correct - res.mean_wrong
res["correct_by_sum"] = res.margin_sum > 0
res["correct_by_mean"] = res.margin_mean > 0
res["same_ntok"] = res.ntok_correct == res.ntok_wrong
cols = ["id", "error_type", "status", "correct", "wrong", "sum_correct", "sum_wrong", "margin_sum",
        "mean_correct", "mean_wrong", "margin_mean", "correct_by_sum", "correct_by_mean", "ntok_correct", "ntok_wrong"]
res[cols].to_csv(out / "test_a_pairs.csv", index=False)
res[cols].head()
""")

md(r"""
## 7. Accuracy by error type

- **acc_sum**: share of pairs where the correct sentence has the higher total score.
- **acc_mean**: the same with the mean score per token.
- **ci95**: 95% bootstrap interval of acc_sum.
- **p_binom**: exact binomial test against chance (50%), two-sided.
""")
code(r"""
from scipy.stats import binomtest

def boot_ci(x, n=N_BOOT, seed=SEED):
    rng = random.Random(seed)
    x = list(x)
    k = len(x)
    accs = sorted(sum(x[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return accs[int(0.025 * n)], accs[int(0.975 * n)]

def summarize(d):
    k = len(d)
    hits = int(d.correct_by_sum.sum())
    lo, hi = boot_ci(d.correct_by_sum.astype(int))
    return pd.Series({
        "n_pairs": k,
        "acc_sum": hits / k,
        "ci95_low": lo, "ci95_high": hi,
        "p_binom": binomtest(hits, k, 0.5).pvalue,
        "acc_mean": d.correct_by_mean.mean(),
        "median_margin_sum": d.margin_sum.median(),
        "share_same_ntok": d.same_ntok.mean(),
    })

table_a = res.groupby("error_type").apply(summarize)
table_a.loc["ALL"] = summarize(res)
table_a.to_csv(out / "test_a_accuracy.csv")
table_a.round(3)
""")
code(r"""
# Does the token length of the two sentences explain the result?
# If the wrong sentence has more tokens, its total score is usually lower for that reason alone.
res["len_group"] = pd.cut(res.ntok_wrong - res.ntok_correct, [-99, -1, 0, 99], labels=["wrong shorter", "same", "wrong longer"])
res.groupby("len_group", observed=True).agg(n=("id", "size"), acc_sum=("correct_by_sum", "mean"), acc_mean=("correct_by_mean", "mean")).round(3)
""")
code(r"""
# Errors: pairs where the model prefers the wrong sentence (largest negative margin first)
res.sort_values("margin_sum")[["id", "error_type", "correct", "wrong", "margin_sum", "margin_mean"]].head(15)
""")
code(r"""
# Where does the model notice the error? Token log-probabilities for one pair.
def show_pair(pid):
    r = res[res.id == pid].iloc[0]
    for label, toks, lps in [("correct", r.tokens_correct, r.lp_correct), ("wrong", r.tokens_wrong, r.lp_wrong)]:
        print(label.upper(), f"(sum {sum(lps):.2f})")
        print("  " + " | ".join(f"{t}:{lp:.1f}" for t, lp in zip(toks, lps)))

show_pair(res.sort_values("margin_sum").id.iloc[0])
""")

md(r"""
## 8. Paragraphs (Test E, likelihood part)

A paragraph is correct if the consistent version gets the higher score. The two versions differ only in the noun-class marker in the last sentence, so this tests agreement across sentences.
""")
code(r"""
para_rows = []
for r in tqdm(para_df.itertuples(), total=len(para_df)):
    c, s = score(r.consistent_final), score(r.switched_final)
    para_rows.append({"id": r.id, "status": r.STATUS, "referent": r.referent, "class_switch": r.class_switch,
                      "consistent": r.consistent_final, "switched": r.switched_final,
                      "sum_consistent": c["sum"], "sum_switched": s["sum"],
                      "mean_consistent": c["mean"], "mean_switched": s["mean"]})
para_res = pd.DataFrame(para_rows)
if len(para_res):
    para_res["margin_sum"] = para_res.sum_consistent - para_res.sum_switched
    para_res["correct_by_sum"] = para_res.margin_sum > 0
    para_res["correct_by_mean"] = (para_res.mean_consistent - para_res.mean_switched) > 0
    para_res.to_csv(out / "test_e_paragraph_likelihood.csv", index=False)
    k, hits = len(para_res), int(para_res.correct_by_sum.sum())
    print(f"Paragraphs: acc_sum {hits/k:.3f} ({hits}/{k}), acc_mean {para_res.correct_by_mean.mean():.3f}, "
          f"p_binom {binomtest(hits, k, 0.5).pvalue:.4f}")
else:
    print("No usable paragraphs yet.")
""")

md(r"""
## 9. Variation items

Both forms are correct. The cell reports the absolute score difference of each pair. If this difference is as large as the typical margin of the minimal pairs, the model treats correct variation like an error.
""")
code(r"""
var_rows = []
for r in tqdm(var_df.itertuples(), total=len(var_df)):
    a, b = score(r.form_a_final), score(r.form_b_final)
    var_rows.append({"id": r.id, "variation_type": r.variation_type, "form_a": r.form_a_final, "form_b": r.form_b_final,
                     "sum_a": a["sum"], "sum_b": b["sum"], "abs_diff_sum": abs(a["sum"] - b["sum"]),
                     "abs_diff_mean": abs(a["mean"] - b["mean"])})
var_res = pd.DataFrame(var_rows)
if len(var_res):
    ref = res.margin_sum.abs().median()
    var_res["larger_than_median_pair_margin"] = var_res.abs_diff_sum > ref
    var_res.to_csv(out / "variation_likelihood.csv", index=False)
    print(f"Median |margin| of minimal pairs: {ref:.2f}")
    print(f"Variation items with a larger difference: {var_res.larger_than_median_pair_margin.sum()}/{len(var_res)}")
    display(var_res[["id", "variation_type", "form_a", "form_b", "abs_diff_sum", "larger_than_median_pair_margin"]].round(2))
else:
    print("No usable variation items yet.")
""")

md(r"""
## 10. Summary

The cell writes `test_a_summary.md` with the main numbers. Test F uses `margin_sum` from `test_a_pairs.csv` as the second confidence signal.
""")
code(r"""
lines = [f"# Test A results ({MODEL_ID})", ""]
if INCLUDE_TODO:
    lines += ["**TEST RUN with unvalidated items. Do not report.**", ""]
lines += [f"Usable minimal pairs: {len(res)} (status OK or FIXED).", "",
          "| Error type | n | Accuracy (sum) | 95% CI | p (binomial) | Accuracy (mean per token) |",
          "|---|---|---|---|---|---|"]
for et, r in table_a.iterrows():
    lines.append(f"| {et} | {int(r.n_pairs)} | {r.acc_sum:.1%} | {r.ci95_low:.1%}–{r.ci95_high:.1%} | {r.p_binom:.4f} | {r.acc_mean:.1%} |")
if len(para_res):
    lines += ["", f"Paragraphs (Test E likelihood): {para_res.correct_by_sum.mean():.1%} of {len(para_res)} "
                  f"prefer the consistent version (mean per token: {para_res.correct_by_mean.mean():.1%})."]
if len(var_res):
    lines += [f"Variation items: {int(var_res.larger_than_median_pair_margin.sum())} of {len(var_res)} have a score "
              f"difference larger than the median minimal-pair margin."]
lines += ["", "Chance level is 50%."]
(out / "test_a_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
json.dump({"model": MODEL_ID, "gpu": GPU_NAME, "dtype": str(DTYPE), "transformers": transformers.__version__,
           "torch": torch.__version__, "start_token": START_TOK, "items_path": ITEMS_PATH,
           "include_todo": INCLUDE_TODO, "n_pairs": len(res), "n_paragraphs": len(para_res), "n_variation": len(var_res)},
          open(out / "run_info.json", "w"), indent=2)
print("\n".join(lines))
print("\nFiles:", sorted(p.name for p in out.iterdir()))
""")

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "L4", "name": "02_test_a_likelihood.ipynb"},
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
