"""Build notebooks/01_model_selection.ipynb from the cell list below."""
import json
import sys

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.strip("\n")})


md(r"""
# 01 — Model selection for the Luganda evaluation

This notebook selects **one** small open model (1B–4B parameters) for the next evaluations (Tests A–F).

**Method.** The notebook takes **30 rows** from the `test` split of `reuben256/alpaca-luganda-cleaned`. Each row has a Luganda instruction and a Luganda reference answer. Each candidate model gets the same instruction and writes an answer. The notebook then measures how closely each answer matches the reference answer.

**Metrics.**

| Metric | What it measures | Better |
|---|---|---|
| **chrF++** (main metric) | Character and word overlap between the model answer and the reference answer. chrF++ works well for languages with long words, such as Luganda. | Higher |
| **Reference bits per byte** | The model reads the instruction. Then the notebook computes the log-probability of the reference answer. This is the same score as Test A. Bits per byte does not depend on the tokenizer, so you can compare models. | Lower |
| BLEU | Word n-gram overlap (a second overlap metric) | Higher |
| Luganda share | Share of answers that the GlotLID language detector labels `lug_Latn` | Higher |
| Tokens per word | Tokenizer fertility on the reference answers (information for Question 3) | Information only |

**Decision rule.** Rank the models by corpus chrF++. All models that are not clearly worse than the best model (the 95% bootstrap interval of the difference includes 0) are tied. Among the tied models, select the model with the lowest reference bits per byte. A blind human rating is available as an optional step (section 10).

**Before you start** — see the checklist in `README.md`:
- Colab runtime with an **L4 or A100** GPU. Gemma 4 E4B needs about 16 GB of GPU memory.
- Colab secret `HF_TOKEN` that can read your gated dataset.

Run the cells from top to bottom. The model loop saves the results of each model immediately to Google Drive. If the runtime stops, run the notebook again: the loop skips models that have results.
""")

md("## 1. Install packages")
code(r"""
%pip install -q "transformers==5.19.0" "huggingface_hub>=2.2" accelerate datasets sacrebleu pandas torchvision pillow fasttext-numpy2-wheel
""")

md(r"""
## 2. Settings

Change these values if necessary.

- `OUTPUT_DIR` is in your Google Drive folder `fatima technical challenge`. All prompts, raw outputs, scores and labels go there.
- `EXTRA_BUCKET_URI` is optional. It makes a second copy of the outputs: `"hf://buckets/<user>/<bucket>/<path>"` (Hugging Face Storage Bucket) or `"gs://<bucket>/<path>"` (Google Cloud Storage). Keep it empty if Drive is sufficient.
""")
code(r"""
DATASET_ID = "reuben256/alpaca-luganda-cleaned"
DATASET_SPLIT = "test"   # held-out split: these rows stay out of any later fine-tuning (Question 3)
SEED = 42
N_ITEMS = 30             # rows used to select the model
MAX_NEW_TOKENS = 200
SKIP_ROWS = [2282]       # dataset row indices to exclude. 2282: the reference talks about the English verb "runed".

DRIVE_ROOT = "/content/drive/MyDrive/fatima technical challenge"
RUN_NAME = "run2"        # run1 had the "nan" prompt error; keep its files as a record
OUTPUT_DIR = f"{DRIVE_ROOT}/outputs/01_model_selection/{RUN_NAME}"
EXTRA_BUCKET_URI = ""    # optional second copy, see above
FORCE_RERUN = False      # True = run models again even if results exist
DELETE_WEIGHTS_AFTER_RUN = True  # remove each model from the local HF cache after its run (saves disk)

# Candidate models (key -> Hugging Face repository). Checked on the Hub in October 2026.
# None of these four is gated.
CANDIDATES = {
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",                # 4B, Apache 2.0, 201 languages; thinking turned off below
    "gemma4-e4b": "google/gemma-4-E4B-it",          # 4.5B effective (8B with per-layer embeddings), Apache 2.0
    "smollm3-3b": "HuggingFaceTB/SmolLM3-3B",       # 3B, Apache 2.0
    "phi4-mini":  "microsoft/Phi-4-mini-instruct",  # 3.8B, MIT (replaces the gated Llama 3.2 3B)
}
# Optional extra candidates (add them to CANDIDATES if you want):
#   "tiny-aya-earth": "CohereLabs/tiny-aya-earth"  (3.4B, African and West Asian languages; gated with
#                      automatic approval; license CC-BY-NC = non-commercial use only)
#   "gemma4-e2b":     "google/gemma-4-E2B-it"      (2.3B effective; use it instead of E4B on a T4)

# The same system prompt for all models. It is in English so that every model understands it.
SYSTEM_PROMPT = (
    "You are a helpful assistant. Always answer in Luganda only. "
    "Do not use English. Keep the answer short and clear."
)
""")

md("## 3. Mount Google Drive, log in to Hugging Face, and check the GPU")
code(r"""
import os, json, math, time, re, gc, random, shutil, subprocess
from pathlib import Path

from google.colab import drive
drive.mount("/content/drive")

import torch
import transformers
from huggingface_hub import login, whoami, HfApi, hf_hub_download

try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except Exception as e:
    print("Colab secret HF_TOKEN not read:", e)

login(token=os.environ.get("HF_TOKEN"))
print("Logged in as:", whoami()["name"])

assert torch.cuda.is_available(), "No GPU. Select Runtime > Change runtime type > GPU."
GPU_NAME = torch.cuda.get_device_name(0)
GPU_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
# bf16 needs Ampere or newer (L4, A100). A T4 uses fp16.
DTYPE = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
print(f"GPU: {GPU_NAME} ({GPU_GB:.0f} GB) | dtype: {DTYPE} | transformers {transformers.__version__} | torch {torch.__version__}")
if DTYPE == torch.float16:
    print("WARNING: fp16 on this GPU. Gemma models can give NaN scores in fp16. Use an L4 or A100 runtime if possible.")
if GPU_GB < 20 and "gemma4-e4b" in CANDIDATES:
    print("WARNING: Gemma 4 E4B needs about 16 GB for the weights. Replace it with google/gemma-4-E2B-it on this GPU.")

out = Path(OUTPUT_DIR)
(out / "raw").mkdir(parents=True, exist_ok=True)
print("Outputs go to:", out)
""")

md(r"""
### 3.1 Check access to each model

If this cell shows `NO ACCESS`, open the model page on Hugging Face, accept the license, and run the cell again.
""")
code(r"""
api = HfApi()
access_ok = {}
for key, repo in CANDIDATES.items():
    try:
        hf_hub_download(repo, "config.json")
        access_ok[key] = True
        print(f"OK         {repo}")
    except Exception as e:
        access_ok[key] = False
        print(f"NO ACCESS  {repo}  ->  https://huggingface.co/{repo}  ({type(e).__name__})")
""")

md(r"""
## 4. Look at the Hugging Face trending list (optional)

The plan says: check the trending list. This cell shows trending text-generation models with 1B–4.6B parameters. If you find a better candidate, add it to `CANDIDATES` in section 2. Then run sections 2 and 3 again.
""")
code(r"""
try:
    rows = []
    for m in api.list_models(pipeline_tag="text-generation", sort="trending_score",
                             limit=300, expand=["safetensors", "downloads", "gated"]):
        st = getattr(m, "safetensors", None)
        n = getattr(st, "total", None) if st else None
        if n and 1.0e9 <= n <= 4.6e9:
            rows.append((m.id, round(n / 1e9, 2), getattr(m, "downloads", None), getattr(m, "gated", None)))
    print(f"{'model':60s} {'params (B)':>10s} {'downloads':>12s} {'gated':>8s}")
    for r in rows[:30]:
        print(f"{r[0]:60s} {r[1]:>10} {str(r[2]):>12s} {str(r[3]):>8s}")
except Exception as e:
    print("Trending list not available:", e)
""")

md(r"""
## 5. Load the dataset and take the 30 rows

The notebook uses a fixed seed, so the sample is the same each time. The filters keep rows that are a fair test of Luganda:

- The prompt (instruction + input) and the reference answer are both Luganda. The GlotLID language detector checks this. Some rows in the dataset are English or translation tasks (for example, "Use English to rephrase this Luganda sentence"). The filter removes these rows.
- The reference answer is short (20–300 characters, about one or two sentences). chrF++ is not reliable for long, open answers, because one prompt can have many good long answers.
- The row has no code and not many digits.
- The dataset stores an empty input as the text `"nan"`. The notebook changes this text to an empty input, so the prompt does not end with "nan".
- The reference answer is not an AI refusal (for example, "I am an AI language model made by OpenAI").
""")
code(r"""
from datasets import load_dataset
import pandas as pd
import fasttext

lid = fasttext.load_model(hf_hub_download("cis-lmu/glotlid", "model.bin"))

def lang_id(text):
    labels, probs = lid.predict(text.replace("\n", " ")[:2000] or " ", k=1)
    return labels[0].replace("__label__", ""), float(probs[0])

def is_luganda(text, min_prob=0.5):
    lang, prob = lang_id(text)
    return lang == "lug_Latn" and prob >= min_prob

ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
print(ds)

CODE_MARKERS = ("```", "def ", "print(", "{", "}", "</", "SELECT ", "import ", "#include", "=>", "==")

MISSING = {"", "nan", "none", "null"}
REFUSAL_RE = re.compile(r"\bAI\b|OpenAI|ChatGPT|model y'olulimi|model y’olulimi")

def clean(s):
    s = "" if s is None else str(s).strip()
    return "" if s.lower() in MISSING else s

def looks_bad(text):
    if any(m in text for m in CODE_MARKERS) or REFUSAL_RE.search(text):
        return True
    digits = sum(ch.isdigit() for ch in text)
    return bool(text) and digits / len(text) > 0.08

def is_candidate(r):
    ins, inp, outp = r["instruction"], r["input"], r["output"]
    return (10 <= len(ins) <= 300 and len(inp) <= 400 and 20 <= len(outp) <= 300
            and not looks_bad(" ".join((ins, inp, outp)))
            and is_luganda(ins + " " + inp) and is_luganda(outp))

rows = [{"row": i, "instruction": clean(r["instruction"]), "input": clean(r["input"]), "output": clean(r["output"])}
        for i, r in enumerate(ds) if i not in SKIP_ROWS]
random.Random(SEED).shuffle(rows)

items = []
for r in rows:
    if is_candidate(r):
        items.append(r)
        if len(items) == N_ITEMS:
            break
print(f"Selected rows: {len(items)}")

for k, r in enumerate(items):
    r["id"] = f"P{k+1:02d}"
    r["prompt"] = r["instruction"] + (("\n\n" + r["input"]) if r["input"] else "")

def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for it in data:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

write_jsonl(out / "selection_items.jsonl", items)

for r in items:
    print(f"--- {r['id']} (row {r['row']})\nPROMPT: {r['prompt']}\nREFERENCE: {r['output']}\n")
""")

md(r"""
**Check the 30 rows above.** If a row is not good Luganda, or the reference answer is wrong, add its row number to `SKIP_ROWS` in section 2. Then run section 2 and section 5 again.
""")

md(r"""
## 6. Helper functions

- `generate` gives the instruction to the model in its chat format and uses greedy decoding (no sampling), so the result is reproducible. Thinking mode is off for Qwen3.5, Gemma 4 and SmolLM3 (`enable_thinking=False`).
- `score_reference` puts the reference answer after the same chat prompt. Then it computes the sum of `log P(token | previous tokens)` over the tokens of the reference answer only. Bits per byte = `−sum_logprob / ln(2) / number of UTF-8 bytes`.
- Qwen3.5 and Gemma 4 are multimodal checkpoints. The notebook loads each model with the first model class that works, and gives it text input only.
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

def chat_text(tok, user_text):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_text}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except Exception:
        # Some chat templates do not accept a system role.
        msgs = [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + user_text}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

THINK_RE = re.compile(r"<think>.*?</think>", flags=re.S)

@torch.no_grad()
def generate(model, tok, user_text):
    text = chat_text(tok, user_text)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    out_ids = model.generate(
        **enc,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=None, top_p=None, top_k=None,
        repetition_penalty=1.05,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
    )
    new = out_ids[0, enc["input_ids"].shape[1]:]
    raw = tok.decode(new, skip_special_tokens=True)
    return {"chat_input": text, "raw_output": raw, "output": THINK_RE.sub("", raw).strip(),
            "n_new_tokens": int(new.shape[0]), "hit_max_tokens": int(new.shape[0]) >= MAX_NEW_TOKENS}

@torch.no_grad()
def score_reference(model, tok, user_text, reference):
    prefix_ids = tok(chat_text(tok, user_text), add_special_tokens=False)["input_ids"]
    ref_ids = tok(reference, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([prefix_ids + ref_ids], device=model.device)
    logits = model(input_ids=ids).logits[0, len(prefix_ids) - 1:-1].float()
    tok_lp = torch.log_softmax(logits, dim=-1).gather(1, ids[0, len(prefix_ids):, None]).squeeze(1)
    total = tok_lp.sum().item()
    n_bytes = len(reference.encode("utf-8"))
    return {
        "ref_n_tokens": len(ref_ids),
        "ref_n_words": len(reference.split()),
        "ref_n_bytes": n_bytes,
        "ref_sum_logprob": total,
        "ref_mean_logprob_per_token": total / len(ref_ids),
        "ref_bits_per_byte": -total / math.log(2) / n_bytes,
        "ref_is_finite": bool(math.isfinite(total)),
    }
""")

md(r"""
## 7. Run all candidate models

For each model: load it, write an answer for each of the 30 prompts, score the 30 reference answers, save the results to Drive, and free the GPU memory. Expect about 5–15 minutes for each model (most of the time is the download).
""")
code(r"""
from tqdm.auto import tqdm

run_info = {"gpu": GPU_NAME, "dtype": str(DTYPE), "transformers": transformers.__version__,
            "torch": torch.__version__, "seed": SEED, "n_items": len(items), "max_new_tokens": MAX_NEW_TOKENS,
            "system_prompt": SYSTEM_PROMPT, "dataset": DATASET_ID, "split": DATASET_SPLIT,
            "candidates": CANDIDATES, "models": {}}
info_path = out / "run_info.json"
if info_path.exists():
    run_info["models"] = json.loads(info_path.read_text()).get("models", {})

for key, repo in CANDIDATES.items():
    res_path = out / "raw" / f"results_{key}.jsonl"
    if res_path.exists() and not FORCE_RERUN:
        print(f"[{key}] results exist, skip.")
        continue
    if not access_ok.get(key, False):
        print(f"[{key}] no access, skip. Accept the license and run section 3.1 again.")
        continue

    print(f"\n===== {key}: {repo} =====")
    t0 = time.time()
    tok, model = load_model(repo)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params/1e9:.2f}B")

    results = []
    for it in tqdm(items, desc=key):
        g = generate(model, tok, it["prompt"])
        s = score_reference(model, tok, it["prompt"], it["output"])
        results.append({"id": it["id"], "row": it["row"], "model": key, "prompt": it["prompt"],
                        "reference": it["output"], **g, **s})
    write_jsonl(res_path, results)

    run_info["models"][key] = {"repo": repo, "n_params": n_params,
                               "minutes": round((time.time() - t0) / 60, 1)}
    info_path.write_text(json.dumps(run_info, indent=2, ensure_ascii=False))
    print(f"[{key}] done in {run_info['models'][key]['minutes']} min. Example ({results[0]['id']}):")
    print("MODEL:    ", results[0]["output"][:300])
    print("REFERENCE:", results[0]["reference"][:300])

    del model, tok
    gc.collect(); torch.cuda.empty_cache()
    if DELETE_WEIGHTS_AFTER_RUN:
        try:
            from huggingface_hub import scan_cache_dir
            cache = scan_cache_dir()
            for r in cache.repos:
                if r.repo_id == repo:
                    cache.delete_revisions(*[rev.commit_hash for rev in r.revisions]).execute()
        except Exception as e:
            print("Cache not cleaned:", e)
""")

md(r"""
## 8. Compare the answers with the reference answers

- **Corpus chrF++ / BLEU**: computed over all 30 answers together (the standard method).
- **Mean sentence chrF++**: the mean of the 30 sentence scores. The bootstrap uses these scores.
- **Reference bits per byte**: total bits / total bytes over the 30 reference answers.
""")
code(r"""
from sacrebleu.metrics import CHRF, BLEU

chrf = CHRF(word_order=2)
bleu = BLEU(effective_order=True)

models_done = [k for k in CANDIDATES if (out / "raw" / f"results_{k}.jsonl").exists()]
res_all = []
for k in models_done:
    for r in read_jsonl(out / "raw" / f"results_{k}.jsonl"):
        r["lang"], r["lang_prob"] = lang_id(r["output"])
        r["chrf"] = chrf.sentence_score(r["output"], [r["reference"]]).score
        res_all.append(r)
res_df = pd.DataFrame(res_all)
res_df.drop(columns=["chat_input"]).to_csv(out / "answers_with_metrics.csv", index=False)

if not res_df["ref_is_finite"].all():
    print("WARNING: some scores are not finite (NaN/inf). Check the dtype (section 3).")

summary = {}
for k in models_done:
    d = res_df[res_df.model == k].sort_values("id")
    f = d[d.ref_is_finite]
    summary[k] = {
        "chrf_corpus": chrf.corpus_score(d.output.tolist(), [d.reference.tolist()]).score,
        "chrf_mean": d.chrf.mean(),
        "bleu_corpus": bleu.corpus_score(d.output.tolist(), [d.reference.tolist()]).score,
        "ref_bits_per_byte": -f.ref_sum_logprob.sum() / math.log(2) / f.ref_n_bytes.sum(),
        "luganda_share": (d.lang == "lug_Latn").mean(),
        "tokens_per_word": d.ref_n_tokens.sum() / d.ref_n_words.sum(),
        "hit_max_tokens": d.hit_max_tokens.mean(),
    }
summary = pd.DataFrame(summary).T.sort_values(["chrf_corpus", "ref_bits_per_byte"], ascending=[False, True])
summary.round(3)
""")
code(r"""
# Paired bootstrap: is the best model really better than each other model?
# Resample the 30 items with replacement and compute the difference of mean sentence chrF++.
def paired_bootstrap(a, b, n=5000, seed=SEED):
    rng = random.Random(seed)
    idx = range(len(a))
    diffs = []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        diffs.append(sum(a[i] - b[i] for i in s) / len(s))
    diffs.sort()
    return sum(a[i] - b[i] for i in idx) / len(a), diffs[int(0.025 * n)], diffs[int(0.975 * n)]

pivot = res_df.pivot(index="id", columns="model", values="chrf")
best_chrf = summary.index[0]
boot_rows = []
for k in summary.index[1:]:
    d, lo, hi = paired_bootstrap(pivot[best_chrf].tolist(), pivot[k].tolist())
    boot_rows.append({"best": best_chrf, "other": k, "chrf_diff": d, "ci95_low": lo, "ci95_high": hi,
                      "clear_winner": lo > 0})
boot = pd.DataFrame(boot_rows, columns=["best", "other", "chrf_diff", "ci95_low", "ci95_high", "clear_winner"])
boot.to_csv(out / "bootstrap_chrf.csv", index=False)
display(boot.round(2))

# The same test for reference bits per byte (lower is better), against the model with the lowest value.
res_df["ref_bpb_item"] = -res_df.ref_sum_logprob / math.log(2) / res_df.ref_n_bytes
pivot_bpb = res_df.pivot(index="id", columns="model", values="ref_bpb_item")
best_bpb = summary["ref_bits_per_byte"].idxmin()
bpb_rows = []
for k in summary.index:
    if k == best_bpb:
        continue
    d, lo, hi = paired_bootstrap(pivot_bpb[best_bpb].tolist(), pivot_bpb[k].tolist())
    bpb_rows.append({"best": best_bpb, "other": k, "bpb_diff": d, "ci95_low": lo, "ci95_high": hi,
                     "clear_winner": hi < 0,
                     "items_best_lower": int((pivot_bpb[best_bpb] < pivot_bpb[k]).sum())})
boot_bpb = pd.DataFrame(bpb_rows)
boot_bpb.to_csv(out / "bootstrap_bpb.csv", index=False)
display(boot_bpb.round(3))
""")
code(r"""
# Languages that GlotLID found in the answers of each model
res_df.groupby("model")["lang"].value_counts().unstack(fill_value=0)
""")
code(r"""
# Side-by-side view: all answers for some prompts
for pid in res_df["id"].unique()[:5]:
    d = res_df[res_df.id == pid]
    print(f"===== {pid}\nPROMPT:    {d.prompt.iloc[0]}\nREFERENCE: {d.reference.iloc[0]}")
    for _, r in d.iterrows():
        print(f"  [{r.model:12s}] chrF {r.chrf:5.1f} | {r.output[:250]}")
    print()
""")

md(r"""
## 9. Decision

The rule:

1. Find the model with the highest corpus chrF++. Every model that is not clearly worse than it (the bootstrap interval of the difference includes 0) is **tied** with it.
2. If more than one model is tied, select the tied model with the lowest reference bits per byte.

Then the cell writes a draft paragraph for the report. Edit the draft before you use it.
""")
code(r"""
table = summary.copy()
table["repo"] = [CANDIDATES[m] for m in table.index]
tied = [best_chrf] + boot.loc[~boot.clear_winner.astype(bool), "other"].tolist()
best = min(tied, key=lambda m: table.loc[m, "ref_bits_per_byte"])
table["chrf_tied_with_best"] = [m in tied for m in table.index]
table.to_csv(out / "model_selection_summary.csv")
display(table.round(3))
print("Tied on chrF++:", tied)
print("Selected model:", CANDIDATES[best])

b = table.loc[best]
others = [m for m in table.index if m != best]
cmp = ", ".join(f"{CANDIDATES[m]} {table.loc[m, 'chrf_corpus']:.1f}" for m in others)
cmp_bpb = ", ".join(f"{CANDIDATES[m]} {table.loc[m, 'ref_bits_per_byte']:.2f}" for m in others)
parts = [
    f"We selected {CANDIDATES[best]} for the Luganda evaluation.",
    f"We gave {len(table)} models the same {len(items)} Luganda instructions from the held-out test split of "
    f"{DATASET_ID} and compared each answer with the reference answer.",
    f"{CANDIDATES[best]} had a corpus chrF++ of {b.chrf_corpus:.1f}" + (f" (other models: {cmp})." if cmp else "."),
]
if len(tied) > 1:
    parts.append(f"The chrF++ differences between {len(tied)} models were not significant (the 95% paired bootstrap "
                 f"intervals include 0), so we used the log-probability of the reference answers to decide.")
parts.append(f"{CANDIDATES[best]} gave the reference answers {b.ref_bits_per_byte:.2f} bits per byte"
             + (f" (other models: {cmp_bpb})" if cmp_bpb else "")
             + "; this shows how well the model predicts correct Luganda, and it is the same log-probability "
               "score that Test A uses.")
if best == best_bpb and len(boot_bpb):
    n_clear = int(boot_bpb.clear_winner.sum())
    parts.append(f"This difference was significant against {n_clear} of {len(boot_bpb)} other models "
                 f"(paired bootstrap over the {len(items)} items).")
parts.append(f"GlotLID identified {b.luganda_share:.0%} of its answers as Luganda, and its tokenizer used "
             f"{b.tokens_per_word:.2f} tokens per word.")
parts.append("[Add your own observation here: for example, the errors you saw in the answers.]")
paragraph = " ".join(parts)
(out / "model_choice_paragraph.md").write_text(paragraph + "\n", encoding="utf-8")
print("\nDRAFT PARAGRAPH (edit before you use it):\n")
print(paragraph)
""")

md(r"""
## 10. Blind human rating (optional)

chrF++ measures overlap with one reference answer. A correct answer with different words gets a low score. A native-speaker rating gives a second check.

This cell writes `rating_sheet.csv` to the Drive folder. The sheet hides the model names and shuffles the order. The key is in `rating_key.csv`. Do not open the key before you finish.

Open the CSV in Google Sheets. Fill the column `fluency` (1–5) for each row. `notes` is optional: write the error types that you see (noun-class agreement, tense/aspect, negation, subject/object marker). These notes help you write the minimal pairs later.

| Score | Fluency |
|---|---|
| 5 | Natural Luganda, like a native speaker. No grammar errors. |
| 4 | Small errors (for example, one wrong agreement). Meaning is clear. |
| 3 | Several errors. Meaning is mostly clear. |
| 2 | Many errors, or a mix of Luganda with English or another language. |
| 1 | Not Luganda, or not understandable. |

Download the sheet as CSV with the name `rating_sheet_filled.csv`, put it in the same Drive folder, and run the next cell.
""")
code(r"""
sheet = res_df[["id", "model", "prompt", "output"]].copy()
rng_sheet = random.Random(SEED + 1)
codes = {m: f"M{i+1}" for i, m in enumerate(rng_sheet.sample(models_done, len(models_done)))}
sheet["system"] = sheet["model"].map(codes)
sheet = sheet.sample(frac=1.0, random_state=SEED + 1).reset_index(drop=True)
sheet.insert(0, "item", [f"R{i+1:03d}" for i in range(len(sheet))])
sheet[["item", "model", "system"]].to_csv(out / "rating_key.csv", index=False)
sheet = sheet.drop(columns=["model"]).assign(fluency="", notes="")
sheet.to_csv(out / "rating_sheet.csv", index=False)
print(f"Wrote {len(sheet)} rows to {out/'rating_sheet.csv'}")
""")
code(r"""
RATINGS_PATH = out / "rating_sheet_filled.csv"
if RATINGS_PATH.exists():
    ratings = pd.read_csv(RATINGS_PATH).merge(pd.read_csv(out / "rating_key.csv")[["item", "model"]], on="item")
    ratings["fluency"] = pd.to_numeric(ratings["fluency"], errors="coerce")
    human = ratings.groupby("model").agg(fluency_mean=("fluency", "mean"), fluency_sd=("fluency", "std"),
                                         n_rated=("fluency", "count"))
    human.to_csv(out / "human_rating_summary.csv")
    display(summary.join(human).round(3))
else:
    print("No ratings yet:", RATINGS_PATH)
""")

md(r"""
## 11. Optional second copy and file list

All files are already in your Drive folder. This cell makes a second copy if `EXTRA_BUCKET_URI` is set, and lists the files.
""")
code(r"""
def copy_to_bucket(local_dir, uri):
    local_dir = Path(local_dir)
    if uri.startswith("hf://buckets/"):
        from huggingface_hub import create_bucket, batch_bucket_files
        parts = uri[len("hf://buckets/"):].strip("/").split("/")
        bucket_id, prefix = "/".join(parts[:2]), "/".join(parts[2:])
        create_bucket(bucket_id, private=True, exist_ok=True)
        add = [(str(p), "/".join(x for x in (prefix, p.relative_to(local_dir).as_posix()) if x))
               for p in local_dir.rglob("*") if p.is_file()]
        batch_bucket_files(bucket_id, add=add)
        print(f"Uploaded {len(add)} files to https://huggingface.co/buckets/{bucket_id}")
    elif uri.startswith("gs://"):
        from google.colab import auth
        auth.authenticate_user()
        subprocess.run(["gsutil", "-m", "rsync", "-r", str(local_dir), uri.rstrip("/")], check=True)
    else:
        raise ValueError(f"Unknown EXTRA_BUCKET_URI: {uri}")

if EXTRA_BUCKET_URI:
    copy_to_bucket(out, EXTRA_BUCKET_URI)
for p in sorted(out.rglob("*")):
    if p.is_file():
        print(p.relative_to(out))
""")

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "L4", "name": "01_model_selection.ipynb"},
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
