# Luganda minimal-pair evaluation of small open models

Question: can a small open model (1B–4B parameters) tell correct Luganda from Luganda with one wrong morpheme? Does the model know when it is wrong?

## Notebooks

| Step | Notebook | Purpose |
|---|---|---|
| 1 | [`notebooks/01_model_selection.ipynb`](notebooks/01_model_selection.ipynb) | Compare 4 candidate models on a sample of `reuben256/alpaca-luganda-cleaned`. Select the model with the most fluent Luganda. |
| 2 | (next) Test A — likelihood on minimal pairs | |

## Checklist before you run notebook 01

1. **Colab runtime.** Select *Runtime > Change runtime type > GPU*. Use an **L4** or **A100** if you can (bf16). A T4 works (fp16), but Gemma 3 can give NaN scores in fp16.
2. **Hugging Face token.** Make a token at <https://huggingface.co/settings/tokens>.
   - *Read* access is sufficient if you save outputs to Google Cloud Storage or Google Drive.
   - Use *Write* access if you save outputs to a Hugging Face dataset repository.
   - In Colab, open the key icon (Secrets). Add a secret with the name `HF_TOKEN`. Turn on *Notebook access*.
3. **Gated models.** Open these pages, log in, and accept the license. Approval can take some minutes to some hours:
   - <https://huggingface.co/google/gemma-3-4b-it>
   - <https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct>

   `Qwen/Qwen3-4B` and `HuggingFaceTB/SmolLM3-3B` are open (Apache 2.0).
4. **Bucket.** Set `BUCKET_URI` in section 2 of the notebook (`gs://…`, `hf://datasets/…`, or `/content/drive/MyDrive/…`).
5. **Time.** The model loop takes about 5–15 minutes for each model (most of it is the download). The blind rating of 40 answers (10 prompts × 4 models) takes about 30–45 minutes.

## What the notebook gives

All files go to `OUTPUT_DIR` and then to `BUCKET_URI`:

| File | Content |
|---|---|
| `gen_prompts.jsonl`, `score_texts.jsonl` | The sample (10 prompts, 200 reference texts) with dataset row numbers |
| `raw/gen_<model>.jsonl` | Full chat input, raw output, and cleaned output for each prompt |
| `raw/score_<model>.jsonl` | Log-probability, tokens, bytes, and bits per byte for each reference text |
| `generations_with_auto_metrics.csv` | Answers with language ID and chrF++ |
| `rating_sheet.csv`, `rating_key.csv` | Blind rating sheet and its key |
| `rating_sheet_filled.csv` | Your ratings |
| `model_selection_summary.csv` | Final table for each model |
| `model_choice_paragraph.md` | Draft paragraph for the report (edit it) |
| `run_info.json` | GPU, versions, settings, and run time |

## Notes

- No training happens. The log-probability (the training loss with the sign changed) is used only to score text.
- Generation uses greedy decoding with the same system prompt for all models, so the results are reproducible.
