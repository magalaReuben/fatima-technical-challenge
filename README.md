# Luganda minimal-pair evaluation of small open models

Question: can a small open model (1B–4B parameters) tell correct Luganda from Luganda with one wrong morpheme? Does the model know when it is wrong?

## Notebooks

The notebooks are in this repository and in the Google Drive folder `fatima technical challenge`. Open them from Drive with Google Colab.

| Step | Notebook | Purpose |
|---|---|---|
| 1 | [`notebooks/01_model_selection.ipynb`](notebooks/01_model_selection.ipynb) ([open in Colab from Drive](https://colab.research.google.com/drive/1h7rZDlDIXI4ONXdBXR-dnd_XsjHUmuuv)) | Give 4 candidate models the same 30 Luganda instructions from `reuben256/alpaca-luganda-cleaned` (test split). Compare each answer with the reference answer. Select the best model for the next evaluations. |
| 2 | (next) Test A — likelihood on minimal pairs | |

## Candidate models (checked on the Hugging Face Hub, October 2026)

| Key | Repository | Parameters | License | Gated |
|---|---|---|---|---|
| qwen3.5-4b | `Qwen/Qwen3.5-4B` | 4B | Apache 2.0 | No |
| gemma4-e4b | `google/gemma-4-E4B-it` | 4.5B effective (8B with embeddings) | Apache 2.0 | No |
| smollm3-3b | `HuggingFaceTB/SmolLM3-3B` | 3B | Apache 2.0 | No |
| phi4-mini | `microsoft/Phi-4-mini-instruct` | 3.8B | MIT | No |

`microsoft/Phi-4-mini-instruct` replaces `meta-llama/Llama-3.2-3B-Instruct`, because the Llama model is gated. Optional extra: `CohereLabs/tiny-aya-earth` (African and West Asian languages; gated with automatic approval; non-commercial license).

## Checklist before you run notebook 01

1. **Colab runtime.** Select *Runtime > Change runtime type > GPU* and pick an **L4** or **A100**. Gemma 4 E4B needs about 16 GB of GPU memory, so it does not fit on a T4. On a T4, replace it with `google/gemma-4-E2B-it`.
2. **Hugging Face token.** Make a token at <https://huggingface.co/settings/tokens>. It must be able to read your gated dataset `reuben256/alpaca-luganda-cleaned`. In Colab, open the key icon (Secrets). Add a secret with the name `HF_TOKEN`. Turn on *Notebook access*.
3. **Google Drive.** The notebook mounts your Drive. Accept the request when Colab asks.
4. **Time.** About 5–15 minutes for each model (most of it is the download).

## What the notebook gives

All files go to `My Drive/fatima technical challenge/outputs/01_model_selection/<RUN_NAME>/` (run1 had the "nan" prompt error; use run2):

| File | Content |
|---|---|
| `selection_items.jsonl` | The 30 rows (prompt, reference answer, dataset row number) |
| `raw/results_<model>.jsonl` | Full chat input, raw output, cleaned output, and reference log-probability for each row |
| `answers_with_metrics.csv` | Answers with chrF++ and language ID |
| `model_selection_summary.csv` | Final table for each model |
| `bootstrap_chrf.csv`, `bootstrap_bpb.csv` | 95% paired bootstrap intervals of the chrF++ and bits-per-byte differences |
| `model_choice_paragraph.md` | Draft paragraph for the report (edit it) |
| `rating_sheet.csv`, `rating_key.csv` | Optional blind rating sheet and its key |
| `run_info.json` | GPU, versions, settings, and run time |

## Notes

- No training happens. The log-probability (the training loss with the sign changed) is used only to score text.
- Generation uses greedy decoding and the same system prompt for all models, so the results are reproducible.
