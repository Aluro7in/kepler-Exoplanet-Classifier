# Kepler Q1‑Q17 DR25 TCE — ML + DL + Transformer Classification Pipeline

A full, runnable pipeline that trains and compares **4 classical ML models**, a **PyTorch MLP**, and a **PyTorch Transformer** (FT-Transformer-style, tabular self-attention) on the Kepler DR25 Threshold Crossing Event (TCE) table — classifying each TCE as a likely planet signal (`PLANET`) vs. not (`NOT_PLANET`) — with live/animated visualization of how each neural network learns, and an optional LangChain-generated narrative report.

Built from the master prompt in `MASTER_PROMPT.md`.

---

## ⚠️ Read this first: label strategy

This delivery of the DR25 TCE table (`av_training_set`, `av_pred_class`, etc.) ships with **empty disposition columns** — NASA didn't include them in this export. `src/data_loader.py` handles this automatically:

1. **Option A (tries first):** downloads NASA's cumulative KOI table and merges in the real `koi_disposition` (`CONFIRMED` / `CANDIDATE` / `FALSE POSITIVE`) by `(kepid, tce_plnt_num)`. This needs outbound internet access to `exoplanetarchive.ipac.caltech.edu`.
2. **Option B (automatic fallback):** if that request fails (e.g. no internet, firewalled sandbox — **this is what happened when this repo was built and run**, see logs below), it builds a **weak proxy label** instead: `PLANET = (tce_rogue_flag == 0) AND (tce_nkoi > 0)`.

**The numbers in `reports/model_comparison.csv` in this repo were produced using the Option B proxy label**, because this build environment had no route to the NASA Exoplanet Archive (`HTTP 403`). Near-perfect accuracy across every model is expected in that case — `tce_nkoi`/`tce_rogue_flag` correlate strongly with pipeline-internal features already in the table, so this is really "can the model reconstruct a rule from the same table" rather than a validated exoplanet-detection benchmark.

**To get scientifically meaningful results, run this on a machine with internet access** — `data_loader.py` will then automatically use Option A and you'll get real, harder, 3-class-collapsed-to-2-class disposition labels. No code changes needed.

---

## What's in here

```
kepler-tce-project/
├── data/
│   ├── raw/                           # Raw TCE CSV exports
│   └── kepler_performance_and_predictions.db
├── src/
│   ├── data_loader.py                 # Label building (Option A / B, see above)
│   ├── preprocessing.py               # Column selection, log1p, impute + scale
│   ├── split.py                       # Grouped train/val/test split (by kepid)
│   ├── train_ml.py                    # LogReg, Random Forest, XGBoost, SVM
│   ├── models_pytorch.py              # MLP, TabTransformer, CNN1D, and RNN definitions
│   ├── train_nn.py                    # Trains MLP / Transformer, live viz, gifs
│   ├── train_cnn.py                   # 1D CNN for transit shape motifs
│   ├── train_rnn.py                   # GRU recurrent sequence classifier
│   ├── evaluate.py                    # Scores all saved models on the test set
│   ├── experiment_tracking.py         # Append-only experiment run registry
│   ├── rag_pipeline.py                # RAG pipeline for candidate signal summarization
│   ├── run_rag_pipeline.py            # End-to-end execution runner
│   ├── langchain_report.py            # LangChain narrative report (LLM or template)
│   └── predict.py                     # Score new TCE rows with any saved model
├── models/
│   ├── ml/*.joblib                    # Trained classical models
│   ├── dl/*.pt, *_config.joblib       # Trained MLP & Transformer checkpoints
│   └── sequence/*.pt                  # Trained CNN and RNN weights
├── reports/
│   ├── figures/*.png, *.gif           # Training curves, ROC curves, PCA GIFs
│   ├── model_comparison.csv          # Evaluation metric comparison
│   └── summary_report.txt             # Auto-generated textual evaluation summary
├── tests/
│   └── test_rag_pipeline.py           # Unit tests for discovery & retrieval
├── visuals/                           # Visual summaries of model and RAG results
├── requirements.txt                   # Environment package dependencies
└── MASTER_PROMPT.md                   # Original spec this project was built from
```

---

## Results (Proxy Labels — see caveat above)

| model | accuracy | precision | recall | f1 | roc_auc |
|---|---|---|---|---|---|
| logistic_regression | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| random_forest | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| gradient_boosting (xgboost) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **mlp (pytorch)** | 0.996 | 0.995 | 0.995 | 0.995 | 1.000 |
| **transformer (pytorch)** | 0.995 | 0.991 | 0.995 | 0.993 | 0.999 |
| svm | 0.990 | 0.982 | 0.989 | 0.986 | 1.000 |

---

## The "watch it learn" visualizations

Every neural model run produces, in `reports/figures/`:

- `<model>_training_curves.png` — train vs. validation loss & accuracy per epoch.
- `<model>_embedding_evolution.gif` — **the neural pattern forming, live**: the network's learned representation of every validation TCE, projected to 2D with PCA, snapshotted every few epochs, stitched into an animated GIF. Watch the red (`NOT_PLANET`) and green (`PLANET`) points start mixed together and pull apart as the network learns.
- `<model>_test_diagnostics.png` — confusion matrix + ROC curve on the held-out test set.

If you run `train_nn.py` with `--live` on a machine with a display, it also pops up a real-time matplotlib window that redraws the loss curve every epoch (`plt.ion()` mode) — the GIF is the headless-safe equivalent, so it also works in CI/servers/this sandbox.

---

## How to run it yourself

```bash
pip install -r requirements.txt

# 1. Classical ML — trains & saves LogReg, RF, XGBoost, SVM
python src/train_ml.py

# 2. Neural network (MLP)
python src/train_nn.py --model mlp --epochs 30

# 3. Transformer (tabular self-attention)
#    NOTE: on a single-CPU machine this is slow. Defaults below (d_model=32,
#    2 layers, 4 heads) are fine on a GPU or multi-core CPU; on a weak CPU,
#    the smaller config used to build this repo finishes in ~6 minutes:
python src/train_nn.py --model transformer --epochs 10 \
    --batch_size 2048 --d_model 16 --n_heads 2 --n_layers 1 --patience 4

# 4. Compare everything on the held-out test set
python src/evaluate.py

# 5. Optional: LangChain narrative report
#    (set ANTHROPIC_API_KEY or OPENAI_API_KEY to get a real LLM write-up;
#    otherwise a deterministic template report is produced automatically)
python src/langchain_report.py

# 6. Score new TCE rows with any saved model
python src/predict.py --model random_forest --input new_tces.csv
python src/predict.py --model mlp --input new_tces.csv
python src/predict.py --model transformer --input new_tces.csv
```

---

## A bug that was caught and fixed while building this

An earlier version of `preprocessing.py`'s `log1p` step computed its shift constant (to handle a couple of columns with small negative values) from `df[col].min()` **on whatever DataFrame was passed in** — correct on the full 34k-row training set, but silently wrong when `predict.py` later called it on a tiny batch of new rows (a different min $\rightarrow$ a different shift $\rightarrow$ a different transformed value $\rightarrow$ wrong predictions). 

Random Forest and MLP predictions on the same 20 sample rows disagreed sharply until this was found. Fixed by hardcoding the shift constants (`LOG1P_SHIFTS` in `preprocessing.py`) from the full training distribution, so the transform is now identical at train time and inference time. If you retrain on very different data, double check those shift constants still make sense (they just need to keep every value $\ge 0$ before `log1p`).

---

## Extending this

- **Real labels:** run on a machine with internet access to the NASA Exoplanet Archive to get real `koi_disposition` labels via Option A.
- **3-class problem:** `data_loader.py`'s `build_labels_from_koi_merge` currently collapses to a 2-class `PLANET`/`NOT_PLANET` target; switch to the 3-class `CONFIRMED`/`CANDIDATE`/`FALSE_POSITIVE_OR_NTP` mapping (commented in the master prompt) if you want the finer-grained task.
- **Feature importance / SHAP:** the langchain report's auto-generated "next step" suggests this — worth doing before trusting any model, especially given the proxy-label caveat above.
- **Bigger transformer:** if you have more compute, bump `d_model`/`n_layers` back up to the defaults (32 / 2) or beyond — the architecture in `models_pytorch.py` scales cleanly.