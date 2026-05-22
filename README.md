# GPU BERTopic — interactive hyperparameter explorer

Streamlit UI on top of the [video_notebook_for_GPU_Accelerated_BERTopic_DGX_Station_40M.ipynb](video_notebook_for_GPU_Accelerated_BERTopic_DGX_Station_40M.ipynb)
pipeline lets you tweak UMAP + HDBSCAN hyperparameters in the sidebar and
refresh every BERTopic visualization without re-embedding.

The expensive step — encoding reviews with `all-MiniLM-L6-v2` — runs once and
caches to disk. Subsequent refits only run the GPU-accelerated
`UMAP → HDBSCAN → c-TF-IDF` pipeline.

## Files

| File | Purpose |
|---|---|
| [topic_modeling_app.py](topic_modeling_app.py) | Streamlit app |
| [run_app.sh](run_app.sh) | Launcher pinned to the `rapids-25.10` conda env |
| `Electronics.jsonl.gz` | McAuley Lab Amazon Electronics reviews (download separately, see notebook) |
| `.cache/` | Auto-created. Holds `preprocessed_texts_{n}.pkl` + `embeddings_{n}.npy` per sample size |

## Requirements

- One NVIDIA GPU (set `CUDA_VISIBLE_DEVICES` to pick one)
- The `rapids-25.12-python-3.13` conda env — the only env on this box with `streamlit`,
  `cuml.accel`, `bertopic`, and `datamapplot` all installed
- `Electronics.jsonl.gz` in the repo root (download via the `wget` cell in the
  notebook)

## Run

```bash
./run_app.sh
# open http://localhost:8501
```

Overrides:

```bash
PORT=8600 CUDA_VISIBLE_DEVICES=1 ./run_app.sh
PYTHON=/path/to/other/python ./run_app.sh
```

## How it works

1. **First launch at a given `nrows`** — loads the JSONL, preprocesses text,
   encodes with `SentenceTransformer`, writes both artifacts to `.cache/`.
   ~1 min for 100k docs, ~8 min for 700k docs (GPU-dependent).
2. **Click "Refit topics"** — rebuilds `UMAP` + `HDBSCAN` from the sidebar
   values, fits BERTopic on the cached embeddings, refreshes all tabs.
3. **Change `nrows`** — rebuilds the embedding cache for that size and refits.

## Sidebar controls

| Section | Params |
|---|---|
| Data | `nrows` |
| UMAP | `n_components`, `n_neighbors`, `min_dist`, `metric` |
| HDBSCAN | `min_cluster_size`, `min_samples`, `metric` (euclidean only — cuML limitation) |
| Visualizations | barchart top-N · heatmap top-N · datamap doc sample % |

## Tabs

- **Intertopic map** — `topic_model.visualize_topics()`
- **Top words barchart** — `topic_model.visualize_barchart(top_n_topics=N)`
- **Similarity heatmap** — `topic_model.visualize_heatmap(top_n_topics=N)`
- **Document datamap** — `topic_model.visualize_document_datamap(...)` on a
  configurable sample (default 1% — the call is slow even on GPU)
- **Topic table** — `topic_model.get_topic_info()`

## Cache invalidation

Delete the `.cache/` directory to force a fresh embed:

```bash
rm -rf .cache
```

Embeddings depend only on `nrows` + the embedding model name, not on the
UMAP/HDBSCAN sliders — refits never invalidate the cache.
