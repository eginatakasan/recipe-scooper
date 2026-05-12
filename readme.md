# Recipe Scooper

End-to-end pipeline for extracting structured recipe data (title, author, ingredients, steps, date) from arbitrary recipe web pages using weak supervision (Snorkel) and a fine-tuned MarkupLM model.

---

## Setup

```bash
pip install -r requirements.txt
playwright install chromium
```

---

## 1. Download the Kaggle dataset

Download the [Recipe NLG](https://www.kaggle.com/datasets/paultimothymooney/recipenlg) CSV to:

```
data/kaggle/recipes.csv
```

---

## 2. Sample URLs

Samples train and test URLs from the Kaggle CSV, stratified by domain.

```bash
python -m crawler.sampler --csv data/kaggle/recipes.csv
```

**Output:**
- `data/kaggle/sampled_urls.csv` — train URLs
- `data/kaggle/test_urls.csv` — held-out test URLs

| Parameter | Default | Description |
|---|---|---|
| `--csv` | *(required)* | Path to Kaggle recipes CSV |
| `--out` | `data/kaggle/sampled_urls.csv` | Train URL output path |
| `--test-out` | `data/kaggle/test_urls.csv` | Test URL output path |
| `--per-domain` | `100` | Max URLs sampled per domain |
| `--top-domains` | `100` | Number of top domains to include |
| `--test-fraction` | `0.15` | Fraction of URLs reserved for test |

**Example — larger sample from top 50 domains:**
```bash
python -m crawler.sampler \
    --csv data/kaggle/recipes.csv \
    --per-domain 200 \
    --top-domains 50 \
    --test-fraction 0.2
```

---

## 3. Crawl

Downloads raw HTML for all sampled URLs. Crawls train and test sets in parallel.

```bash
python -m crawler.run_crawl --csv data/kaggle/sampled_urls.csv
```

**Output:**
- `data/raw_html/` — crawled train HTML + `crawl_log.csv`
- `data/raw_html_test/` — crawled test HTML + `crawl_log.csv`

| Parameter | Default | Description |
|---|---|---|
| `--csv` | `data/kaggle/sampled_urls.csv` | Train URL CSV |
| `--out` | `data/raw_html` | Output directory for train HTML |
| `--test-csv` | `data/kaggle/test_urls.csv` | Test URL CSV |
| `--test-out` | `data/raw_html_test` | Output directory for test HTML |
| `--concurrency` | `5` | Number of concurrent workers |
| `--delay` | `1.0` | Delay (seconds) between requests per worker |

**Example — slower crawl to avoid rate limiting:**
```bash
python -m crawler.run_crawl \
    --csv data/kaggle/sampled_urls.csv \
    --concurrency 3 \
    --delay 2.0
```

---

## 4. Label

### 4a. Generate training labels (weak supervision)

Applies Snorkel labeling functions to all crawled HTML, fits a LabelModel to resolve conflicts, and exports high-confidence node-level labels.

```bash
python -m labeling.run_labeling \
    --crawl-log data/raw_html/crawl_log.csv \
    --kaggle-csv data/kaggle/sampled_urls.csv \
    --out data/labeled.jsonl
```

**Output:** `data/labeled.jsonl` — one JSON object per page with `url`, `nodes`, `xpaths`, `tags`, `labels`

| Parameter | Default | Description |
|---|---|---|
| `--crawl-log` | `data/raw_html/crawl_log.csv` | Crawl log from step 3 |
| `--kaggle-csv` | `data/kaggle/sampled_urls.csv` | Kaggle CSV for distant supervision |
| `--out` | `data/labeled.jsonl` | Output JSONL path |
| `--confidence` | `0.7` | Minimum LabelModel confidence to assign a non-O label |

**Example — lower confidence threshold to increase label coverage:**
```bash
python -m labeling.run_labeling \
    --crawl-log data/raw_html/crawl_log.csv \
    --kaggle-csv data/kaggle/sampled_urls.csv \
    --out data/labeled.jsonl \
    --confidence 0.6
```

### 4b. Generate test labels (schema.org ground truth)

Builds a gold test set using schema.org metadata as automatic ground truth (no LabelModel).

```bash
python -m labeling.build_test_set \
    --crawl-log data/raw_html_test/crawl_log.csv \
    --out data/test.jsonl
```

| Parameter | Default | Description |
|---|---|---|
| `--crawl-log` | `data/raw_html/test_crawl_log.csv` | Test crawl log |
| `--out` | `data/test.jsonl` | Output JSONL path |
| `--min-fields` | `2` | Minimum schema.org fields required to include a page |

**Example — stricter quality filter:**
```bash
python -m labeling.build_test_set \
    --crawl-log data/raw_html_test/crawl_log.csv \
    --out data/test.jsonl \
    --min-fields 3
```

### 4c. Test labeling functions

Inspect which labeling functions fire on specific pages before committing to a full labeling run. Useful for debugging and tuning LFs.

**Against 10 random crawled files (default):**
```bash
python -m labeling.test_lfs --category author
python -m labeling.test_lfs --category ingredients
python -m labeling.test_lfs --category steps
python -m labeling.test_lfs --category all
```

**Against a live URL (fetched at runtime):**
```bash
python -m labeling.test_lfs --category author --url https://www.recipetineats.com/beef-stew/
```

**Against a local HTML file:**
```bash
python -m labeling.test_lfs --category ingredients --file htmls/some_recipe.htm
```

**Against a specific file by crawl log index:**
```bash
python -m labeling.test_lfs --category steps --index 42
```

| Parameter | Default | Description |
|---|---|---|
| `--category` | *(required)* | `author`, `ingredients`, `steps`, or `all` |
| `--crawl-log` | `data/raw_html/crawl_log.csv` | Crawl log to sample from |
| `--kaggle-csv` | `data/kaggle/sampled_urls.csv` | Kaggle CSV for distant supervision signals |
| `--n` | `10` | Number of random files to test |
| `--index` | — | Test a single file by index in the crawl log |
| `--url` | — | Fetch and test a live URL (overrides `--n`/`--index`) |
| `--file` | — | Test a local `.htm`/`.html` file by path (overrides all others) |

---

## 5. Train

### Option A — Local

```bash
python -m training.train \
    --labeled data/labeled.jsonl \
    --output models/markuplm-recipe
```

| Parameter | Default | Description |
|---|---|---|
| `--labeled` | `data/labeled.jsonl` | Labeled JSONL from step 4a |
| `--output` | `models/markuplm-recipe` | Directory to save the trained model |
| `--epochs` | `5` | Number of training epochs |
| `--batch-size` | `4` | Per-device batch size |
| `--grad-accum` | `2` | Gradient accumulation steps |
| `--lr` | `2e-5` | Learning rate |

**Example — more epochs with larger effective batch:**
```bash
python -m training.train \
    --labeled data/labeled.jsonl \
    --output models/markuplm-recipe \
    --epochs 10 \
    --batch-size 4 \
    --grad-accum 4 \
    --lr 1e-5
```

### Option B — Kaggle (GPU)

1. Upload `data/labeled.jsonl` as a Kaggle dataset named `recipe-scooper-labeled`
2. Create a new Kaggle notebook and upload `training/kaggle_notebook.py`
3. Enable GPU accelerator in notebook settings
4. Run cells top to bottom — the trained model saves to `/kaggle/working/markuplm-recipe/`
5. Download the model output and place it at `models/markuplm-recipe/`

---

## 6. Evaluate

```bash
python -m training.evaluate \
    --model-dir models/markuplm-recipe \
    --test-jsonl data/test.jsonl
```

| Parameter | Default | Description |
|---|---|---|
| `--model-dir` | *(required)* | Path to trained model directory |
| `--test-jsonl` | *(required)* | Gold test set from step 4b |
| `--device` | `cuda` if available, else `cpu` | Device for inference |

**Example — force CPU:**
```bash
python -m training.evaluate \
    --model-dir models/markuplm-recipe \
    --test-jsonl data/test.jsonl \
    --device cpu
```

---

## 7. Inference

Run the trained model on any recipe URL.

```bash
python inference.py --url https://www.recipetineats.com/beef-stew/
```

| Parameter | Default | Description |
|---|---|---|
| `--url` | *(required)* | URL of the recipe page to extract |
| `--model` | `models/markuplm-recipe` | Path to trained model directory |
| `--device` | `cuda` if available, else `cpu` | Device for inference |

**Example — specify model path and force CPU:**
```bash
python inference.py \
    --url https://www.allrecipes.com/recipe/8652/garlic-chicken/ \
    --model models/markuplm-recipe \
    --device cpu
```

**Output format:**
```json
{
  "url": "https://...",
  "title": "Beef Stew",
  "author": "Nagi Maehashi",
  "date": "2019-01-17T20:53:36+00:00",
  "ingredients": ["1.2 kg chuck beef", "..."],
  "steps": ["Sprinkle beef with salt and pepper.", "..."],
  "images": { "primary": "https://...", "body": [] },
  "confidence": { "title": 1.0, "ingredients": 0.0, "steps": 0.0, "author": 1.0, "date": 0.0 },
  "extraction_status": "partial"
}
```

`extraction_status` is `complete` when title, ingredients, and steps are all found; `partial` when at least one is found; `failed` otherwise.
