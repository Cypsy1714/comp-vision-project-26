# comp vision

Fine-grained cat/dog breed recognition. Detector → views → cnn → meta, four stub stages
chained through folders. Build on the stubs; the contract is `def run(cfg)` per file.

## run

```
uv sync
uv run python src/main.py
```

Reads `input/`, writes `output/predictions/predictions.csv`.

Run one stage alone (point its input path at the matching `test/` folder in `config.yaml`):

```
uv run python src/detector.py   # A
uv run python src/views.py      # B
uv run python src/cnn.py        # C
uv run python src/meta.py       # D
```

## config knobs (`config.yaml`)

- `image_size` — crop/resize size (224)
- `pipeline` — step order, add/remove/reorder
- `views.active` — the K views (K = list length), read by views + cnn + meta
- `detector.reject_on_no_animal` and `meta.reject_enabled` — the two reject toggles

## parts

- **A — `src/detector.py`** — yolo crop the biggest animal. in `input/` → out `output/crops`.
- **B — `src/views.py`** — make K views per crop. in `output/crops` → out `output/views`.
- **C — `src/cnn.py`** — train + score views, `(K, num_classes)`. in `output/views` → out `output/scores`.
- **D — `src/meta.py`** — meta-classifier + reject, `predictions.csv`. in `output/scores` → out `output/predictions`.

## folders

- `input/` — images to run.
- `output/` — generated crops, views, scores, predictions.
- `test/` — sample inputs so each stage runs alone: `images/`, `crops/`, `views/`, `scores/`.
