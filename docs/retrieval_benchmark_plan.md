# Retrieval benchmark plan

Task spec for a quantitative benchmark of `tfm-embeddings`. Self-contained:
everything needed to execute it is described here or in the repo.

## Research question

The README's UMAP gallery shows that foundation-model row embeddings *look*
structured, but the load-bearing question is quantitative:

> Do rows retrieved by nearest-neighbor search in foundation-model embedding
> space match the query's label more often than rows retrieved with classical
> distances on raw features?

If yes, "row-RAG" with these models has real value. If no, the library is a
visualization toy — equally important to know. The result should hold up as a
README section and as a follow-up comment on the three issues we opened with
the model authors (soda-inria/tabicl#133, PriorLabs/tabpfn-extensions#334,
google-research/tabfm#38).

## Method

For each (dataset, method) pair:

1. Split 50/50 into corpus and query sets (stratified for classification,
   `random_state=42`, same splits for every method).
2. Embed/represent all rows with the method (fit on the corpus **with**
   `y_corpus`, embed queries as unseen rows — for `TabularEmbedder` this is
   exactly `model.fit(X_corpus, y_corpus)` then `model.search(X_query, top_k=k)`).
3. For each query row, retrieve the top-k most similar corpus rows.
4. Score retrieval quality (metrics below).

### Methods to compare

Baselines (raw features, standardized with `StandardScaler` fit on the corpus):

- Euclidean distance on standardized features
- Cosine similarity on standardized features

Foundation models (via `TabularEmbedder`, cosine similarity, default
`aggregate="mean"`):

- `TabularEmbedder("tabicl")`
- `TabularEmbedder("tabpfn")` — requires `TABPFN_TOKEN` env var (license auth);
  skip with a note if unavailable
- `TabularEmbedder("tabfm", n_estimators=4)` — slow (~1–2 min per fit+encode on
  CPU at n=500); requires Python >= 3.11

Optional secondary condition: fit **without** `y` (pseudo-target mode) to
quantify how much the label-awareness contributes. This directly feeds the
"Benchmarks for the y=None pseudo-target strategy" roadmap item.

### Metrics

Classification datasets:

- **Precision@k** (k = 1, 5, 10): fraction of retrieved rows whose label equals
  the query row's label. Report mean over queries.
- **kNN-classifier accuracy** (k = 5, majority vote over retrieved labels) as a
  single-number summary.

Regression datasets:

- **Neighbor target MAE** (k = 5): |mean(target of retrieved rows) − query
  target|, averaged over queries. Lower is better.

### Datasets

Small enough for CPU, diverse enough to be credible (all available via
sklearn/OpenML):

| Dataset | Task | Source |
|---------|------|--------|
| Breast cancer | binary | `sklearn.datasets.load_breast_cancer` |
| Digits (subsample 1000, seed 42) | 10-class | `sklearn.datasets.load_digits` |
| Adult / census income (subsample 2000) | binary, mixed dtypes | `sklearn.datasets.fetch_openml("adult", version=2)` |
| California housing (subsample 1000) | regression | `sklearn.datasets.fetch_california_housing` |

Adult matters: it has categorical columns, which the raw-feature baselines must
one-hot encode while the foundation models ingest natively — a realistic
advantage/disadvantage to surface. Use `pd.DataFrame` inputs there.

## Deliverables

1. `examples/retrieval_benchmark.py` — runs the full grid, prints a results
   table, saves it to `docs/retrieval_benchmark.md` (markdown table) and
   optionally a bar-chart figure to `docs/retrieval_benchmark.png`.
2. A short "Does it beat raw-feature kNN?" section in `README.md` with the
   headline table and a link to the script.
3. Honest reporting: if a foundation model loses to standardized Euclidean on
   some dataset, that goes in the table too.

## Practical notes

- Work in `/Users/vstenby/code/tfm-embeddings`, uses uv (`uv run python ...`).
  Extras: `uv pip install -e ".[all]"` style; the venv in `.venv` already has
  all three backends installed.
- Pin versions the way `examples/embedding_gallery.py` does: set
  `os.environ.setdefault("TABPFN_MODEL_VERSION", "v3")` **before** importing
  tabpfn-related code; label columns TabICLv2 / TabFM v1.0.0 / TabPFNv3.
- Never write the TabPFN token to any file or commit; read it from the
  `TABPFN_TOKEN` env var only, and skip TabPFN gracefully when unset.
- Reuse one fitted model per (dataset, backend) for all metrics — fits are the
  expensive part, scoring is cheap.
- `search()` caches corpus embeddings on the model (`corpus_embeddings_`), so
  repeated queries are cheap. Note: retrieval is over the *corpus* rows, and
  query rows are embedded as unseen test rows, so query labels never leak.
- Expected total runtime: dominated by TabFM, roughly 10–20 min on CPU for the
  grid above. Print progress per cell.
