<div align="center">

# tfm-embeddings

**SentenceTransformers, but for tables.**

Embeddings, similarity search, and retrieval for tabular data —<br>
powered by tabular foundation models.

[![test](https://github.com/vstenby/tfm-embeddings/actions/workflows/test.yml/badge.svg)](https://github.com/vstenby/tfm-embeddings/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/tfm-embeddings.svg)](https://pypi.org/project/tfm-embeddings/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Backends](https://img.shields.io/badge/backends-TabICL%20·%20TabPFN%20·%20TabFM-orange.svg)](#backends)

</div>

Tabular foundation models like [TabICL](https://github.com/soda-inria/tabicl),
[TabPFN](https://github.com/PriorLabs/TabPFN), and
[TabFM](https://github.com/google-research/tabfm) compute rich per-row
representations internally as part of in-context learning. `tfm-embeddings`
exposes them behind the familiar encode/similarity API from
[sentence-transformers](https://www.sbert.net/), for similarity search,
retrieval, clustering, visualization, and feature extraction on tabular data.

> [!WARNING]
>
> #### Experimental Code Notice
> This project is experimental and early-stage. APIs may change without notice.

<img src="https://raw.githubusercontent.com/vstenby/tfm-embeddings/main/docs/embedding_gallery.png" alt="UMAP of held-out test embeddings for three datasets (rows) and three backends (columns)" width="100%">

*Held-out test rows embedded by each backend (columns) on three datasets
(rows: binary classification, 10-class classification, regression), then
UMAP-projected and colored by the true class or target. Every panel uses the
same two lines of code: `model.fit(X_train, y_train)` then
`model.encode(X_test)` — and the test rows are never part of the context, so
the visible structure is genuinely inferred. Reproduce with
[`examples/embedding_gallery.py`](examples/embedding_gallery.py).*

## ✨ Highlights

- **One API, three foundation models** — TabICL, TabPFN, and TabFM
  behind the same `fit` / `encode` / `similarity` / `search` interface.
- **Retrieval built in** — top-k nearest-row search over your context table,
  RAG-style.
- **Leakage-aware by design** — rows are always embedded as unseen test rows,
  and `fit_transform_oof` provides out-of-fold embeddings for downstream
  training.
- **Ensemble control** — mean, concatenated, or raw per-member embeddings.
- **scikit-learn compatible** — drop it in a `Pipeline` as a transformer step.

## Installation

```bash
pip install tfm-embeddings[tabicl]   # TabICL backend
pip install tfm-embeddings[tabpfn]   # TabPFN backend
pip install tfm-embeddings[tabfm]    # TabFM backend (Python >= 3.11)
pip install tfm-embeddings[all]      # everything
```

## Usage

```python
from tfm_embeddings import TabularEmbedder

# 1. Load a tabular foundation model
model = TabularEmbedder("tabicl")

# 2. Set the context table — all embeddings are conditioned on it
model.fit(X_corpus, y_corpus)  # y is optional

# 3. Calculate embeddings by calling model.encode()
embeddings = model.encode(X_corpus)
print(embeddings.shape)
# (469, 512)

# 4. Calculate the embedding similarities
query_embeddings = model.encode(X_query)
similarities = model.similarity(query_embeddings, embeddings)

# 5. Or retrieve the most similar corpus rows directly
indices, scores = model.search(X_query, top_k=5)
```

## How this differs from sentence embeddings

Sentence embeddings are context-free: a sentence always maps to the same vector.
Tabular foundation model embeddings are **context-dependent** — a row's vector is
conditioned on the entire context table (column distributions and, for
target-aware models, the labels). This has three practical consequences:

1. **`fit` comes first.** The context table is part of the model state. Every
   `encode` call embeds rows against it as unseen test rows.
2. **Embeddings are only comparable within one fitted model.** There is no shared
   embedding space across tables, contexts, or backends.
3. **Labels shape the space.** With `y` provided, similarity is task-aware:
   rows that the model treats similarly *for predicting `y`* end up close. With
   `y=None`, a pseudo-target is synthesized for approximately unsupervised
   embeddings (experimental).

Rows passed to `encode` are always embedded as test rows, so their own labels are
never visible to the model — embedding your corpus does not leak its labels.

## Backends

| Backend | Model string | Embedding source | Requires |
|---------|-------------|------------------|----------|
| TabICL  | `"tabicl"` or `"tabicl/<checkpoint>"` | Row representations after column-wise embedding + row-wise interaction (pre-ICL), extracted via forward hook | `tabicl>=2.1` |
| TabPFN  | `"tabpfn"` or `"tabpfn/<model_path>"` | Per-row transformer outputs via the public `get_embeddings` API | `tabpfn>=2.0` (local, not the API client); downloading weights requires [license authentication](https://ux.priorlabs.ai) (`TABPFN_TOKEN`) |
| TabFM | `"tabfm"` or `"tabfm/<checkpoint_path>"` | Row representations before the in-context learning transformer (`row_interactor_2`), extracted via forward hook | `tabfm[pytorch]>=1.0.0`, Python >= 3.11 |

Backend-specific options are passed through the constructor:

```python
model = TabularEmbedder("tabicl", n_estimators=4, device="cpu", random_state=0)
```

### Do I need `y`?

All three backends are supervised in-context learners: the context table must
contain a target. TabICL's and TabFM's column/cell embedders are target-aware,
and TabPFN attends over labeled context tokens — none of them has a truly
unsupervised mode.

`fit` therefore never skips the target; it resolves it:

| You pass | What happens | Embedding space |
|----------|--------------|-----------------|
| Discrete `y` | Classification checkpoint | Shaped by the classification task |
| Continuous `y` | Regression checkpoint | Shaped by the regression task |
| No `y` | Regression checkpoint with a **pseudo-target** (standard-normal noise, seeded by `random_state`) | Approximately task-neutral (**experimental**) |

This behavior is identical across all backends. The pseudo-target makes the
context labels uninformative so the embeddings mostly reflect feature
structure, but this strategy is unvalidated — if you have a meaningful target,
pass it. Note that with no `y`, changing `random_state` changes the noise and
therefore the embeddings.

### Ensemble aggregation

All backends ensemble over multiple views of the table (e.g. feature shuffles),
and each ensemble member produces embeddings in its own space. `aggregate`
controls how they are combined:

- `"mean"` (default): average across members → `(n_rows, dim)`
- `"concat"`: concatenate members → `(n_rows, n_members * dim)`
- `"none"`: raw members → `(n_members, n_rows, dim)`

## Using embeddings as features (out-of-fold)

For retrieval and visualization, `fit` + `encode` is all you need. But when
embeddings become *features for training a downstream model*, there is a
subtle trap: `encode(X_train)` after `fit(X_train, y_train)` embeds every
training row with an identical, labeled copy of itself in the context, so
each embedding partially encodes its own label. Held-out evaluation stays
honest, but the downstream model learns to rely on a signal that is absent
for genuinely unseen rows.

`fit_transform_oof` removes that self-influence with out-of-fold embedding
(following [A Closer Look at TabPFN v2](https://arxiv.org/abs/2502.17361)):
the data is split into folds, each fold is embedded by a model fitted on the
*other* folds, and a final full-data model is fitted for embedding unseen
rows:

```python
model = TabularEmbedder("tabicl")

train_embeds = model.fit_transform_oof(X_train, y_train, n_fold=5)  # leakage-free
test_embeds = model.encode(X_test)                                  # full-data model

clf = LogisticRegression().fit(train_embeds, y_train)
clf.score(test_embeds, y_test)
```

Three things to know:

- **`encode` never returns out-of-fold embeddings.** It always uses the final
  full-data model, so `model.encode(X_train)` afterwards gives different
  (self-influenced) vectors than the OOF ones. Keep the return value of
  `fit_transform_oof`.
- **Non-independent rows need `groups=`.** If the same entity appears in
  multiple rows (duplicated records, repeated measurements per patient),
  plain K-fold can place its copies in different folds and the label leaks
  back in through the copy. Pass `groups=` to keep an entity's rows in one
  fold — only you know your data's grouping structure.
- **Cost is `n_fold + 1`** fits and encodes. Splits are stratified for
  classification targets.

`TabularEmbedder` also implements the scikit-learn transformer API
(`transform` / `fit_transform` / `get_params`), so it works directly in a
pipeline:

```python
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(TabularEmbedder("tabicl"), LogisticRegression())
pipe.fit(X_train, y_train).score(X_test, y_test)
```

Note that a pipeline's `fit` embeds training rows with themselves in the
context (the self-influence described above) — convenient, but for the most
honest training features use `fit_transform_oof` outside a pipeline.

## Examples

See [`examples/`](examples/):

- [`breast_cancer_retrieval.py`](examples/breast_cancer_retrieval.py) —
  fit a corpus, embed queries, retrieve the most similar rows.
- [`embedding_gallery.py`](examples/embedding_gallery.py) — the gallery
  figure above: three datasets × three backends.
- [`visualize_test_embeddings.py`](examples/visualize_test_embeddings.py) —
  single-dataset comparison of all backends on a held-out test split.
- [`umap_visualization.py`](examples/umap_visualization.py) — minimal
  full-dataset UMAP.

## Roadmap

- More backends as they expose embeddings
- Benchmarks for the `y=None` pseudo-target strategy

## Related work

- [`TabPFNEmbedding`](https://github.com/PriorLabs/tabpfn-extensions) in
  tabpfn-extensions — embedding extraction for TabPFN with out-of-fold support.
- [TabICL PR (in progress)](https://github.com/soda-inria/tabicl) adding native
  `get_row_embeddings()` support, which will replace the hook-based extraction
  used here.

## License

MIT
