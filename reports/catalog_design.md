# Catalog design — a rebuildable index over `.thzbundle` analyses

**Status:** IMPLEMENTED (2026-07-21). Design below; see §13 for the as-built notes.
**Author context:** Samuel + Claude. Supersedes the `save_database` flat-text index and the
ad-hoc `database_viewer.py` for the "find my saved analyses" use case.

---

## 1. Problem & goal

Analyses are saved as self-describing `.thzbundle` directories (`session_bundle.py`:
`recipe.json` + `snapshot.pkl` + `report.md`). They are excellent *save* units but there is
**no index** — a bundle is findable only if you remember its path. Goal: a single catalogue you
consult to find "which directory holds which analysis" by querying metadata (measurement type,
sample, date, notes, has-fits, flags), without remembering paths.

Three older persistence paths exist and overlap:

| Mechanism | File | Role after this work |
|---|---|---|
| `.thzbundle` (session_bundle) | `dataset_core/adapters/session_bundle.py` | **Canonical save unit** — the thing the catalogue indexes |
| pickle-db + `_database_index.txt` | `dataset.py::save_database`/`load_database` | **Deprecate** — its indexing role is exactly what the catalogue replaces, better |
| `_state.pkl` local scratch | `dataset.py::save_state`/`load_state` | **Keep** — different purpose (fast local reload cache, no catalogue) |
| HDF5 export + `thz_series_registry.json` | `services/database.py::DatabaseService` | **Keep as interchange** (arrays + metadata for other tools); not the catalogue's job |

Portability that `save_database` uniquely gave ("single file to carry to another PC") is
recovered by a `pack_bundle`/`unpack_bundle` zip helper on the bundle — one format, not two.

## 2. Design principles applied

- **Catalogue = derived, rebuildable index, not a source of record.** Bundles are self-describing,
  so `rebuild(root)` walks the disk and reconstructs the whole index. Lose/corrupt/move it → one
  rebuild fixes it. This is the property the flat txt index lacks (it silently drifts). (Repo
  philosophy #4: self-describing, replayable artifacts.)
- **Pure metadata read path.** The extractor + query layers read only `recipe.json`/`report.md`,
  never load `snapshot.pkl`, never import `DataSet`. Only `Catalog.open()` imports `DataSet`, and
  lazily. (Repo philosophy #1: pure-core / thin-adapter — querying must not pull in the runtime.)
- **Single place to add a searchable field.** One `extract_record()` maps bundle → record; a new
  query dimension is a one-line edit there, not a parallel schema change. (Registry discipline.)
- **Anti-brittleness by construction:** configured root (not hardcoded), catalogue file lives *at*
  the root, entries store paths *relative* to the root, plus a stable `bundle_id`. Move the root
  (new drive/PC/folder name) → still resolves; rename a bundle within the root → `bundle_id` still
  identifies it.
- **Injected store** (`CatalogStore` protocol) so JSON now / SQLite later is a constructor swap.
- **Non-fatal integration:** a catalogue failure must never break a `save_session`.

## 3. Module layout

```
dataset_core/adapters/catalog/
    __init__.py     # public re-exports: Catalog, CatalogRecord, resolve_catalog_root
    record.py       # CatalogRecord dataclass + extract_record()   ← the ONE extractor
    store.py        # CatalogStore protocol + JsonCatalogStore  (SqliteCatalogStore later)
    config.py       # resolve_catalog_root()
    facade.py       # Catalog: update / rebuild / verify / find / open
catalog_browse.py   # root-level CLI/viewer (retires database_viewer.py)
```

## 4. Record schema

Lightweight, all queryable, **no arrays**.

```python
@dataclass(frozen=True)
class CatalogRecord:
    bundle_id: str            # uuid, stamped into recipe.json at save (stable across move/rename)
    relative_path: str        # bundle dir RELATIVE to catalog root  ← the resolvable pointer
    series_name: str | None
    source_dir: str | None    # raw-data origin (informational only)
    created: str | None       # from recipe
    git_sha: str | None
    notes: str
    measurement_type: str     # 'transmission'|'reflection'|'gold'|'unknown'  (derived from steps)
    polarization: str | None  # 's'|'p'  (config['geometry'])
    geometry_detail: str | None   # 'window'|'gold'|...  (invert stage)
    sample_tags: list[str]    # from metadata / file_metadata
    n_files: int
    quantities: list[str]     # subset of ['n','k','sigma','fit']  (derived from which stages ran)
    fit_models: list[str]     # from report.md "## Fits"
    flags_raised: list[str]   # from report.md diagnostics
    indexed_at: str
    catalog_schema_version: str
```

Notes on two derivations (reality-checked against the repo):
- **`measurement_type` / `quantities` come from `recipe["steps"]`, not config.** Transmission vs
  reflection is carried by *which invert stage ran* (`invert_nk` vs `invert_nk_reflection` with
  `geometry='window'|'gold'`); σ by whether the conductivity stage ran. `config['geometry']` only
  holds `polarization`/`theta_external_deg`/`r_reference`. Reading steps keeps this a pure-metadata
  read (no snapshot load).
- **`relative_path` + `bundle_id`, never an absolute path** — the whole anti-brittleness lever.

## 5. The extractor (single source of searchable fields)

```python
def extract_record(bundle_dir: str, root: str) -> CatalogRecord:
    """Build a record from recipe.json (+ report.md). Never loads snapshot.pkl, never imports
    DataSet. Adding a new queryable field = editing THIS function only."""
```

## 6. Store (injected; JSON now, SQLite later)

```python
class CatalogStore(Protocol):
    def upsert(self, record: CatalogRecord) -> None
    def remove(self, bundle_id: str) -> None
    def get(self, bundle_id: str) -> CatalogRecord | None
    def all(self) -> list[CatalogRecord]
    def replace_all(self, records: Iterable[CatalogRecord]) -> None   # rebuild

class JsonCatalogStore(CatalogStore):
    def __init__(self, path: str): ...   # <root>/_thz_catalog.json — human-readable, git-diffable
```

Volume is dozens–hundreds of runs → a single JSON is right; SQLite is premature. The protocol lets
us swap later without touching the facade.

## 7. Facade (the public entry point)

```python
class Catalog:
    def __init__(self, root: str | None = None, store: CatalogStore | None = None):
        self.root  = resolve_catalog_root(root)
        self.store = store or JsonCatalogStore(os.path.join(self.root, "_thz_catalog.json"))

    def update(self, bundle_dir: str) -> CatalogRecord   # upsert one bundle (called after save)
    def rebuild(self) -> int                             # glob **/*.thzbundle under root -> replace_all
    def verify(self) -> list[str]                        # relative_paths whose bundle is now missing
    def find(self, **filters) -> list[CatalogRecord]     # query (see below)
    def list_all(self) -> list[CatalogRecord]
    def open(self, record_or_id) -> "DataSet"            # ONLY method importing DataSet, lazily
```

`find` filters (all optional, AND-combined):
`measurement_type=`, `polarization=`, `sample=`, `since=` / `until=` (date),
`notes_contains=`, `has_fits=`, `has_flags=`, `text=` (free-text over series/notes/tags).

## 8. Root resolution (non-brittle)

```python
def resolve_catalog_root(explicit: str | None = None) -> str:
    # 1. explicit arg
    # 2. env  THZ_CATALOG_ROOT
    # 3. config file  ~/.thz/catalog.toml   (root = "...")
    # 4. default  C:/Users/Samuel/Data/THz  (documented default, not scattered through code)
```

Catalogue file lives **at** the root (`<root>/_thz_catalog.json`) → "find the catalogue" = "find
the root", one thing to know. `~/.thz/` is a dedicated per-user THz config dir (room for other
tools — SFG etc. — later).

## 9. Integration & portability (small, additive)

- **Save hook** in `save_session`: after writing the bundle, best-effort
  `Catalog().update(bundle_dir)`, wrapped so a catalogue failure **never** breaks the save.
- **Stable id:** `_build_recipe` stamps a `bundle_id` uuid into `recipe.json` (re-used on re-save).
- **Portability** (the `save_database` single-file replacement): `pack_bundle(bundle_dir) -> zip`
  and `unpack_bundle(zip, dest)` in `session_bundle`.
- **CLI** `catalog_browse.py`: `--find`, `--rebuild`, `--verify`, `--open <id>`
  (`--open` → `load_session` + `launch_results_viewer`). Retires `database_viewer.py`.

## 10. Tests

- **Unit:** `extract_record` on synthetic transmission & reflection bundles (correct
  type/quantities/flags/polarization); `JsonCatalogStore` upsert/get/all/replace_all round-trip;
  **root-move survival** (build catalogue → move the root dir → entries still resolve via relative
  paths); `rebuild` reconstructs from disk; `verify` flags a deleted bundle; `find` filter matrix.
- **Integration:** `save_session` triggers an `update`; a broken/unwritable catalogue leaves the
  save intact (non-fatal path exercised).

## 11. Deprecation plan (staged, non-breaking)

1. Ship catalogue + tests (indexes bundles). No behaviour change to existing saves beyond the
   added `bundle_id` and the best-effort update hook.
2. Add `pack_bundle`/`unpack_bundle`; document as the portable-single-file path.
3. Mark `save_database`/`load_database` deprecated (docstring + one-time runtime warning), point to
   catalogue + bundles. Keep `save_state`/`load_state` and the HDF5 interchange path untouched.
4. Remove `database_viewer.py` once `catalog_browse.py` covers its use.

## 12. Open questions / deferred

- Free-text search is substring-based initially; revisit if we want fielded/fuzzy search.
- SQLite backing store only if run count grows past comfortable JSON territory.
- Ties into the separate **coupling/dependency audit** TODO: confirm the pure-metadata read path
  really needs no `DataSet` import (it shouldn't) — the catalogue is a good forcing function for it.

## 13. As-built notes (2026-07-21)

Implemented exactly as designed. Files: `dataset_core/adapters/catalog/{config,record,store,facade,__init__}.py`,
CLI `catalog_browse.py`; `session_bundle.py` gained `bundle_id`/`metadata`/`n_files` in the recipe,
the non-fatal `_index_in_catalog` hook, and `pack_bundle`/`unpack_bundle`. Tests: `tests/test_catalog.py`
(22). Full suite 65 pass.

Two refinements beyond the doc, both from the CLI full-workflow smoke test:
- **Auto-index is scoped to the root.** `_index_in_catalog` only indexes bundles saved *under* the
  catalogue root (via `Path.is_relative_to`); bundles saved elsewhere (temp/ad-hoc) are left for an
  explicit `rebuild`. Prevents temp saves polluting the managed index.
- **Robustness on read.** `report.md` is read with `errors="replace"` and `rebuild` skips-and-warns
  on any single malformed bundle instead of aborting — a batch scan must survive one bad run. (The
  smoke test caught a real UnicodeDecodeError that strict UTF-8 reading would have crashed on.)

Not yet done (design §11 staging): deprecation warning on `save_database`/`load_database`; removing
`database_viewer.py`. Left for a follow-up so this change stays additive.
