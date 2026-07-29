"""Catalogue: a rebuildable index over ``.thzbundle`` analyses.

See ``reports/catalog_design.md`` for the design rationale.

    from dataset_core.adapters.catalog import Catalog
    catalog = Catalog()                       # root from env / ~/.thz/catalog.toml / default
    catalog.rebuild()                          # scan the root for bundles
    for record in catalog.find(measurement_type="reflection", sample="CNT"):
        print(record.relative_path)
    dataset = catalog.open(record)             # load the saved results (no recompute)
"""

from dataset_core.adapters.catalog.config import resolve_catalog_root
from dataset_core.adapters.catalog.facade import Catalog
from dataset_core.adapters.catalog.query import FILTER_KEYS, parse_filter_tokens
from dataset_core.adapters.catalog.record import CatalogRecord, extract_record
from dataset_core.adapters.catalog.store import (
    CatalogStore,
    JsonCatalogStore,
    default_catalog_path,
)

__all__ = [
    "Catalog",
    "CatalogRecord",
    "CatalogStore",
    "JsonCatalogStore",
    "extract_record",
    "resolve_catalog_root",
    "default_catalog_path",
    "FILTER_KEYS",
    "parse_filter_tokens",
]
