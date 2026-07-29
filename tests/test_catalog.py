"""Tests for the analysis catalogue — extractor, store, facade, portability.

Synthetic bundles (hand-written recipe.json + report.md) keep these pure and fast: no real
data, no DataSet, no snapshot. That mirrors the catalogue's own pure metadata read path.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_core.adapters import session_bundle
from dataset_core.adapters.catalog import Catalog, JsonCatalogStore, extract_record
from dataset_core.adapters.catalog.config import resolve_catalog_root, ENV_VAR_NAME
from dataset_core.adapters.catalog.record import CatalogRecord
from dataset_core.adapters.catalog.store import default_catalog_path


def _write_bundle(
    root, name, *, steps, config=None, series_name=None, created="2026-07-01T00:00:00+00:00",
    bundle_id=None, notes="", metadata=None, n_files=None, fit_model=None, flag_name=None,
):
    """Create a synthetic .thzbundle (recipe.json + optional report.md) and return its dir."""
    bundle_dir = os.path.join(root, name + ".thzbundle")
    os.makedirs(bundle_dir, exist_ok=True)
    recipe = {
        "schema_version": "1.0",
        "bundle_id": bundle_id,
        "created": created,
        "source_dir": os.path.join(root, name + "_raw"),
        "series_name": series_name or name,
        "notes": notes,
        "metadata": metadata or {},
        "n_files": n_files,
        "config": config or {},
        "steps": [{"stage": stage, "kwargs": kwargs} for stage, kwargs in steps],
    }
    with open(os.path.join(bundle_dir, "recipe.json"), "w", encoding="utf-8") as f:
        json.dump(recipe, f)
    if fit_model or flag_name:
        lines = ["# Processing report", "", "## Fits"]
        if fit_model:
            lines.append(f"- **sample.acc** — `{fit_model}`  (R² = 0.9900)")
        else:
            lines.append("_No fits stored._")
        lines += ["", "## Per-stage diagnostics (raised flags / warnings)"]
        if flag_name:
            lines.append(f"  - `transfer_metrics` flag **{flag_name}**")
        with open(os.path.join(bundle_dir, "report.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    return bundle_dir


# ── root resolution ────────────────────────────────────────────────────────────


class TestRootResolution(unittest.TestCase):
    def test_explicit_wins(self):
        self.assertEqual(
            resolve_catalog_root(r"D:/some/root"), os.path.abspath(r"D:/some/root")
        )

    def test_env_var_used(self):
        original = os.environ.get(ENV_VAR_NAME)
        try:
            os.environ[ENV_VAR_NAME] = r"E:/env/root"
            self.assertEqual(resolve_catalog_root(), os.path.abspath(r"E:/env/root"))
        finally:
            if original is None:
                os.environ.pop(ENV_VAR_NAME, None)
            else:
                os.environ[ENV_VAR_NAME] = original


# ── the extractor (pure metadata derivation) ────────────────────────────────────


class TestExtractor(unittest.TestCase):
    def test_transmission(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = _write_bundle(
                root, "trans", n_files=3,
                steps=[("invert_nk", {"thickness_m": 5e-4}), ("derive_eps_sigma", {})],
            )
            record = extract_record(bundle, root)
        self.assertEqual(record.measurement_type, "transmission")
        self.assertEqual(record.quantities, ["n", "k", "sigma"])
        self.assertIsNone(record.geometry_detail)
        self.assertEqual(record.n_files, 3)

    def test_reflection_window_with_polarization(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = _write_bundle(
                root, "refl",
                config={"geometry": {"polarization": "p"}},
                steps=[("invert_nk_reflection", {"geometry": "window"})],
            )
            record = extract_record(bundle, root)
        self.assertEqual(record.measurement_type, "reflection")
        self.assertEqual(record.geometry_detail, "window")
        self.assertEqual(record.polarization, "p")

    def test_gold_reference(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = _write_bundle(
                root, "gold",
                steps=[("invert_nk_reflection", {"geometry": "gold"})],
            )
            record = extract_record(bundle, root)
        self.assertEqual(record.measurement_type, "gold")

    def test_fits_and_flags_from_report(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = _write_bundle(
                root, "fitted",
                steps=[("invert_nk", {})],
                fit_model="drude_joint", flag_name="low_dynamic_range",
            )
            record = extract_record(bundle, root)
        self.assertEqual(record.fit_models, ["drude_joint"])
        self.assertIn("fit", record.quantities)
        self.assertEqual(record.flags_raised, ["low_dynamic_range"])

    def test_relative_path_is_forward_slashed(self):
        with tempfile.TemporaryDirectory() as root:
            nested = os.path.join(root, "sub")
            bundle = _write_bundle(nested, "deep", steps=[("invert_nk", {})])
            record = extract_record(bundle, root)
        self.assertEqual(record.relative_path, "sub/deep.thzbundle")
        self.assertNotIn("\\", record.relative_path)

    def test_missing_bundle_id_is_deterministic(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = _write_bundle(root, "noid", bundle_id=None, steps=[("invert_nk", {})])
            first = extract_record(bundle, root).bundle_id
            second = extract_record(bundle, root).bundle_id
        self.assertEqual(first, second)  # stable fallback id from relative path

    def test_explicit_bundle_id_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = _write_bundle(root, "hasid", bundle_id="abc-123", steps=[("invert_nk", {})])
            self.assertEqual(extract_record(bundle, root).bundle_id, "abc-123")

    def test_odd_encoded_report_does_not_crash(self):
        # A report.md written in a non-UTF-8 encoding (cp1252 em-dash = 0x97) must not raise.
        with tempfile.TemporaryDirectory() as root:
            bundle = _write_bundle(root, "cp1252", steps=[("invert_nk", {})])
            with open(os.path.join(bundle, "report.md"), "wb") as report_file:
                report_file.write(b"## Fits\n- **s.acc** \x97 `drude`  (R = 0.99)\n")  # raw cp1252 em-dash
            record = extract_record(bundle, root)  # must not raise
        self.assertEqual(record.measurement_type, "transmission")

    def test_sample_tags_from_series_and_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = _write_bundle(
                root, "run", series_name="2026-07-03_silicon_ntype",
                metadata={"operator": "Sam"}, steps=[("invert_nk", {})],
            )
            record = extract_record(bundle, root)
        self.assertIn("silicon", record.sample_tags)
        self.assertIn("Sam", record.sample_tags)


# ── store round-trip ────────────────────────────────────────────────────────────


class TestJsonStore(unittest.TestCase):
    def _record(self, bundle_id="id1", **overrides):
        base = dict(
            bundle_id=bundle_id, relative_path=f"{bundle_id}.thzbundle", series_name="s",
            source_dir=None, created="2026-07-01", git_sha=None, notes="", measurement_type="reflection",
            polarization=None, geometry_detail=None, sample_tags=[], n_files=1, quantities=["n"],
            fit_models=[], flags_raised=[], indexed_at="2026-07-01", catalog_schema_version="1.0",
        )
        base.update(overrides)
        return CatalogRecord(**base)

    def test_upsert_get_all_and_reload(self):
        with tempfile.TemporaryDirectory() as root:
            path = default_catalog_path(root)
            store = JsonCatalogStore(path)
            store.upsert(self._record("a"))
            store.upsert(self._record("b"))
            store.upsert(self._record("a", notes="updated"))  # same id overwrites
            self.assertEqual(len(store.all()), 2)
            self.assertEqual(store.get("a").notes, "updated")

            # A fresh store reading the same file sees the persisted records.
            reloaded = JsonCatalogStore(path)
            self.assertEqual(len(reloaded.all()), 2)

    def test_replace_all_and_remove(self):
        with tempfile.TemporaryDirectory() as root:
            store = JsonCatalogStore(default_catalog_path(root))
            store.replace_all([self._record("x"), self._record("y")])
            self.assertEqual({r.bundle_id for r in store.all()}, {"x", "y"})
            store.remove("x")
            self.assertEqual({r.bundle_id for r in store.all()}, {"y"})


# ── facade: rebuild / verify / find / portability ───────────────────────────────


class TestCatalogFacade(unittest.TestCase):
    def _populate(self, root):
        _write_bundle(root, "si_trans", series_name="silicon_trans", created="2026-06-15T00:00:00",
                      steps=[("invert_nk", {}), ("derive_eps_sigma", {})])
        _write_bundle(root, "cnt_refl", series_name="CNT21_reflection", created="2026-07-10T00:00:00",
                      config={"geometry": {"polarization": "p"}},
                      steps=[("invert_nk_reflection", {"geometry": "window"})],
                      fit_model="drude_joint")
        _write_bundle(root, "cnt_gold", series_name="CNT21_gold", created="2026-07-11T00:00:00",
                      steps=[("invert_nk_reflection", {"geometry": "gold"})], flag_name="mirror_like")

    def test_rebuild_discovers_all(self):
        with tempfile.TemporaryDirectory() as root:
            self._populate(root)
            catalog = Catalog(root=root)
            self.assertEqual(catalog.rebuild(), 3)
            self.assertEqual(len(catalog.list_all()), 3)

    def test_find_filters(self):
        with tempfile.TemporaryDirectory() as root:
            self._populate(root)
            catalog = Catalog(root=root)
            catalog.rebuild()

            self.assertEqual(len(catalog.find(measurement_type="reflection")), 1)
            self.assertEqual(len(catalog.find(measurement_type="gold")), 1)
            self.assertEqual(len(catalog.find(sample="CNT")), 2)
            self.assertEqual(len(catalog.find(since="2026-07")), 2)
            self.assertEqual(len(catalog.find(polarization="p")), 1)
            self.assertEqual(len(catalog.find(has_fits=True)), 1)
            self.assertEqual(len(catalog.find(has_flags=True)), 1)
            self.assertEqual(len(catalog.find(text="silicon")), 1)

    def test_verify_flags_missing(self):
        with tempfile.TemporaryDirectory() as root:
            self._populate(root)
            catalog = Catalog(root=root)
            catalog.rebuild()
            self.assertEqual(catalog.verify(), [])

            import shutil
            shutil.rmtree(os.path.join(root, "cnt_gold.thzbundle"))
            missing = catalog.verify()
            self.assertEqual(missing, ["cnt_gold.thzbundle"])

    def test_survives_root_move(self):
        # Records store paths RELATIVE to root, so moving the whole root keeps them resolvable.
        with tempfile.TemporaryDirectory() as parent:
            root_a = os.path.join(parent, "data_a")
            os.makedirs(root_a)
            self._populate(root_a)
            Catalog(root=root_a).rebuild()  # writes root_a/_thz_catalog.json (moves with the dir)

            root_b = os.path.join(parent, "data_b")
            os.rename(root_a, root_b)

            moved = Catalog(root=root_b)
            self.assertEqual(len(moved.list_all()), 3)
            self.assertEqual(moved.verify(), [])  # every bundle still resolves under the new root
            resolved = moved.resolve_bundle_dir(moved.list_all()[0])
            self.assertTrue(os.path.exists(os.path.join(resolved, "recipe.json")))

    def test_rebuild_skips_malformed_bundle(self):
        with tempfile.TemporaryDirectory() as root:
            self._populate(root)
            # A corrupt bundle (invalid recipe.json) must not abort the rebuild.
            broken = os.path.join(root, "broken.thzbundle")
            os.makedirs(broken)
            with open(os.path.join(broken, "recipe.json"), "w", encoding="utf-8") as f:
                f.write("{ not valid json")
            catalog = Catalog(root=root)
            self.assertEqual(catalog.rebuild(), 3)  # the 3 good bundles, broken one skipped

    def test_injected_store(self):
        # Dependency injection: the facade accepts any CatalogStore.
        with tempfile.TemporaryDirectory() as root:
            self._populate(root)
            store = JsonCatalogStore(os.path.join(root, "custom_index.json"))
            catalog = Catalog(root=root, store=store)
            catalog.rebuild()
            self.assertTrue(os.path.exists(os.path.join(root, "custom_index.json")))
            self.assertEqual(len(store.all()), 3)


# ── save hook + portability helpers ─────────────────────────────────────────────


class TestSaveHookAndPortability(unittest.TestCase):
    def test_index_in_catalog_is_non_fatal(self):
        # A broken bundle path must not raise out of the save hook.
        try:
            session_bundle._index_in_catalog(os.path.join(tempfile.gettempdir(), "does_not_exist.thzbundle"))
        except Exception as error:
            self.fail(f"_index_in_catalog should swallow errors, raised {error!r}")

    def test_index_in_catalog_adds_entry(self):
        original = os.environ.get(ENV_VAR_NAME)
        with tempfile.TemporaryDirectory() as root:
            try:
                os.environ[ENV_VAR_NAME] = root
                bundle = _write_bundle(root, "run", steps=[("invert_nk", {})])
                session_bundle._index_in_catalog(bundle)
                self.assertEqual(len(Catalog(root=root).list_all()), 1)
            finally:
                if original is None:
                    os.environ.pop(ENV_VAR_NAME, None)
                else:
                    os.environ[ENV_VAR_NAME] = original

    def test_pack_and_unpack_roundtrip(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = _write_bundle(root, "portable", steps=[("invert_nk", {})])
            zip_path = session_bundle.pack_bundle(bundle)
            self.assertTrue(os.path.exists(zip_path))

            dest = os.path.join(root, "unpacked")
            restored = session_bundle.unpack_bundle(zip_path, dest)
            self.assertTrue(os.path.exists(os.path.join(restored, "recipe.json")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
