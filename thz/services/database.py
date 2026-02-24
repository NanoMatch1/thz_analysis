
import json
from pathlib import Path
from pyparsing import Optional

@classmethod
class DatabaseService:
    DEFAULT_REGISTRY_NAME = "thz_series_registry.json"

    def export_h5(
        self,
        path: str | Path,
        *,
        dataset_metadata: Optional[Dict[str, Any]] = None,
        per_file_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        compression: Optional[str] = "gzip",
        compression_opts: int = 1,
        store_acq_mask: bool = True,
        registry_path: Optional[str | Path] = None,
        update_registry: bool = True,
    ) -> Path:
        """
        Export the entire DataSet into a single HDF5 file.

        - dataset_metadata: dynamic dict stored at root (merged with self.metadata).
        - per_file_metadata: optional dict keyed by filename with dynamic dict values.
        - metadata_hook(): called for each file and merged into that file’s metadata (so you can implement later).
        - store_acq_mask:
            If True, stores an acquisition mask inferred from raw_data shape (all True),
            unless you supply a mask inside per-file metadata (recommended later).
            (You can extend to store real masks once you track them.)

        Registry:
          If update_registry=True, writes/updates a JSON registry that maps seriesname -> last_export_path.
        """
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        root_meta = {}
        root_meta.update(self.metadata or {})
        root_meta.update(dataset_metadata or {})

        per_file_metadata = per_file_metadata or {}

        with h5py.File(path, "w") as f:
            # Root attributes
            f.attrs["schema_version"] = "1.0"
            f.attrs["export_timestamp_utc"] = _utc_now_iso()
            f.attrs["seriesname"] = self.seriesname or ""
            f.attrs["dataset_metadata_json"] = _json_dumps(root_meta)

            g_files = f.create_group("files")

            for filename, thzdata in self.data.items():
                key = _safe_hdf5_key(filename)
                g = g_files.create_group(key)
                g.attrs["original_filename"] = filename

                raw = np.asarray(thzdata.raw_data)
                # Keep dtype; or you can force float32 to reduce size:
                # raw = raw.astype(np.float32, copy=False)

                # Decide compression (tiny arrays -> compression overhead is fine; keep consistent)
                create_kw = {}
                if compression:
                    create_kw.update(
                        compression=compression,
                        compression_opts=compression_opts,
                        shuffle=True,
                        # chunks=True lets h5py pick a reasonable chunking; fine for your scale
                        chunks=True,
                    )

                g.create_dataset("raw_data", data=raw, **create_kw)

                # Dynamic metadata per file:
                file_meta = {}
                file_meta.update(per_file_metadata.get(filename, {}) or {})
                # Merge hook metadata (hook can override/add keys)
                try:
                    hook_meta = self.metadata_hook(filename, thzdata) or {}
                except Exception as e:
                    hook_meta = {"_metadata_hook_error": str(e)}
                file_meta.update(hook_meta)

                g.attrs["file_metadata_json"] = _json_dumps(file_meta)

                # Optional acquisition mask
                if store_acq_mask:
                    # By convention: raw_data[:, 0] is time; acquisitions start at col 1
                    ncols = raw.shape[1] if raw.ndim == 2 else 0
                    n_acq = max(0, ncols - 1)
                    # If you later store a real mask in file_meta, prefer it:
                    mask = file_meta.get("acq_mask", None)
                    if mask is None:
                        mask_arr = np.ones((n_acq,), dtype=bool)
                    else:
                        mask_arr = np.asarray(mask, dtype=bool)
                        if mask_arr.shape != (n_acq,):
                            # store it anyway, but also record mismatch for debugging
                            g.attrs["acq_mask_shape_warning"] = (
                                f"mask shape {mask_arr.shape} != expected {(n_acq,)}"
                            )
                    g.create_dataset("acq_mask", data=mask_arr)

        if update_registry:
            self._update_registry(
                seriesname=self.seriesname,
                export_path=str(path),
                registry_path=registry_path,
            )

        return path

    @classmethod
    def load_h5(
        cls,
        path: str | Path,
        *,
        registry_path: Optional[str | Path] = None,
        seriesname: Optional[str] = None,
        strict_schema: bool = False,
    ) -> "DataSet":
        """
        Load a DataSet from a single HDF5 file.

        - If seriesname is None, will use the HDF5 stored 'seriesname' if present.
        - strict_schema=True will raise if schema_version is unexpected.
        """
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"HDF5 not found: {path}")

        with h5py.File(path, "r") as f:
            schema = f.attrs.get("schema_version", "")
            if strict_schema and schema not in ("1.0",):
                raise ValueError(f"Unsupported schema_version: {schema!r}")

            stored_series = f.attrs.get("seriesname", "")
            use_series = seriesname if seriesname is not None else stored_series

            dataset_meta = _json_loads(f.attrs.get("dataset_metadata_json", ""))

            data: Dict[str, THzData] = {}

            if "files" not in f:
                # empty dataset
                return cls(seriesname=use_series, data=data, metadata=dataset_meta)

            g_files = f["files"]
            for key in g_files.keys():
                g = g_files[key]
                filename = g.attrs.get("original_filename", key)

                raw = np.array(g["raw_data"])  # loads into memory (fast for your sizes)
                thz = THzData(raw_data=raw)

                # If you want, you can also load file_metadata_json and store it somewhere.
                # For now we keep metadata dynamic at dataset-level; you can extend.
                data[filename] = thz

        ds = cls(seriesname=use_series, data=data, metadata=dataset_meta)

        # Optionally register this load as "last opened" for the series
        if use_series:
            ds._update_registry(
                seriesname=use_series,
                last_opened_path=str(path),
                registry_path=registry_path,
            )

        return ds

    # ---------------------------------------------------------------------
    # Registry helpers: seriesname -> paths (exports/last_opened, etc.)
    # ---------------------------------------------------------------------
    def _registry_default_path(self) -> Path:
        """
        Where to store the registry if not provided.
        Chosen to be stable but user-visible. Adjust if you prefer e.g. ~/.config/...
        """
        # Store next to the current working directory by default:
        return Path.cwd() / self.DEFAULT_REGISTRY_NAME

    def _update_registry(
        self,
        *,
        seriesname: str,
        registry_path: Optional[str | Path] = None,
        export_path: Optional[str] = None,
        last_opened_path: Optional[str] = None,
    ) -> Path:
        """
        Update the JSON registry with paths associated with a seriesname.
        """
        if not seriesname:
            # If the seriesname is empty, don’t write registry entries.
            return Path(registry_path).expanduser().resolve() if registry_path else self._registry_default_path()

        reg_path = Path(registry_path).expanduser().resolve() if registry_path else self._registry_default_path()
        reg_path.parent.mkdir(parents=True, exist_ok=True)

        if reg_path.exists():
            try:
                registry = json.loads(reg_path.read_text(encoding="utf-8"))
            except Exception:
                registry = {}
        else:
            registry = {}

        entry = registry.get(seriesname, {})
        entry["seriesname"] = seriesname
        entry["updated_utc"] = _utc_now_iso()
        if export_path is not None:
            entry["last_export_path"] = export_path
        if last_opened_path is not None:
            entry["last_opened_path"] = last_opened_path

        registry[seriesname] = entry
        reg_path.write_text(_json_dumps(registry), encoding="utf-8")
        return reg_path


    def resolve_series_from_registry(
            cls,
            seriesname: str,
            *,
            registry_path: Optional[str | Path] = None,
            prefer: str = "last_export_path",
        ) -> Optional[Path]:
            """
            Look up a seriesname in the registry and return its stored path.

            prefer: "last_export_path" or "last_opened_path"
            """
            reg_path = Path(registry_path).expanduser().resolve() if registry_path else (Path.cwd() / cls.DEFAULT_REGISTRY_NAME)
            if not reg_path.exists():
                return None
            try:
                registry = json.loads(reg_path.read_text(encoding="utf-8"))
            except Exception:
                return None

            entry = registry.get(seriesname)
            if not entry:
                return None
            p = entry.get(prefer) or entry.get("last_export_path") or entry.get("last_opened_path")
            return Path(p).expanduser().resolve() if p else None