"""ACC File Manager — join and split .acc THz data files.

A tkinter-based file browser that lets you:

  - **Join**: append scans from one or more ``.acc`` files onto a primary
    file, producing a single combined ``.acc``.
  - **Split**: partition the scans of one ``.acc`` file into two separate
    files, either at a fixed position or by even/odd interleaving.

Reuses the load/save machinery from ``acquisition_editor``.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from pathlib import Path
from typing import Optional

import numpy as np

import acquisition_editor


# ---------------------------------------------------------------------------
# Data manipulation helpers
# ---------------------------------------------------------------------------

def _join_acc_dicts(dicts: list[dict]) -> dict:
    """Concatenate scans from multiple acc dicts, appending to the first.

    All files must share the same time-axis length.  Headers from the primary
    file are kept; per-scan headers from all files are concatenated.
    """
    if not dicts:
        raise ValueError("No files provided to join.")

    primary = dicts[0]
    time_axis = primary["data"][:, 0]
    n_points = len(time_axis)

    combined_signal_blocks: list[np.ndarray] = [primary["data"][:, 1:]]
    combined_scan_headers: list[list[str]] = list(primary.get("scan_headers", []))

    for other in dicts[1:]:
        if other["data"].shape[0] != n_points:
            raise ValueError(
                f"Cannot join '{other['filename']}' with '{primary['filename']}': "
                f"time-axis length differs ({other['data'].shape[0]} vs {n_points} points)."
            )
        combined_signal_blocks.append(other["data"][:, 1:])
        combined_scan_headers.extend(other.get("scan_headers", []))

    all_signals = np.concatenate(combined_signal_blocks, axis=1)
    combined_data = np.column_stack([time_axis, all_signals])

    return {
        "filename": primary["filename"],
        "header": primary.get("header", []),
        "scan_headers": combined_scan_headers,
        "data": combined_data,
    }


def _split_acc_at_position(data_dict: dict, n_first: int) -> tuple[dict, dict]:
    """Split scans at a position: first *n_first* → part A, remainder → part B."""
    data = data_dict["data"]
    n_scans = data.shape[1] - 1
    n_first = max(1, min(n_first, n_scans - 1))

    time = data[:, 0]
    scan_headers = data_dict.get("scan_headers", [])

    data_a = np.column_stack([time, data[:, 1 : n_first + 1]])
    data_b = np.column_stack([time, data[:, n_first + 1 :]])

    part_a = {**data_dict, "data": data_a, "scan_headers": scan_headers[:n_first]}
    part_b = {**data_dict, "data": data_b, "scan_headers": scan_headers[n_first:]}
    return part_a, part_b


def _split_acc_even_odd(data_dict: dict) -> tuple[dict, dict]:
    """Split into odd-indexed scans (1,3,5,…) → A and even-indexed (2,4,6,…) → B."""
    data = data_dict["data"]
    n_scans = data.shape[1] - 1
    time = data[:, 0]
    scan_headers = data_dict.get("scan_headers", [])

    # Column indices in data array are 1-based for scans
    odd_col_indices = [i for i in range(1, n_scans + 1) if i % 2 == 1]
    even_col_indices = [i for i in range(1, n_scans + 1) if i % 2 == 0]

    def _build(col_indices: list[int]) -> np.ndarray:
        if not col_indices:
            return data[:, [0]]
        return np.column_stack([time] + [data[:, c] for c in col_indices])

    data_a = _build(odd_col_indices)
    data_b = _build(even_col_indices)

    headers_a = [scan_headers[c - 1] for c in odd_col_indices if c - 1 < len(scan_headers)]
    headers_b = [scan_headers[c - 1] for c in even_col_indices if c - 1 < len(scan_headers)]

    part_a = {**data_dict, "data": data_a, "scan_headers": headers_a}
    part_b = {**data_dict, "data": data_b, "scan_headers": headers_b}
    return part_a, part_b


def _n_scans(data_dict: dict) -> int:
    data = data_dict.get("data")
    if data is None or data.ndim != 2:
        return 0
    return max(0, data.shape[1] - 1)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class AccFileManager(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("ACC File Manager")
        self.geometry("840x720")
        self.resizable(True, True)

        self._directory: Optional[Path] = None
        self._files: list[Path] = []

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # --- Directory row ---
        dir_frame = ttk.Frame(self, padding=(6, 6, 6, 2))
        dir_frame.pack(fill="x")
        ttk.Label(dir_frame, text="Directory:").pack(side="left")
        self._dir_var = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self._dir_var, width=58).pack(side="left", padx=4)
        ttk.Button(dir_frame, text="Browse…", command=self._browse_directory).pack(side="left")
        ttk.Button(dir_frame, text="Refresh", command=self._refresh_files).pack(side="left", padx=4)

        # --- Files list ---
        files_frame = ttk.LabelFrame(self, text="Files in directory (.acc)", padding=5)
        files_frame.pack(fill="both", expand=False, padx=6, pady=2)

        files_inner = ttk.Frame(files_frame)
        files_inner.pack(fill="both")
        files_scrollbar = ttk.Scrollbar(files_inner, orient="vertical")
        self._files_listbox = tk.Listbox(
            files_inner,
            selectmode="extended",
            height=7,
            exportselection=False,
            yscrollcommand=files_scrollbar.set,
        )
        files_scrollbar.configure(command=self._files_listbox.yview)
        self._files_listbox.pack(side="left", fill="both", expand=True)
        files_scrollbar.pack(side="right", fill="y")
        self._files_listbox.bind("<<ListboxSelect>>", self._on_file_select)

        # --- Operations notebook ---
        self._notebook = ttk.Notebook(self, padding=4)
        self._notebook.pack(fill="both", expand=True, padx=6, pady=4)

        self._build_join_tab()
        self._build_split_tab()

        # --- Log ---
        log_frame = ttk.LabelFrame(self, text="Log", padding=5)
        log_frame.pack(fill="x", padx=6, pady=(0, 6))
        self._log = scrolledtext.ScrolledText(log_frame, height=5, state="disabled", wrap="word")
        self._log.pack(fill="x")

    def _build_join_tab(self) -> None:
        tab = ttk.Frame(self._notebook, padding=10)
        self._notebook.add(tab, text="Join")

        ttk.Label(
            tab,
            text=(
                "Select files in the directory list above (Ctrl+click to multi-select), "
                "then click 'Use selection'. The first file is the primary; "
                "subsequent files are appended to it in the listed order."
            ),
            wraplength=760,
            justify="left",
        ).pack(anchor="w")

        # Queue list
        sel_frame = ttk.LabelFrame(tab, text="Files to join (in order)", padding=5)
        sel_frame.pack(fill="x", pady=6)

        sel_inner = ttk.Frame(sel_frame)
        sel_inner.pack(fill="x")
        self._join_listbox = tk.Listbox(sel_inner, height=4, exportselection=False)
        self._join_listbox.pack(side="left", fill="x", expand=True)
        btn_col = ttk.Frame(sel_inner)
        btn_col.pack(side="left", padx=4)
        ttk.Button(btn_col, text="↑ Up", width=9, command=lambda: self._move_join_item(-1)).pack(pady=2)
        ttk.Button(btn_col, text="↓ Down", width=9, command=lambda: self._move_join_item(1)).pack(pady=2)
        ttk.Button(btn_col, text="Remove", width=9, command=self._remove_join_item).pack(pady=2)

        ttk.Button(tab, text="Use selection from directory list ↑", command=self._populate_join_list).pack(anchor="w")

        # Output
        out_frame = ttk.Frame(tab)
        out_frame.pack(fill="x", pady=8)
        ttk.Label(out_frame, text="Output filename:").pack(side="left")
        self._join_output_var = tk.StringVar()
        ttk.Entry(out_frame, textvariable=self._join_output_var, width=40).pack(side="left", padx=4)
        ttk.Label(out_frame, text="(saved in same directory)", foreground="gray").pack(side="left")

        ttk.Button(tab, text="Join Files", command=self._do_join).pack(pady=4)

    def _build_split_tab(self) -> None:
        tab = ttk.Frame(self._notebook, padding=10)
        self._notebook.add(tab, text="Split")

        ttk.Label(
            tab,
            text="Select a single file in the directory list above, then choose how to split its scans.",
            wraplength=760,
            justify="left",
        ).pack(anchor="w")

        # Source file info
        info_frame = ttk.LabelFrame(tab, text="Source file", padding=5)
        info_frame.pack(fill="x", pady=6)
        self._split_source_var = tk.StringVar(value="(none selected)")
        ttk.Label(info_frame, textvariable=self._split_source_var, font=("", 10, "bold")).pack(anchor="w")
        self._split_info_var = tk.StringVar(value="")
        ttk.Label(info_frame, textvariable=self._split_info_var, foreground="gray").pack(anchor="w")

        # Split mode
        mode_frame = ttk.LabelFrame(tab, text="Split mode", padding=5)
        mode_frame.pack(fill="x", pady=4)

        self._split_mode = tk.StringVar(value="position")

        pos_frame = ttk.Frame(mode_frame)
        pos_frame.pack(anchor="w", pady=2)
        ttk.Radiobutton(
            pos_frame,
            text="Split at position: first",
            variable=self._split_mode,
            value="position",
            command=self._update_split_preview,
        ).pack(side="left")
        self._split_pos_var = tk.IntVar(value=1)
        self._split_pos_var.trace_add("write", lambda *_: self._update_split_preview())
        self._split_pos_spin = ttk.Spinbox(
            pos_frame,
            from_=1,
            to=1,
            textvariable=self._split_pos_var,
            width=6,
        )
        self._split_pos_spin.pack(side="left", padx=4)
        ttk.Label(pos_frame, text="scans → File A, remaining scans → File B").pack(side="left")

        eo_frame = ttk.Frame(mode_frame)
        eo_frame.pack(anchor="w", pady=2)
        ttk.Radiobutton(
            eo_frame,
            text="Even/odd interleave: odd scans (1, 3, 5, …) → File A, even scans (2, 4, 6, …) → File B",
            variable=self._split_mode,
            value="evenodd",
            command=self._update_split_preview,
        ).pack(side="left")

        # Preview
        preview_frame = ttk.LabelFrame(tab, text="Preview", padding=5)
        preview_frame.pack(fill="x", pady=4)
        self._split_preview_a = tk.StringVar(value="File A: —")
        self._split_preview_b = tk.StringVar(value="File B: —")
        ttk.Label(preview_frame, textvariable=self._split_preview_a).pack(anchor="w")
        ttk.Label(preview_frame, textvariable=self._split_preview_b).pack(anchor="w")

        # Output names
        out_frame = ttk.LabelFrame(tab, text="Output filenames (saved in same directory)", padding=5)
        out_frame.pack(fill="x", pady=4)

        a_row = ttk.Frame(out_frame)
        a_row.pack(fill="x", pady=2)
        ttk.Label(a_row, text="File A:", width=7).pack(side="left")
        self._split_out_a_var = tk.StringVar()
        ttk.Entry(a_row, textvariable=self._split_out_a_var, width=40).pack(side="left")

        b_row = ttk.Frame(out_frame)
        b_row.pack(fill="x", pady=2)
        ttk.Label(b_row, text="File B:", width=7).pack(side="left")
        self._split_out_b_var = tk.StringVar()
        ttk.Entry(b_row, textvariable=self._split_out_b_var, width=40).pack(side="left")

        ttk.Button(tab, text="Split File", command=self._do_split).pack(pady=4)

    # ------------------------------------------------------------------
    # Directory / file list
    # ------------------------------------------------------------------

    def _browse_directory(self) -> None:
        chosen = filedialog.askdirectory(title="Select directory containing .acc files")
        if chosen:
            self._dir_var.set(chosen)
            self._directory = Path(chosen)
            self._refresh_files()

    def _refresh_files(self) -> None:
        dir_text = self._dir_var.get().strip()
        if not dir_text:
            return
        self._directory = Path(dir_text)
        self._files_listbox.delete(0, "end")
        if not self._directory.is_dir():
            self._log_message(f"Not a valid directory: {self._directory}")
            return
        self._files = sorted(
            p for p in self._directory.iterdir()
            if p.is_file() and p.suffix.lower() == ".acc"
        )
        for f in self._files:
            self._files_listbox.insert("end", f.name)
        self._log_message(f"Found {len(self._files)} .acc file(s) in {self._directory}")

    def _on_file_select(self, _event=None) -> None:
        selected = self._files_listbox.curselection()
        if len(selected) == 1:
            self._update_split_source(self._files[selected[0]])

    # ------------------------------------------------------------------
    # Split tab helpers
    # ------------------------------------------------------------------

    def _update_split_source(self, filepath: Path) -> None:
        self._split_source_var.set(filepath.name)
        try:
            data_dict = acquisition_editor.load_file(filepath)
            n = _n_scans(data_dict)
            self._split_info_var.set(f"{n} scan(s), {data_dict['data'].shape[0]} points")
            self._split_pos_spin.configure(to=max(1, n - 1))
            self._split_pos_var.set(max(1, n // 2))
            stem = filepath.stem
            self._split_out_a_var.set(f"{stem}_A.acc")
            self._split_out_b_var.set(f"{stem}_B.acc")
            self._update_split_preview()
        except Exception as exc:
            self._split_info_var.set(f"Error reading file: {exc}")

    def _update_split_preview(self) -> None:
        info_text = self._split_info_var.get()
        if "scan(s)" not in info_text:
            return
        try:
            n_scans = int(info_text.split(" scan(s)")[0])
        except ValueError:
            return
        if n_scans < 2:
            self._split_preview_a.set("File A: —")
            self._split_preview_b.set("File B: —  (need at least 2 scans to split)")
            return

        mode = self._split_mode.get()
        if mode == "position":
            try:
                pos = int(self._split_pos_var.get())
            except (ValueError, tk.TclError):
                return
            pos = max(1, min(pos, n_scans - 1))
            self._split_preview_a.set(f"File A: scans 1–{pos}  ({pos} scan(s))")
            self._split_preview_b.set(f"File B: scans {pos + 1}–{n_scans}  ({n_scans - pos} scan(s))")
        elif mode == "evenodd":
            n_odd = (n_scans + 1) // 2
            n_even = n_scans // 2
            self._split_preview_a.set(f"File A: odd scans (1, 3, 5, …)  → {n_odd} scan(s)")
            self._split_preview_b.set(f"File B: even scans (2, 4, 6, …) → {n_even} scan(s)")

    # ------------------------------------------------------------------
    # Join tab helpers
    # ------------------------------------------------------------------

    def _populate_join_list(self) -> None:
        selected = self._files_listbox.curselection()
        if not selected:
            messagebox.showwarning("No selection", "Select files in the directory list above first.")
            return
        self._join_listbox.delete(0, "end")
        for idx in selected:
            self._join_listbox.insert("end", self._files[idx].name)
        primary_stem = self._files[selected[0]].stem
        self._join_output_var.set(f"{primary_stem}_joined.acc")

    def _move_join_item(self, direction: int) -> None:
        sel = self._join_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        new_idx = idx + direction
        if new_idx < 0 or new_idx >= self._join_listbox.size():
            return
        text = self._join_listbox.get(idx)
        self._join_listbox.delete(idx)
        self._join_listbox.insert(new_idx, text)
        self._join_listbox.selection_set(new_idx)

    def _remove_join_item(self) -> None:
        sel = self._join_listbox.curselection()
        if sel:
            self._join_listbox.delete(sel[0])

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _do_join(self) -> None:
        if not self._directory:
            messagebox.showerror("No directory", "Select a directory first.")
            return

        names = list(self._join_listbox.get(0, "end"))
        if len(names) < 2:
            messagebox.showerror("Too few files", "Add at least two files to the join list.")
            return

        output_name = self._join_output_var.get().strip()
        if not output_name:
            messagebox.showerror("No output name", "Specify an output filename.")
            return
        if not output_name.lower().endswith(".acc"):
            output_name += ".acc"

        # Warn if output would overwrite an existing file
        out_path = self._directory / output_name
        if out_path.exists():
            if not messagebox.askyesno(
                "Overwrite?", f"'{output_name}' already exists. Overwrite it?"
            ):
                return

        try:
            dicts: list[dict] = []
            for name in names:
                fp = self._directory / name
                self._log_message(f"Loading {name}…")
                dicts.append(acquisition_editor.load_file(fp))

            combined = _join_acc_dicts(dicts)
            acquisition_editor.save_acc(combined, out_path)
            n_total = _n_scans(combined)
            self._log_message(f"Saved '{output_name}' ({n_total} scans total)")
            messagebox.showinfo(
                "Done",
                f"Joined {len(names)} file(s) → '{output_name}'\n({n_total} scans total)",
            )
            self._refresh_files()
        except Exception as exc:
            self._log_message(f"Error during join: {exc}")
            messagebox.showerror("Join failed", str(exc))

    def _do_split(self) -> None:
        source_name = self._split_source_var.get()
        if source_name == "(none selected)":
            messagebox.showerror("No file", "Select a file in the directory list.")
            return

        if not self._directory:
            messagebox.showerror("No directory", "Select a directory first.")
            return

        out_a = self._split_out_a_var.get().strip()
        out_b = self._split_out_b_var.get().strip()
        if not out_a or not out_b:
            messagebox.showerror("No output names", "Specify output filenames for both parts.")
            return
        if not out_a.lower().endswith(".acc"):
            out_a += ".acc"
        if not out_b.lower().endswith(".acc"):
            out_b += ".acc"

        # Warn on overwrites
        for name in (out_a, out_b):
            if (self._directory / name).exists():
                if not messagebox.askyesno("Overwrite?", f"'{name}' already exists. Overwrite it?"):
                    return

        try:
            fp = self._directory / source_name
            self._log_message(f"Loading {source_name}…")
            data_dict = acquisition_editor.load_file(fp)
            n = _n_scans(data_dict)

            if n < 2:
                messagebox.showerror("Cannot split", f"'{source_name}' has only {n} scan — need at least 2.")
                return

            mode = self._split_mode.get()
            if mode == "position":
                pos = int(self._split_pos_var.get())
                pos = max(1, min(pos, n - 1))
                part_a, part_b = _split_acc_at_position(data_dict, pos)
            elif mode == "evenodd":
                part_a, part_b = _split_acc_even_odd(data_dict)
            else:
                raise ValueError(f"Unknown split mode: {mode!r}")

            path_a = self._directory / out_a
            path_b = self._directory / out_b

            acquisition_editor.save_acc(part_a, path_a)
            na = _n_scans(part_a)
            acquisition_editor.save_acc(part_b, path_b)
            nb = _n_scans(part_b)

            self._log_message(f"Saved '{out_a}' ({na} scans) and '{out_b}' ({nb} scans)")
            messagebox.showinfo(
                "Done",
                f"Split '{source_name}' into:\n  '{out_a}': {na} scans\n  '{out_b}': {nb} scans",
            )
            self._refresh_files()
        except Exception as exc:
            self._log_message(f"Error during split: {exc}")
            messagebox.showerror("Split failed", str(exc))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _log_message(self, msg: str) -> None:
        self._log.configure(state="normal")
        self._log.insert("end", msg + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = AccFileManager()
    app.mainloop()
