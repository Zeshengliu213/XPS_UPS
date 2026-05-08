# app.py
# ============================================================
# UPS IBW Processor - 图形界面 (v3.2)
# 依赖：reader, plots, export_csv, ui_theme, file_loader, fit_window
# 可选：windnd（拖拽添加文件，Windows）
# ============================================================

import os
import sys
import traceback

import numpy as np

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from export_csv import (
    export_csv_separate,
    export_csv_merged_horizontal,
    get_scan_range_tag,
)
from ui_theme import COLORS, FONTS, create_button, create_section, create_log_panel
from file_loader import FileLoader
from fit_window import FitWindow

try:
    import windnd
    HAS_WINDND = True
except ImportError:
    HAS_WINDND = False


class UPSFrame(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=COLORS["bg"])

        self.files = []
        self.spectra = []
        self.out_dir = None

        self.zoomA = (18.0, 15.0)
        self.zoomB = (-1.0, 2.0)

        self._loader = FileLoader(schedule_fn=lambda fn: self.after(0, fn))

        self._build_ui()
        self._setup_drag_drop()

    # ------------------------------------------------------------------ #
    #  Styled widget helpers                                               #
    # ------------------------------------------------------------------ #

    def _opt_label(self, parent, text, row):
        tk.Label(parent, text=text, bg=COLORS["card"],
                 fg=COLORS["text"], font=FONTS["body"],
                 ).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)

    def _opt_radio(self, parent, text, var, value, row, col):
        tk.Radiobutton(
            parent, text=text, variable=var, value=value,
            bg=COLORS["card"], fg=COLORS["text"],
            selectcolor=COLORS["primary_light"],
            activebackground=COLORS["card"],
            activeforeground=COLORS["primary"],
            font=FONTS["body"], cursor="hand2",
        ).grid(row=row, column=col, sticky="w", padx=8, pady=4)

    def _opt_check(self, parent, text, var, row, col):
        cb = tk.Checkbutton(
            parent, text=text, variable=var,
            bg=COLORS["card"], fg=COLORS["text"],
            selectcolor=COLORS["primary_light"],
            activebackground=COLORS["card"],
            activeforeground=COLORS["primary"],
            font=FONTS["body"], cursor="hand2",
        )
        cb.grid(row=row, column=col, sticky="w", padx=8, pady=4)
        return cb

    def _zoom_entry(self, parent, textvar, row, col):
        e = tk.Entry(
            parent, textvariable=textvar, width=6,
            bg=COLORS["list_bg"], fg=COLORS["text"], relief="flat",
            font=FONTS["body"],
            highlightthickness=1, highlightbackground=COLORS["card_border"],
            insertbackground=COLORS["primary"],
        )
        e.grid(row=row, column=col, sticky="w", padx=2, pady=3)
        return e

    # ------------------------------------------------------------------ #
    #  UI build                                                            #
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        # ── 顶部 3px 主色彩条 ─────────────────────────────────────────
        tk.Frame(self, bg=COLORS["primary"], height=3).pack(fill="x")

        # ── Header ────────────────────────────────────────────────────
        header = tk.Frame(self, bg=COLORS["bg"])
        header.pack(fill="x", padx=20, pady=(12, 6))

        # 左侧：图标徽章 + 标题
        left_h = tk.Frame(header, bg=COLORS["bg"])
        left_h.pack(side="left")

        badge = tk.Frame(left_h, bg=COLORS["primary"], padx=10, pady=4)
        badge.pack(side="left")
        tk.Label(badge, text="⚗", font=("Segoe UI", 13),
                 fg="white", bg=COLORS["primary"]).pack(side="left")
        tk.Label(badge, text="XPS / UPS", font=FONTS["section"],
                 fg="white", bg=COLORS["primary"]).pack(side="left", padx=(4, 0))

        tk.Label(left_h, text="  IBW Processor", font=FONTS["title"],
                 fg=COLORS["text"], bg=COLORS["bg"]).pack(side="left")

        # 右侧：版本 & 光子能量徽章
        right_h = tk.Frame(header, bg=COLORS["bg"])
        right_h.pack(side="right", padx=(0, 4))

        for txt, fg, bg in [
            ("v3.2",          COLORS["secondary"],  COLORS["button_bg"]),
            ("He I  21.22 eV", COLORS["primary"],   COLORS["primary_light"]),
        ]:
            tk.Label(right_h, text=txt, font=FONTS["badge"],
                     fg=fg, bg=bg, padx=8, pady=3,
                     ).pack(side="left", padx=4)

        # ── Main area: 3-step wizard + log ─────────────────────────────
        main = tk.Frame(self, bg=COLORS["bg"])
        main.pack(fill="both", expand=True, padx=20, pady=(4, 0))

        left = create_section(main, "UPS Workflow",
                             side="left", fill="both", expand=True)

        self.steps_nb = ttk.Notebook(left)
        self.steps_nb.pack(fill="both", expand=True)

        self.step_files = tk.Frame(self.steps_nb, bg=COLORS["card"])
        self.step_preview = tk.Frame(self.steps_nb, bg=COLORS["card"])
        self.step_export = tk.Frame(self.steps_nb, bg=COLORS["card"])

        self.steps_nb.add(self.step_files, text="1) Files")
        self.steps_nb.add(self.step_preview, text="2) Preview")
        self.steps_nb.add(self.step_export, text="3) Export")

        # Build step pages
        self._build_step_files(self.step_files)
        self._build_step_preview(self.step_preview)
        self._build_step_export(self.step_export)

        # Disable steps until spectra loaded
        self._set_steps_enabled(loaded=False)

        # Log panel (always visible)
        right = create_section(main, "Log",
                              side="left", fill="both", expand=True, padx=(8, 0))
        self.log = create_log_panel(right)

        # ── Status bar ────────────────────────────────────────────────
        status_frame = tk.Frame(self, bg=COLORS["card"],
                                highlightbackground=COLORS["card_border"],
                                highlightthickness=1, height=30)
        status_frame.pack(fill="x", side="bottom", padx=20, pady=6)
        status_frame.pack_propagate(False)

        self._status_dot = tk.Label(
            status_frame, text="●", font=("Segoe UI", 11),
            fg=COLORS["success"], bg=COLORS["card"])
        self._status_dot.pack(side="left", padx=(10, 4), pady=4)

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(status_frame, textvariable=self.status_var,
                 font=FONTS["small"], fg=COLORS["secondary"],
                 bg=COLORS["card"]).pack(side="left", pady=4)

        self.progress = ttk.Progressbar(status_frame,
                                        mode="indeterminate", length=120)

        self._log("Ready — select or drag & drop .ibw files to begin.", "dim")

    def _build_step_files(self, parent):
        """Step 1: choose files and show list."""
        top = tk.Frame(parent, bg=COLORS["card"])
        top.pack(fill="x", padx=10, pady=10)
        create_button(top, "Select .ibw files", self.pick_files, primary=True).pack(side="left")
        create_button(top, "Choose output folder", self.pick_out_dir).pack(side="left", padx=8)

        self.out_dir_var = tk.StringVar(value="Default: same folder as first IBW")
        tk.Label(top, textvariable=self.out_dir_var, fg=COLORS["secondary"], bg=COLORS["card"],
                 font=FONTS["small"]).pack(side="left", padx=8)

        mid = tk.Frame(parent, bg=COLORS["card"])
        mid.pack(fill="x", padx=10, pady=(0, 6))
        create_button(mid, "Remove selected", self.remove_selected).pack(side="left")
        create_button(mid, "Clear list", self.clear_list, danger=True).pack(side="left", padx=8)

        tk.Label(parent,
                 text="Tip: Drag & drop .ibw files onto the window to add them.",
                 fg=COLORS["secondary"], bg=COLORS["card"], font=FONTS["small"]).pack(
            anchor="w", padx=10, pady=(0, 6)
        )

        box = tk.Frame(parent, bg=COLORS["card"])
        box.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.listbox = tk.Listbox(
            box, height=12,
            font=FONTS["body"],
            bg=COLORS["list_bg"], fg=COLORS["list_fg"],
            selectbackground=COLORS["list_select"], selectforeground=COLORS["text"],
            relief="flat", highlightthickness=0, activestyle="none",
        )
        self.listbox.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(box, orient="vertical", command=self.listbox.yview)
        sb.pack(side="left", fill="y")
        self.listbox.config(yscrollcommand=sb.set)

    def _build_step_preview(self, parent):
        """Step 2: plotting options + preview."""
        opt = tk.Frame(parent, bg=COLORS["card"])
        opt.pack(fill="x", padx=10, pady=10)

        self.plot_mode = tk.StringVar(value="overlay")
        self._opt_label(opt, "Plot mode:", 0)
        self._opt_radio(opt, "Overlay (one figure)", self.plot_mode, "overlay", 0, 1)
        self._opt_radio(opt, "Separate (per file)", self.plot_mode, "separate", 0, 2)

        self.save_homo_png_var = tk.BooleanVar(value=False)
        self._opt_check(opt, "Preview HOMO stitched (EF + SECO)", self.save_homo_png_var, 1, 1)

        self.zoom_enable = tk.BooleanVar(value=True)
        self.zoom_check_btn = self._opt_check(opt, "Add two zoom panels", self.zoom_enable, 1, 2)

        # Zoom range entries
        tk.Label(opt, text="Zoom A (eV):", bg=COLORS["card"], fg=COLORS["text"], font=FONTS["body"]).grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=3
        )
        self.zoom_a_lo_var = tk.StringVar(value="18")
        self.zoom_a_hi_var = tk.StringVar(value="15")
        self._zoom_entry(opt, self.zoom_a_lo_var, 2, 1)
        tk.Label(opt, text="–", bg=COLORS["card"], fg=COLORS["secondary"], font=FONTS["body"]).grid(
            row=2, column=2, sticky="w", padx=2, pady=3
        )
        self._zoom_entry(opt, self.zoom_a_hi_var, 2, 3)

        tk.Label(opt, text="Zoom B (eV):", bg=COLORS["card"], fg=COLORS["text"], font=FONTS["body"]).grid(
            row=3, column=0, sticky="w", padx=(0, 8), pady=3
        )
        self.zoom_b_lo_var = tk.StringVar(value="-1")
        self.zoom_b_hi_var = tk.StringVar(value="3")
        self._zoom_entry(opt, self.zoom_b_lo_var, 3, 1)
        tk.Label(opt, text="–", bg=COLORS["card"], fg=COLORS["secondary"], font=FONTS["body"]).grid(
            row=3, column=2, sticky="w", padx=2, pady=3
        )
        self._zoom_entry(opt, self.zoom_b_hi_var, 3, 3)

        self.zoom_hint_label = tk.Label(
            opt,
            text="Zoom A: SECO region  ·  Zoom B: near EF  ·  Available when spectrum starts ≥ 20 eV",
            fg=COLORS["secondary"], bg=COLORS["card"], font=FONTS["small"], justify="left",
        )
        self.zoom_hint_label.grid(row=4, column=0, columnspan=4, sticky="w", padx=0, pady=(4, 0))
        self.zoom_check_btn.grid_remove()
        self.zoom_hint_label.grid_remove()

        act = tk.Frame(parent, bg=COLORS["card"])
        act.pack(fill="x", padx=10, pady=(0, 10))
        self.preview_btn = create_button(act, "▶  Preview", self.preview, primary=True)
        self.preview_btn.pack(side="left")
        tk.Label(act, text="(opens matplotlib windows)", fg=COLORS["secondary"], bg=COLORS["card"],
                 font=FONTS["small"]).pack(side="left", padx=10)

    def _build_step_export(self, parent):
        """Step 3: export options + export."""
        opt = tk.Frame(parent, bg=COLORS["card"])
        opt.pack(fill="x", padx=10, pady=10)

        self.export_mode = tk.StringVar(value="separate_csv")
        self._opt_label(opt, "CSV export:", 0)
        self._opt_radio(opt, "Separate CSV per file", self.export_mode, "separate_csv", 0, 1)
        self._opt_radio(opt, "Merged CSV (horizontal)", self.export_mode, "merged_horizontal", 0, 2)

        self.save_png_var = tk.BooleanVar(value=True)
        self._opt_check(opt, "Export PNG figures", self.save_png_var, 1, 1)
        self.save_svg_var = tk.BooleanVar(value=False)
        self._opt_check(opt, "Export SVG figures", self.save_svg_var, 1, 2)

        self.export_homo_var = tk.BooleanVar(value=False)
        self._opt_check(opt, "Export HOMO stitched PNG (EF + SECO)", self.export_homo_var, 2, 1)

        act = tk.Frame(parent, bg=COLORS["card"])
        act.pack(fill="x", padx=10, pady=(0, 10))
        self.export_btn = create_button(act, "⬇  Export", self.export, primary=True)
        self.export_btn.pack(side="left")
        tk.Label(act, text="(CSV + figures)", fg=COLORS["secondary"], bg=COLORS["card"],
                 font=FONTS["small"]).pack(side="left", padx=10)

    def _set_steps_enabled(self, loaded: bool):
        """Enable/disable Preview & Export steps depending on load state."""
        state = "normal" if loaded else "disabled"
        try:
            self.steps_nb.tab(1, state=state)  # Preview
            self.steps_nb.tab(2, state=state)  # Export
        except Exception:
            pass
        if hasattr(self, "preview_btn"):
            self.preview_btn.configure(state=("normal" if loaded else "disabled"))
        if hasattr(self, "export_btn"):
            self.export_btn.configure(state=("normal" if loaded else "disabled"))
        if not loaded:
            try:
                self.steps_nb.select(0)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    #  Zoom helpers                                                        #
    # ------------------------------------------------------------------ #

    def _spectra_start_from_20ev(self):
        if not self.spectra:
            return False
        return any(float(np.max(s["x"])) >= 20.0 for s in self.spectra)

    def _update_zoom_ui(self):
        if self._spectra_start_from_20ev():
            # In Step 2 (Preview) grid:
            # - zoom_check_btn is at row=1
            # - zoom_hint_label is at row=4
            self.zoom_check_btn.grid(row=1, column=2, sticky="w", padx=8, pady=4)
            self.zoom_hint_label.grid(row=4, column=0, columnspan=4, sticky="w", padx=0, pady=(4, 0))
        else:
            self.zoom_check_btn.grid_remove()
            self.zoom_hint_label.grid_remove()
            self.zoom_enable.set(False)

    def _get_zoom_ranges(self):
        try:
            a_lo, a_hi = float(self.zoom_a_lo_var.get()), float(self.zoom_a_hi_var.get())
            zoomA = (min(a_lo, a_hi), max(a_lo, a_hi))
        except (ValueError, TypeError):
            zoomA = (15.0, 18.0)
        try:
            b_lo, b_hi = float(self.zoom_b_lo_var.get()), float(self.zoom_b_hi_var.get())
            zoomB = (min(b_lo, b_hi), max(b_lo, b_hi))
        except (ValueError, TypeError):
            zoomB = (-1.0, 2.0)
        return zoomA, zoomB

    def _zoom_effective(self):
        return self._spectra_start_from_20ev() and self.zoom_enable.get()

    # ------------------------------------------------------------------ #
    #  Drag & drop                                                         #
    # ------------------------------------------------------------------ #

    def _setup_drag_drop(self):
        if HAS_WINDND:
            try:
                windnd.hook_dropfiles(self.winfo_toplevel(), func=self._on_drop_files)
            except Exception:
                pass

    def _on_drop_files(self, paths):
        if not paths:
            return
        decoded = []
        for p in paths:
            if isinstance(p, bytes):
                try:
                    p = os.fsdecode(p)
                except (UnicodeDecodeError, TypeError):
                    p = p.decode("utf-8", errors="replace")
            if p and p.lower().endswith(".ibw"):
                decoded.append(p)
        if decoded:
            self._add_files(decoded, replace=False)

    # ------------------------------------------------------------------ #
    #  File management                                                     #
    # ------------------------------------------------------------------ #

    def _add_files(self, paths, replace=True):
        paths = [os.path.normpath(p) for p in paths]
        if replace:
            self.files = list(paths)
        else:
            seen = set(self.files)
            for p in paths:
                if p not in seen:
                    seen.add(p)
                    self.files.append(p)
        self._refresh_listbox()
        if not self.files:
            self._set_status("Ready")
            return
        if self.out_dir is None:
            self.out_dir_var.set(f"→  {os.path.dirname(self.files[0])}")
        self._log(f"Loading {len(self.files)} file(s)…", "dim")
        self._load_spectra_async()

    def _refresh_listbox(self):
        self.listbox.delete(0, "end")
        for p in self.files:
            # Show "filename  (folder)" for readability
            base = os.path.basename(p)
            folder = os.path.basename(os.path.dirname(p))
            self.listbox.insert("end", f"  {base}  ({folder})")

    def remove_selected(self):
        sel = list(self.listbox.curselection())
        if not sel:
            self._log("No file selected — click a row to select first.", "dim")
            return
        for i in reversed(sel):
            self.files.pop(i)
        self._refresh_listbox()
        if not self.files:
            self.spectra = []
            self.out_dir_var.set("Default: same folder as first IBW")
            self._update_zoom_ui()
            self._set_steps_enabled(loaded=False)
            self._log("List cleared.", "dim")
            return
        self._log(f"Reloading {len(self.files)} file(s)…", "dim")
        self._load_spectra_async()

    def clear_list(self):
        self._loader.cancel()
        self.files = []
        self.spectra = []
        self.listbox.delete(0, "end")
        self.out_dir_var.set("Default: same folder as first IBW")
        self._set_status("Ready")
        self._update_zoom_ui()
        self._set_steps_enabled(loaded=False)
        self._log("List cleared.", "dim")

    def pick_files(self):
        paths = filedialog.askopenfilenames(
            title="Select IBW files",
            filetypes=[("IBW files", "*.ibw"), ("All files", "*.*")],
        )
        if not paths:
            return
        self._add_files(list(paths), replace=True)

    def pick_out_dir(self):
        d = filedialog.askdirectory(title="Choose output folder")
        if not d:
            return
        self.out_dir = d
        self.out_dir_var.set(f"→  {d}")
        self._log(f"Output folder: {d}", "dim")

    # ------------------------------------------------------------------ #
    #  Async loading                                                       #
    # ------------------------------------------------------------------ #

    def _load_spectra_async(self):
        self._set_status("Loading…")
        self.progress.pack(side="right", padx=10, pady=5)
        self.progress.start(40)
        self._loader.load(list(self.files), on_done=self._on_load_done)

    def _on_load_done(self, spectra, ok, bad, failures):
        self.spectra = spectra
        self.progress.stop()
        self.progress.pack_forget()
        self._set_status("Ready")
        for fp, err in failures:
            self._log(f"FAILED  {os.path.basename(fp)}: {err}", "err")
        tag = "ok" if bad == 0 else "warn"
        self._log(f"Loaded {ok} spectrum{'s' if ok != 1 else ''}   Failed {bad}", tag)
        self._update_zoom_ui()
        self._set_steps_enabled(loaded=(ok > 0))

    # ------------------------------------------------------------------ #
    #  Guard / helpers                                                     #
    # ------------------------------------------------------------------ #

    def ensure_ready(self):
        if not self.files:
            messagebox.showwarning("No Files",
                                   "Please select one or more .ibw files first.")
            return False
        if not self.spectra:
            messagebox.showwarning("Not Loaded",
                                   "No spectrum was loaded successfully.")
            return False
        return True

    def get_out_dir(self):
        return self.out_dir or os.path.dirname(self.files[0])

    def _log(self, msg: str, tag: str = ""):
        """向日志追加一行，tag 控制颜色：ok/err/warn/phi/dim/bold。"""
        self.log.insert("end", msg + "\n", tag)
        self.log.see("end")

    def _set_status(self, text: str):
        self.status_var.set(text)
        dot_color = {
            "Ready":       COLORS["success"],
            "Loading…":    COLORS["warning"],
            "Exporting…":  COLORS["warning"],
        }.get(text, COLORS["secondary"])
        self._status_dot.configure(fg=dot_color)

    def _log_seco_results(self, spectra, loA, hiA):
        from plots import find_seco, HV_HEI
        self._log("Work function  φ = 21.22 − SECO (eV):", "bold")
        for s in spectra:
            BE_cut, phi, aux = find_seco(
                s["x"], s["y_norm"], search_region=(loA, hiA), hv=HV_HEI)
            if BE_cut is not None:
                self._log(
                    f"  {s['base']}:  SECO = {BE_cut:.2f} eV  →  φ = {phi:.2f} eV",
                    "phi")
            else:
                reason = (aux.get("reason", "unknown") if aux
                          else f"no edge in {loA}–{hiA} eV")
                self._log(f"  {s['base']}:  not found — {reason}", "warn")

    # ------------------------------------------------------------------ #
    #  Actions                                                             #
    # ------------------------------------------------------------------ #

    def preview(self):
        if not self.ensure_ready():
            return
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            from plots import (
                plot_preview_two_regions_overlay,
                plot_preview_two_regions_separate,
                plot_homo_stitched,
                HV_HEI,
            )

            z = self._zoom_effective()
            zA, zB = self._get_zoom_ranges()
            if self.plot_mode.get() == "overlay":
                fig = plot_preview_two_regions_overlay(
                    self.spectra,
                    region_seco=zA,
                    region_ef=zB,
                    annotate_edges=z,
                )
                fig.canvas.manager.set_window_title("UPS Preview (SECO + EF)")
                plt.show()
            else:
                figs = plot_preview_two_regions_separate(
                    self.spectra,
                    region_seco=zA,
                    region_ef=zB,
                    annotate_edges=z,
                )
                for base, fig in figs:
                    fig.canvas.manager.set_window_title(f"{base} · Preview (SECO + EF)")
                plt.show()

            # 额外预览：HOMO stitched 图（与导出同一个勾选项）
            if self.save_homo_png_var.get():
                for s in self.spectra:
                    fig = plot_homo_stitched(s, zoomA=zA, homo_range=(-1.0, 5.0), hv=HV_HEI)
                    fig.canvas.manager.set_window_title(f"{s['base']} · HOMO stitched")
                plt.show()

            if z:
                self._log_seco_results(self.spectra, min(zA), max(zA))
            self._log("Preview done.", "ok")
        except Exception:
            self._log("Preview failed:\n" + traceback.format_exc(), "err")
            messagebox.showerror("Error", "Preview failed — check the Log.")

    def export(self):
        if not self.ensure_ready():
            return
        self._set_status("Exporting…")
        self.update_idletasks()
        out_dir = self.get_out_dir()
        os.makedirs(out_dir, exist_ok=True)
        try:
            from plots import plot_overlay, plot_separate, plot_homo_stitched, save_figure, HV_HEI

            if self.export_mode.get() == "separate_csv":
                paths = export_csv_separate(self.spectra, out_dir)
                self._log(f"CSV: {len(paths)} file(s)  →  {out_dir}", "ok")
            else:
                path = export_csv_merged_horizontal(self.spectra, out_dir)
                self._log(f"Merged CSV  →  {path}", "ok")

            z = self._zoom_effective()
            zA, zB = self._get_zoom_ranges()

            if self.save_png_var.get() or self.save_svg_var.get():
                if self.plot_mode.get() == "overlay":
                    fig = plot_overlay(self.spectra, zoom_enable=z, zoomA=zA, zoomB=zB)
                    tag = get_scan_range_tag(self.spectra[0])
                    if self.save_png_var.get():
                        png = os.path.join(out_dir, f"UPS_overlay_{tag}.png")
                        save_figure(fig, png)
                        self._log(f"PNG  →  {png}", "ok")
                        fig = None
                    if self.save_svg_var.get():
                        if fig is None:
                            fig = plot_overlay(self.spectra, zoom_enable=z, zoomA=zA, zoomB=zB)
                        svg = os.path.join(out_dir, f"UPS_overlay_{tag}.svg")
                        save_figure(fig, svg)
                        self._log(f"SVG  →  {svg}", "ok")
                else:
                    figs = plot_separate(self.spectra, zoom_enable=z, zoomA=zA, zoomB=zB)
                    for (base, fig), s in zip(figs, self.spectra):
                        tag = get_scan_range_tag(s)
                        if self.save_png_var.get():
                            png = os.path.join(out_dir, f"{base}_{tag}.png")
                            save_figure(fig, png)
                            self._log(f"PNG  →  {png}", "ok")
                            fig = None
                        if self.save_svg_var.get():
                            if fig is None:
                                # 重新生成一次（上面保存会 close figure）
                                fig2s = plot_separate([s], zoom_enable=z, zoomA=zA, zoomB=zB)
                                _, fig = fig2s[0]
                            svg = os.path.join(out_dir, f"{base}_{tag}.svg")
                            save_figure(fig, svg)
                            self._log(f"SVG  →  {svg}", "ok")

            # HOMO stitched readout figure (always per spectrum)
            if self.export_homo_var.get():
                for s in self.spectra:
                    fig = plot_homo_stitched(s, zoomA=zA, homo_range=(-1.0, 5.0), hv=HV_HEI)
                    tag = get_scan_range_tag(s)
                    png = os.path.join(out_dir, f"{s['base']}_HOMO_{tag}.png")
                    save_figure(fig, png)
                self._log(f"HOMO PNG: {len(self.spectra)} figure(s)  →  {out_dir}", "ok")

            if z:
                self._log_seco_results(self.spectra, min(zA), max(zA))

            self._set_status("Ready")
            messagebox.showinfo("Export done", f"All files saved to:\n{out_dir}")
        except Exception:
            self._set_status("Ready")
            self._log("Export failed:\n" + traceback.format_exc(), "err")
            messagebox.showerror("Error", "Export failed — check the Log.")

    def _open_xps_module(self):
        messagebox.showinfo("XPS module", "Please restart the program and choose the XPS module from the start screen.")


if __name__ == "__main__":
    try:
        root = tk.Tk()
        root.title("XPS / UPS IBW Processor")
        root.geometry("1020x700")
        root.minsize(960, 660)
        UPSFrame(root).pack(fill="both", expand=True)
        root.mainloop()
    except Exception:
        print(traceback.format_exc())
        sys.exit(1)
