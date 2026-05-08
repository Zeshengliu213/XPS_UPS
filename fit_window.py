# fit_window.py
# ============================================================
# XPS 峰拟合交互窗口（Toplevel）
#
# 布局：
from __future__ import annotations

#   左侧固定面板 ── 谱选择 / 拟合范围 / 背景 / 峰列表 / 操作按钮
#   右侧可扩展   ── matplotlib 画布（主图 + 残差图）+ 拟合结果表格
# ============================================================

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt

from ui_theme import COLORS, FONTS, create_button
from peak_fit import fit_xps, pseudo_voigt, linear_background, shirley_background, _trapezoid
from xps_chem_states import guess_core_from_range, match_chem_states

try:
    from autofit_engine import auto_fit as _dl_auto_fit
    _HAS_AUTOFIT = True
    _AUTOFIT_IMPORT_ERR = ""
except Exception as _e:
    _HAS_AUTOFIT = False
    _AUTOFIT_IMPORT_ERR = str(_e)

try:
    from scipy.signal import find_peaks, savgol_filter
    _HAS_SCIPY_SIGNAL = True
except Exception:
    _HAS_SCIPY_SIGNAL = False


# ------------------------------------------------------------------ #
#  Matplotlib paper-style defaults (for XPS fitting figure)            #
# ------------------------------------------------------------------ #

_PLOT_STYLE = {
    # Prefer Arial; fallback to common sans-serif
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    # Slightly larger, journal-ready sizes
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "axes.linewidth": 1.4,
    "grid.linewidth": 0.7,
    "lines.linewidth": 2.2,
    "savefig.dpi": 300,
    "figure.dpi": 120,
}


def _apply_plot_style():
    # Apply once (idempotent)
    try:
        matplotlib.rcParams.update(_PLOT_STYLE)
    except Exception:
        pass


# ------------------------------------------------------------------ #
#  内部工具函数                                                         #
# ------------------------------------------------------------------ #

_btn = create_button


def _section_header(parent, title):
    """在左侧面板内画一个带分割线的小节标题。"""
    tk.Label(parent, text=title, font=FONTS["section"],
             fg=COLORS["primary"], bg=COLORS["card"]).pack(anchor="w", padx=12, pady=(10, 2))
    tk.Frame(parent, bg=COLORS["divider"], height=1).pack(fill="x", padx=12, pady=(0, 6))


# ------------------------------------------------------------------ #
#  添加峰对话框                                                         #
# ------------------------------------------------------------------ #

class _AddPeakDialog(tk.Toplevel):
    """模态对话框：输入峰中心、FWHM、η。"""

    def __init__(self, parent, default_center="", default_fwhm="1.0", default_eta="0.3"):
        super().__init__(parent)
        self.title("Add Peak")
        self.resizable(False, False)
        self.configure(bg=COLORS["card"])
        self.grab_set()
        self.result = None

        # 顶部彩条
        tk.Frame(self, bg=COLORS["primary"], height=3).pack(fill="x")

        # 在 Toplevel 内不要混用 pack/grid：外层用 pack，内容区单独用 grid
        body = tk.Frame(self, bg=COLORS["card"])
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)

        rows = [
            ("Center (eV):",              default_center),
            ("FWHM (eV):",                default_fwhm),
            ("η  (0 = Gauss, 1 = Lorentz):", default_eta),
        ]
        self._vars = []
        for i, (label, default) in enumerate(rows):
            tk.Label(body, text=label, bg=COLORS["card"],
                     font=FONTS["body"], fg=COLORS["text"],
                     ).grid(row=i, column=0, sticky="w", padx=(16, 8), pady=7)
            v = tk.StringVar(value=default)
            entry = tk.Entry(body, textvariable=v, width=11,
                             bg=COLORS["list_bg"], fg=COLORS["text"],
                             relief="flat", font=FONTS["body"],
                             highlightthickness=1,
                             highlightbackground=COLORS["card_border"],
                             insertbackground=COLORS["primary"])
            entry.grid(row=i, column=1, sticky="ew", padx=(0, 16), pady=7)
            self._vars.append(v)

        btn_row = tk.Frame(body, bg=COLORS["card"])
        btn_row.grid(row=len(rows), column=0, columnspan=2, pady=(4, 14))
        _btn(btn_row, "OK",     self._ok,       primary=True).pack(side="left", padx=6)
        _btn(btn_row, "Cancel", self.destroy,              ).pack(side="left", padx=6)

        self.wait_window()

    def _ok(self):
        try:
            center = float(self._vars[0].get())
            fwhm   = float(self._vars[1].get())
            eta    = float(self._vars[2].get())
            if fwhm <= 0:
                raise ValueError("FWHM 必须 > 0")
            if not (0.0 <= eta <= 1.0):
                raise ValueError("η 须在 0–1 之间")
            self.result = {"center": center, "fwhm": fwhm, "eta": eta}
            self.destroy()
        except ValueError as e:
            messagebox.showerror("Input error", str(e), parent=self)


class _StackExportDialog(tk.Toplevel):
    """模态对话框：选择两个谱用于 stacked 导出。"""

    def __init__(self, parent, spectrum_names: list[str], default_a: str = "", default_b: str = ""):
        super().__init__(parent)
        self.title("Export stacked figure")
        self.resizable(False, False)
        self.configure(bg=COLORS["card"])
        self.grab_set()
        self.result = None

        tk.Frame(self, bg=COLORS["primary"], height=3).pack(fill="x")

        body = tk.Frame(self, bg=COLORS["card"])
        body.pack(fill="both", expand=True)

        tk.Label(
            body,
            text="Choose two spectra (top / bottom):",
            bg=COLORS["card"],
            fg=COLORS["text"],
            font=FONTS["body"],
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=16, pady=(14, 8))

        tk.Label(body, text="Top:", bg=COLORS["card"], fg=COLORS["secondary"], font=FONTS["body"]).grid(
            row=1, column=0, sticky="e", padx=(16, 8), pady=6
        )
        self._a = tk.StringVar(value=default_a or (spectrum_names[0] if spectrum_names else ""))
        cmb_a = ttk.Combobox(body, textvariable=self._a, values=spectrum_names, state="readonly", width=30)
        cmb_a.grid(row=1, column=1, sticky="w", padx=(0, 16), pady=6)

        tk.Label(body, text="Bottom:", bg=COLORS["card"], fg=COLORS["secondary"], font=FONTS["body"]).grid(
            row=2, column=0, sticky="e", padx=(16, 8), pady=6
        )
        self._b = tk.StringVar(value=default_b or (spectrum_names[1] if len(spectrum_names) > 1 else self._a.get()))
        cmb_b = ttk.Combobox(body, textvariable=self._b, values=spectrum_names, state="readonly", width=30)
        cmb_b.grid(row=2, column=1, sticky="w", padx=(0, 16), pady=6)

        btn_row = tk.Frame(body, bg=COLORS["card"])
        btn_row.grid(row=3, column=0, columnspan=2, pady=(10, 14))
        _btn(btn_row, "Export", self._ok, primary=True).pack(side="left", padx=6)
        _btn(btn_row, "Cancel", self.destroy).pack(side="left", padx=6)

        self.wait_window()

    def _ok(self):
        a = self._a.get().strip()
        b = self._b.get().strip()
        if not a or not b:
            messagebox.showerror("Select spectra", "Please choose both top and bottom spectra.", parent=self)
            return
        if a == b:
            messagebox.showerror("Select spectra", "Top and bottom spectra must be different.", parent=self)
            return
        self.result = (a, b)
        self.destroy()


# ------------------------------------------------------------------ #
#  峰拟合主窗口                                                         #
# ------------------------------------------------------------------ #

class FitWindow(tk.Toplevel):
    """XPS 峰拟合窗口。"""

    _PEAK_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))

    def __init__(self, parent, spectra):
        super().__init__(parent)
        self.title("XPS Peak Fitting")
        self.geometry("1260x780")
        self.minsize(1060, 660)
        self.configure(bg=COLORS["bg"])

        self.spectra = spectra
        self._peak_guesses: list[dict] = []
        self._fit_result:   dict | None = None
        # cache fit results by spectrum name for stacked export
        self._fit_results_by_spec: dict[str, dict] = {}
        self._mpl_click_cid = None

        # Undo / Redo stacks for peak guesses
        self._undo_stack: list[list[dict]] = []
        self._redo_stack: list[list[dict]] = []

        # Background computation cache
        self._bg_cache_key = None
        self._bg_cache_data = None

        self._build_ui()
        self._refresh_canvas()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Keyboard shortcuts
        self.bind_all("<Control-z>", lambda _e: self._undo())
        self.bind_all("<Control-y>", lambda _e: self._redo())

    # ---------------------------------------------------------------- #
    #  UI 构建                                                           #
    # ---------------------------------------------------------------- #

    def _build_ui(self):
        # 顶部彩条
        tk.Frame(self, bg=COLORS["primary"], height=3).pack(fill="x")

        # 标题栏
        hdr = tk.Frame(self, bg=COLORS["bg"])
        hdr.pack(fill="x", padx=16, pady=(10, 6))
        badge = tk.Frame(hdr, bg=COLORS["primary"], padx=9, pady=3)
        badge.pack(side="left")
        tk.Label(badge, text="⚙", font=("Segoe UI", 12),
                 fg="white", bg=COLORS["primary"]).pack(side="left")
        tk.Label(badge, text=" Peak Fitting", font=FONTS["section"],
                 fg="white", bg=COLORS["primary"]).pack(side="left")
        tk.Label(hdr, text="  Pseudo-Voigt  ·  Linear / Shirley Background",
                 font=FONTS["subtitle"], fg=COLORS["secondary"],
                 bg=COLORS["bg"]).pack(side="left", padx=8)

        # 主体
        body = tk.Frame(self, bg=COLORS["bg"])
        body.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        # 左侧控制面板
        left = tk.Frame(body, bg=COLORS["card"], width=340,
                        highlightbackground=COLORS["card_border"],
                        highlightthickness=1)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        tk.Frame(left, bg=COLORS["primary"], width=3).pack(side="left", fill="y")

        # Scrollable controls (left panel)
        ctrl_wrap = tk.Frame(left, bg=COLORS["card"])
        ctrl_wrap.pack(side="left", fill="both", expand=True)

        self._ctrl_canvas = tk.Canvas(
            ctrl_wrap,
            bg=COLORS["card"],
            highlightthickness=0,
            bd=0,
        )
        sb = ttk.Scrollbar(ctrl_wrap, orient="vertical", command=self._ctrl_canvas.yview)
        self._ctrl_canvas.configure(yscrollcommand=sb.set)

        sb.pack(side="right", fill="y")
        self._ctrl_canvas.pack(side="left", fill="both", expand=True)

        self._ctrl_inner = tk.Frame(self._ctrl_canvas, bg=COLORS["card"])
        self._ctrl_window = self._ctrl_canvas.create_window((0, 0), window=self._ctrl_inner, anchor="nw")

        def _on_inner_configure(_e):
            self._ctrl_canvas.configure(scrollregion=self._ctrl_canvas.bbox("all"))

        def _on_canvas_configure(_e):
            # keep inner frame width synced to canvas width
            try:
                self._ctrl_canvas.itemconfigure(self._ctrl_window, width=self._ctrl_canvas.winfo_width())
            except Exception:
                pass

        self._ctrl_inner.bind("<Configure>", _on_inner_configure)
        self._ctrl_canvas.bind("<Configure>", _on_canvas_configure)

        def _bind_mousewheel(widget):
            widget.bind("<Enter>", lambda _e: self._ctrl_canvas.bind_all("<MouseWheel>", _on_mousewheel))
            widget.bind("<Leave>", lambda _e: self._ctrl_canvas.unbind_all("<MouseWheel>"))

        def _on_mousewheel(e):
            # Windows: e.delta is typically 120/-120
            self._ctrl_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

        _bind_mousewheel(self._ctrl_canvas)
        _bind_mousewheel(self._ctrl_inner)

        self._build_controls(self._ctrl_inner)

        # 右侧画布 + 结果
        right = tk.Frame(body, bg=COLORS["bg"])
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self._build_canvas(right)

    def _build_controls(self, parent):
        """左侧控制面板内容（两列表单布局）。"""
        # NOTE: parent is a dedicated inner frame; we use grid exclusively here.
        parent.grid_columnconfigure(0, weight=0)
        parent.grid_columnconfigure(1, weight=1)

        r = 0

        def section(title: str):
            nonlocal r
            tk.Label(
                parent,
                text=title,
                font=FONTS["section"],
                fg=COLORS["primary"],
                bg=COLORS["card"],
            ).grid(row=r, column=0, columnspan=2, sticky="w", padx=12, pady=(12, 4))
            tk.Frame(parent, bg=COLORS["divider"], height=1).grid(row=r + 1, column=0, columnspan=2, sticky="ew", padx=12)
            r += 2

        def label(text: str):
            tk.Label(parent, text=text, bg=COLORS["card"], fg=COLORS["secondary"], font=FONTS["small"]).grid(
                row=r, column=0, sticky="w", padx=12, pady=6
            )

        def field(widget):
            widget.grid(row=r, column=1, sticky="ew", padx=12, pady=6)

        entry_kw = dict(
            bg=COLORS["list_bg"],
            fg=COLORS["text"],
            relief="flat",
            font=FONTS["body"],
            highlightthickness=1,
            highlightbackground=COLORS["card_border"],
            insertbackground=COLORS["primary"],
        )

        # Spectrum
        section("Spectrum")
        self._spec_var = tk.StringVar()
        names = [s["base"] for s in self.spectra]
        self._spec_var.set(names[0] if names else "")
        label("File")
        cmb = ttk.Combobox(parent, textvariable=self._spec_var, values=names, state="readonly", width=20)
        field(cmb)
        cmb.bind("<<ComboboxSelected>>", lambda _e: self._on_spectrum_changed())
        r += 1

        # Fit range
        section("Fit")
        self._range_lo = tk.StringVar()
        self._range_hi = tk.StringVar()
        label("Range (eV)")
        rng = tk.Frame(parent, bg=COLORS["card"])
        rng.grid(row=r, column=1, sticky="ew", padx=12, pady=6)
        rng.columnconfigure(0, weight=1)
        e_lo = tk.Entry(rng, textvariable=self._range_lo, width=7, **entry_kw)
        e_hi = tk.Entry(rng, textvariable=self._range_hi, width=7, **entry_kw)
        e_lo.pack(side="left")
        tk.Label(rng, text="–", bg=COLORS["card"], fg=COLORS["secondary"], font=FONTS["body"]).pack(side="left", padx=6)
        e_hi.pack(side="left")
        _btn(rng, "↺", self._auto_range, padx=8, pady=3).pack(side="left", padx=(8, 0))
        r += 1

        # Background
        self._bg_var = tk.StringVar(value="shirley")
        label("Background")
        bg_row = tk.Frame(parent, bg=COLORS["card"])
        bg_row.grid(row=r, column=1, sticky="w", padx=12, pady=6)
        for val, t in [("none", "None"), ("linear", "Linear"), ("shirley", "Shirley")]:
            tk.Radiobutton(
                bg_row,
                text=t,
                variable=self._bg_var,
                value=val,
                bg=COLORS["card"],
                fg=COLORS["text"],
                selectcolor=COLORS["primary_light"],
                activebackground=COLORS["card"],
                activeforeground=COLORS["primary"],
                font=FONTS["body"],
                cursor="hand2",
            ).pack(side="left", padx=(0, 10))
        r += 1

        # Intensity normalization
        self._norm_var = tk.BooleanVar(value=True)
        label("Intensity")
        norm_row = tk.Frame(parent, bg=COLORS["card"])
        norm_row.grid(row=r, column=1, sticky="w", padx=12, pady=6)
        tk.Checkbutton(
            norm_row,
            text="Normalize 0–1",
            variable=self._norm_var,
            bg=COLORS["card"],
            fg=COLORS["text"],
            selectcolor=COLORS["primary_light"],
            activebackground=COLORS["card"],
            activeforeground=COLORS["primary"],
            font=FONTS["body"],
            cursor="hand2",
            command=self._refresh_canvas,
        ).pack(side="left")
        r += 1

        # Constraints
        section("Constraints")

        # Lock centers
        self._fix_centers_var = tk.BooleanVar(value=True)
        label("Lock centers")
        lock_row = tk.Frame(parent, bg=COLORS["card"])
        lock_row.grid(row=r, column=1, sticky="w", padx=12, pady=6)
        tk.Checkbutton(
            lock_row,
            text="Lock at input values",
            variable=self._fix_centers_var,
            bg=COLORS["card"],
            fg=COLORS["text"],
            selectcolor=COLORS["primary_light"],
            activebackground=COLORS["card"],
            activeforeground=COLORS["primary"],
            font=FONTS["body"],
            cursor="hand2",
        ).pack(side="left")
        r += 1

        # Spin-orbit constraint
        self._so_var = tk.StringVar(value="None")
        label("Spin-orbit")
        so_cmb = ttk.Combobox(parent, textvariable=self._so_var,
                              values=["None", "S2p", "Au4f", "Ti2p", "Cu2p", "Zn2p",
                                      "Fe2p", "Ni2p", "Co2p", "Mn2p", "Mo3d", "W4f", "Pt4f"],
                              state="readonly", width=10)
        field(so_cmb)
        r += 1

        # Peaks
        section("Peaks")
        cols = ("center", "fwhm", "eta")
        self._peak_tree = ttk.Treeview(parent, columns=cols, show="headings", height=6)
        self._peak_tree.heading("center", text="Center (eV)")
        self._peak_tree.heading("fwhm", text="FWHM (eV)")
        self._peak_tree.heading("eta", text="η")
        self._peak_tree.column("center", width=90, anchor="center")
        self._peak_tree.column("fwhm", width=85, anchor="center")
        self._peak_tree.column("eta", width=45, anchor="center")
        self._peak_tree.grid(row=r, column=0, columnspan=2, sticky="ew", padx=12, pady=(6, 4))
        self._peak_tree.bind("<Double-1>", lambda _e: self._edit_peak())
        self._peak_tree.bind("<Return>", lambda _e: self._edit_peak())
        r += 1

        btns = tk.Frame(parent, bg=COLORS["card"])
        btns.grid(row=r, column=0, columnspan=2, sticky="ew", padx=12, pady=(4, 6))
        btns.grid_columnconfigure(0, weight=1)
        btns.grid_columnconfigure(1, weight=1)
        btns.grid_columnconfigure(2, weight=1)
        btns.grid_columnconfigure(3, weight=1)
        btns.grid_columnconfigure(4, weight=1)
        btns.grid_columnconfigure(5, weight=1)
        _btn(btns, "+ Add", self._add_peak).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        _btn(btns, "Auto", self._auto_detect_peaks).grid(row=0, column=1, sticky="ew", padx=(0, 6))
        _btn(btns, "Edit", self._edit_peak).grid(row=0, column=2, sticky="ew", padx=(0, 6))
        _btn(btns, "Remove", self._remove_peak, danger=True).grid(row=0, column=3, sticky="ew", padx=(0, 6))
        _btn(btns, "↩ Undo", self._undo).grid(row=0, column=4, sticky="ew", padx=(0, 6))
        _btn(btns, "↪ Redo", self._redo).grid(row=0, column=5, sticky="ew")
        r += 1

        # Chemical state hints
        section("Chemical state hints")
        self._assign_enable = tk.BooleanVar(value=True)
        label("Suggest")
        sug = tk.Frame(parent, bg=COLORS["card"])
        sug.grid(row=r, column=1, sticky="w", padx=12, pady=6)
        tk.Checkbutton(
            sug,
            text="Suggest assignment",
            variable=self._assign_enable,
            bg=COLORS["card"],
            fg=COLORS["text"],
            selectcolor=COLORS["primary_light"],
            activebackground=COLORS["card"],
            activeforeground=COLORS["primary"],
            font=FONTS["body"],
            cursor="hand2",
            command=self._refresh_canvas,
        ).pack(side="left")
        r += 1

        self._assign_core = tk.StringVar(value="Auto")
        label("Core")
        core_cmb = ttk.Combobox(parent, textvariable=self._assign_core, values=["Auto", "C1s", "O1s", "N1s", "S2p", "F1s"], state="readonly", width=14)
        field(core_cmb)
        core_cmb.bind("<<ComboboxSelected>>", lambda _e: self._refresh_canvas())
        r += 1

        self._assign_tol = tk.StringVar(value="1.0")
        label("Tolerance (eV)")
        field(tk.Entry(parent, textvariable=self._assign_tol, width=10, **entry_kw))
        r += 1

        self._assign_topn = tk.StringVar(value="1")
        label("Top N")
        field(tk.Entry(parent, textvariable=self._assign_topn, width=10, **entry_kw))
        r += 1

        # Actions
        section("Actions")
        actions = tk.Frame(parent, bg=COLORS["card"])
        actions.grid(row=r, column=0, columnspan=2, sticky="ew", padx=12, pady=(8, 10))
        for i in range(2):
            actions.grid_columnconfigure(i, weight=1)

        _btn(actions, "🤖  Auto Fit (DL)", self._run_auto_fit, primary=True).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        _btn(actions, "▶  Fit", self._run_fit, primary=True).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        _btn(actions, "💾  Save params", self._save_params).grid(row=2, column=0, sticky="ew", padx=(0, 6), pady=(0, 6))
        _btn(actions, "📂  Load params", self._load_params).grid(row=2, column=1, sticky="ew", pady=(0, 6))
        _btn(actions, "⬇  Export CSV", self._export_csv).grid(row=3, column=0, sticky="ew", padx=(0, 6), pady=(0, 6))
        _btn(actions, "⬇  Export JSON", self._export_fit_json).grid(row=3, column=1, sticky="ew", pady=(0, 6))
        _btn(actions, "⬇  Stacked Figure", self._export_stacked_figure).grid(row=4, column=0, sticky="ew", padx=(0, 6), pady=(0, 6))
        _btn(actions, "⬇  Export Figure", self._export_figure).grid(row=4, column=1, sticky="ew", pady=(0, 6))
        _btn(actions, "✕  Clear Fit", self._clear_fit, danger=True).grid(row=5, column=0, columnspan=2, sticky="ew")

        self._auto_range()

    def _build_canvas(self, parent):
        """右侧 matplotlib 画布 + 结果表格。"""
        _apply_plot_style()
        # 仅主图，无残差子图
        self._fig, self._ax = plt.subplots(
            1, 1, figsize=(7.2, 5.2), dpi=110,
        )
        self._fig.patch.set_facecolor(COLORS["bg"])
        self._ax.set_facecolor(COLORS["list_bg"])

        canvas_frame = tk.Frame(parent, bg=COLORS["bg"],
                                highlightbackground=COLORS["card_border"],
                                highlightthickness=1)
        canvas_frame.pack(fill="both", expand=True)

        self._canvas = FigureCanvasTkAgg(self._fig, master=canvas_frame)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

        tb_frame = tk.Frame(parent, bg=COLORS["bg"])
        tb_frame.pack(fill="x")
        NavigationToolbar2Tk(self._canvas, tb_frame)

        # 点击图加峰
        if self._mpl_click_cid is None:
            self._mpl_click_cid = self._fig.canvas.mpl_connect("button_press_event", self._on_plot_click)

        # 结果区标题
        res_hdr = tk.Frame(parent, bg=COLORS["bg"])
        res_hdr.pack(fill="x", pady=(8, 2))
        tk.Label(res_hdr, text="Fit Results", font=FONTS["section"],
                 fg=COLORS["primary"], bg=COLORS["bg"]).pack(side="left")
        self._r2_var = tk.StringVar(value="")
        tk.Label(res_hdr, textvariable=self._r2_var, font=FONTS["body"],
                 fg=COLORS["success"], bg=COLORS["bg"]).pack(side="right", padx=4)

        # 结果 Treeview
        res_cols = ("peak", "center", "fwhm", "eta", "area", "area_pct")
        self._res_tree = ttk.Treeview(parent, columns=res_cols,
                                      show="headings", height=4)
        headers = {
            "peak":   "#",
            "center": "Center ±σ (eV)",
            "fwhm":   "FWHM ±σ (eV)",
            "eta":    "η ±σ",
            "area":   "Area (a.u.)",
            "area_pct": "Area (%)",
        }
        widths = {"peak": 32, "center": 145, "fwhm": 125, "eta": 95, "area": 90, "area_pct": 85}
        for c in res_cols:
            self._res_tree.heading(c, text=headers[c])
            self._res_tree.column(c, width=widths[c], anchor="center")
        self._res_tree.pack(fill="x")

    # ---------------------------------------------------------------- #
    #  Helpers                                                           #
    # ---------------------------------------------------------------- #

    def _current_spectrum(self):
        name = self._spec_var.get()
        for s in self.spectra:
            if s["base"] == name:
                return s
        return self.spectra[0] if self.spectra else None

    def _auto_range(self):
        s = self._current_spectrum()
        if s is None:
            return
        self._range_lo.set(f"{float(np.min(s['x'])):.2f}")
        self._range_hi.set(f"{float(np.max(s['x'])):.2f}")

    def _parse_range(self):
        try:
            a, b = float(self._range_lo.get()), float(self._range_hi.get())
            return (min(a, b), max(a, b))
        except ValueError:
            return None

    def _on_spectrum_changed(self):
        self._auto_range()
        self._fit_result = None
        self._bg_cache_key = None
        self._refresh_canvas()

    # ---------------------------------------------------------------- #
    #  Undo / Redo                                                       #
    # ---------------------------------------------------------------- #

    def _push_undo(self):
        """保存当前峰列表快照到 undo 栈，清空 redo 栈。"""
        import copy
        self._undo_stack.append(copy.deepcopy(self._peak_guesses))
        self._redo_stack.clear()

    def _undo(self):
        if not self._undo_stack:
            return
        import copy
        self._redo_stack.append(copy.deepcopy(self._peak_guesses))
        self._peak_guesses = self._undo_stack.pop()
        self._sync_peak_tree()
        self._fit_result = None
        self._refresh_canvas()

    def _redo(self):
        if not self._redo_stack:
            return
        import copy
        self._undo_stack.append(copy.deepcopy(self._peak_guesses))
        self._peak_guesses = self._redo_stack.pop()
        self._sync_peak_tree()
        self._fit_result = None
        self._refresh_canvas()

    def _sync_peak_tree(self):
        """同步 Treeview 显示与 _peak_guesses。"""
        for row in self._peak_tree.get_children():
            self._peak_tree.delete(row)
        for pg in self._peak_guesses:
            self._peak_tree.insert("", "end", values=(
                f"{pg['center']:.2f}",
                f"{pg['fwhm']:.2f}",
                f"{pg['eta']:.2f}",
            ))

    # ---------------------------------------------------------------- #
    #  峰管理                                                            #
    # ---------------------------------------------------------------- #

    def _add_peak(self):
        rng = self._parse_range()
        default = f"{(rng[0] + rng[1]) / 2:.2f}" if rng else ""
        dlg = _AddPeakDialog(self, default_center=default)
        if dlg.result:
            self._push_undo()
            self._peak_guesses.append(dlg.result)
            pg = dlg.result
            self._peak_tree.insert("", "end", values=(
                f"{pg['center']:.2f}",
                f"{pg['fwhm']:.2f}",
                f"{pg['eta']:.2f}",
            ))
            self._fit_result = None
            self._refresh_canvas()

    def _edit_peak(self):
        sel = self._peak_tree.selection()
        if not sel:
            return
        item = sel[0]
        idx = self._peak_tree.index(item)
        if not (0 <= idx < len(self._peak_guesses)):
            return

        self._push_undo()
        pg0 = self._peak_guesses[idx]
        dlg = _AddPeakDialog(
            self,
            default_center=f"{pg0.get('center', 0.0):.2f}",
            default_fwhm=f"{pg0.get('fwhm', 1.0):.2f}",
            default_eta=f"{pg0.get('eta', 0.3):.2f}",
        )
        if not dlg.result:
            return

        self._peak_guesses[idx] = dlg.result
        pg = dlg.result
        self._peak_tree.item(item, values=(
            f"{pg['center']:.2f}",
            f"{pg['fwhm']:.2f}",
            f"{pg['eta']:.2f}",
        ))
        self._fit_result = None
        self._refresh_canvas()

    def _remove_peak(self):
        sel = self._peak_tree.selection()
        if not sel:
            return
        self._push_undo()
        for item in reversed(sel):
            idx = self._peak_tree.index(item)
            self._peak_tree.delete(item)
            if 0 <= idx < len(self._peak_guesses):
                self._peak_guesses.pop(idx)
        self._fit_result = None
        self._refresh_canvas()

    def _clear_fit(self):
        self._fit_result = None
        for row in self._res_tree.get_children():
            self._res_tree.delete(row)
        self._r2_var.set("")
        self._refresh_canvas()

    # ---------------------------------------------------------------- #
    #  画布绘制                                                           #
    # ---------------------------------------------------------------- #

    def _clear_peak_list(self):
        self._peak_guesses = []
        for row in self._peak_tree.get_children():
            self._peak_tree.delete(row)

    def _append_peak_guess(self, pg: dict):
        self._peak_guesses.append(pg)
        self._peak_tree.insert(
            "",
            "end",
            values=(
                f"{pg['center']:.2f}",
                f"{pg['fwhm']:.2f}",
                f"{pg['eta']:.2f}",
            ),
        )

    def _get_fit_data(self):
        """
        返回当前显示范围内的 (x_sorted, y_sorted, y_bg, y_corr)。
        背景按当前 UI 选择计算，带缓存避免重复 Shirley 迭代。
        """
        s = self._current_spectrum()
        if s is None:
            return None
        rng = self._parse_range()
        bg_type = self._bg_var.get()
        norm_on = getattr(self, "_norm_var", None) is not None and self._norm_var.get()

        # 缓存键：谱名 + 范围 + 背景类型 + 归一化开关
        cache_key = (s["base"], rng, bg_type, norm_on)
        if self._bg_cache_key == cache_key and self._bg_cache_data is not None:
            return self._bg_cache_data

        x_all = np.asarray(s["x"])
        y_all = np.asarray(s.get("y_norm", s["y"]))
        if rng:
            lo, hi = rng
            mask = (x_all >= lo) & (x_all <= hi)
            x_d, y_d = x_all[mask], y_all[mask]
        else:
            x_d, y_d = x_all, y_all

        if len(x_d) < 5:
            return None

        idx = np.argsort(x_d)
        x_d, y_d = x_d[idx], y_d[idx]

        if norm_on:
            y0 = float(np.nanmin(y_d))
            y1 = float(np.nanmax(y_d))
            if (y1 - y0) > 0:
                y_d = (y_d - y0) / (y1 - y0)

        if bg_type == "shirley":
            y_bg = shirley_background(x_d, y_d)
        elif bg_type == "linear":
            y_bg = linear_background(x_d, y_d)
        else:
            y_bg = np.zeros_like(y_d)
        y_corr = y_d - y_bg

        result = (x_d, y_d, y_bg, y_corr)
        self._bg_cache_key = cache_key
        self._bg_cache_data = result
        return result

    def _auto_detect_peaks(self):
        if not _HAS_SCIPY_SIGNAL:
            messagebox.showerror(
                "Missing dependency",
                "Auto peak detect requires scipy.signal (SciPy). Please install SciPy and try again.",
                parent=self,
            )
            return

        data = self._get_fit_data()
        if data is None:
            messagebox.showwarning("No data", "No spectrum data in the current range.", parent=self)
            return
        x_d, _y_d, _y_bg, y_corr = data

        # 平滑：窗口长度取奇数，且不超过数据长度
        n = len(y_corr)
        win = min(21, n if n % 2 == 1 else n - 1)
        win = max(win, 5) if n >= 5 else n
        if win >= 5 and win % 2 == 1:
            y_s = savgol_filter(y_corr, window_length=win, polyorder=2, mode="interp")
        else:
            y_s = y_corr

        y_pos = np.maximum(y_s, 0.0)
        y_max = float(np.max(y_pos)) if np.max(y_pos) > 0 else 1.0

        # prominence/height 为相对阈值，避免不同谱强度差异
        peaks, props = find_peaks(
            y_pos,
            prominence=max(1e-12, 0.04 * y_max),
            height=max(1e-12, 0.02 * y_max),
            distance=max(1, int(0.02 * n)),
        )
        if peaks is None or len(peaks) == 0:
            messagebox.showinfo("Auto detect", "No peaks found with current thresholds.", parent=self)
            return

        prominences = props.get("prominences", np.ones_like(peaks, dtype=float))
        order = np.argsort(prominences)[::-1]
        top_n = int(min(8, len(order)))
        keep = np.sort(peaks[order[:top_n]])

        # 用检测结果替换当前峰列表
        self._push_undo()
        self._clear_peak_list()
        for pi in keep:
            self._append_peak_guess({"center": float(x_d[int(pi)]), "fwhm": 1.0, "eta": 0.3})

        self._fit_result = None
        self._refresh_canvas()

    def _on_plot_click(self, event):
        if event.inaxes != self._ax or event.xdata is None:
            return
        # 仅双击添加峰，避免单击误触
        if not getattr(event, "dblclick", False):
            return
        rng = self._parse_range()
        x0 = float(event.xdata)
        if rng:
            lo, hi = rng
            if not (lo <= x0 <= hi):
                return
        self._push_undo()
        self._append_peak_guess({"center": x0, "fwhm": 1.0, "eta": 0.3})
        self._fit_result = None
        self._refresh_canvas()

    def _refresh_canvas(self):
        self._ax.cla()

        s = self._current_spectrum()
        if s is None:
            self._canvas.draw_idle()
            return

        data = self._get_fit_data()
        if data is None:
            self._canvas.draw_idle()
            return
        x_d, y_d, _y_bg, _y_corr = data

        colors = self._PEAK_COLORS
        # 数据散点（投稿风格：灰点更大）
        self._ax.plot(
            x_d,
            y_d,
            "o",
            ms=5.0,
            mec="none",
            color="#6b7280",
            alpha=0.9,
            label="Measured",
            zorder=2,
            rasterized=True,
        )

        if self._fit_result and self._fit_result.get("success"):
            r  = self._fit_result
            xf = r["x_fit"]
            rng = self._parse_range()
            core_auto = guess_core_from_range(*rng) if rng else None

            # 背景：保持可视但不抢戏（不进 legend）
            self._ax.plot(xf, r["y_bg"], "--", color="#9ca3af",
                          lw=1.6, label="_nolegend_", zorder=3)

            # 各峰（填色 + 轮廓）
            for i, (py, fp) in enumerate(zip(r["peaks_y"], r["params"])):
                c = colors[i % 10]
                self._ax.fill_between(xf, r["y_bg"], r["y_bg"] + py,
                                      alpha=0.20, color=c)
                # 图例仅显示 "Peak N"；详细化学态信息在右侧 Chemical state hints 面板
                peak_label = f"Peak {i + 1}"

                # 分峰：更饱和色 + 实线（需要可改成虚线以区分“charged”等）
                self._ax.plot(
                    xf,
                    r["y_bg"] + py,
                    lw=2.2,
                    color=c,
                    label=peak_label,
                )
                # 峰位虚线（示例图风格）
                self._ax.axvline(fp["center"], color="#111827", lw=1.1, ls="--", alpha=0.55, zorder=1)

            # 总拟合（黑色粗线）
            self._ax.plot(
                xf,
                r["y_bg"] + r["y_total_fit"],
                color="#111827",
                lw=2.8,
                label="Cumulative fitted line",
                zorder=5,
            )

            # 残差
            # （应用户要求，不再显示残差曲线）
        else:
            # 仅显示峰猜测竖线
            for i, pg in enumerate(self._peak_guesses):
                self._ax.axvline(pg["center"], color=colors[i % 10],
                                 alpha=0.5, linestyle="--", lw=1.3,
                                 label=f"Guess {i + 1}:  {pg['center']:.2f} eV")

        # 主图样式
        self._ax.set_xlim(max(x_d), min(x_d))
        self._ax.set_xlabel("Binding Energy (eV)")
        self._ax.set_ylabel("Intensity (arb. units)")

        # Element label: upper-right corner based on x range
        rng_for_label = self._parse_range()
        if rng_for_label:
            core_label = guess_core_from_range(*rng_for_label)
        else:
            core_label = guess_core_from_range(float(np.min(x_d)), float(np.max(x_d)))
        if core_label:
            # Pretty format: "C1s", "N1s", etc.
            pretty = core_label[0]  + core_label[1:]
            self._ax.text(
                0.97, 0.97,
                pretty,
                transform=self._ax.transAxes,
                ha="right", va="top",
                fontsize=15,
                fontweight="bold",
                color="#111827",
            )

        # Paper-style axis / ticks
        self._ax.tick_params(direction="out", length=6, width=1.4, colors="#111827")
        for spine in self._ax.spines.values():
            spine.set_color("#111827")
            spine.set_linewidth(1.4)

        self._ax.grid(False)

        leg = self._ax.legend(
            loc="upper left",
            frameon=False,
            borderpad=0.6,
            handlelength=2.6,
            labelspacing=0.45,
        )

        self._fig.tight_layout(pad=1.2)
        self._canvas.draw_idle()

    # ---------------------------------------------------------------- #
    #  拟合                                                              #
    # ---------------------------------------------------------------- #

    def _run_auto_fit(self):
        """DL 自动拟合：选好区域后一键填峰并立即拟合。"""
        if not _HAS_AUTOFIT:
            messagebox.showerror(
                "Auto Fit unavailable",
                "autofit_engine 加载失败：\n"
                f"{_AUTOFIT_IMPORT_ERR}\n\n"
                "需要安装 PyTorch / lmfit：\n"
                "  pip install torch lmfit",
                parent=self,
            )
            return

        s = self._current_spectrum()
        if s is None:
            messagebox.showwarning("No spectrum",
                                   "Please load IBW files in the main window first.",
                                   parent=self)
            return

        rng = self._parse_range()
        if rng is None:
            messagebox.showwarning("No range",
                                   "请先填写有效的拟合区域 (Range eV)。",
                                   parent=self)
            return

        # 用原始（未归一化）强度，让 Shirley/DL 拿到真实计数；BE 排序由引擎自己处理
        be = np.asarray(s["x"], dtype=float)
        y_in = np.asarray(s.get("y", s.get("y_norm")), dtype=float)

        # 显示忙碌光标
        try:
            self.config(cursor="watch")
            self.update_idletasks()
        except Exception:
            pass

        try:
            res = _dl_auto_fit(be=be, y=y_in, fit_range=rng)
        except Exception as e:
            messagebox.showerror("Auto Fit failed", f"{type(e).__name__}: {e}", parent=self)
            return
        finally:
            try:
                self.config(cursor="")
            except Exception:
                pass

        if not res.success or not res.peak_guesses:
            messagebox.showwarning("Auto Fit", res.message or "未检测到峰。", parent=self)
            return

        # 替换当前峰列表
        self._push_undo()
        self._clear_peak_list()
        for pg in res.peak_guesses:
            self._append_peak_guess(pg)

        # DL 已经做了 Shirley + lmfit 精修；这里再走一遍 _run_fit 让结果落到现有 UI 流水线上
        self._fit_result = None
        try:
            self._run_fit()
        except Exception:
            self._refresh_canvas()

        msg_extra = ""
        if res.rms == res.rms:
            msg_extra = f"  rms={res.rms:.3f}"
        messagebox.showinfo(
            "Auto Fit",
            f"{res.orbital} · model={res.model_name}\n"
            f"Detected {res.n_peaks} peak(s){msg_extra}",
            parent=self,
        )

    def _run_fit(self):
        s = self._current_spectrum()
        if s is None:
            messagebox.showwarning("No spectrum",
                                   "Please load IBW files in the main window first.",
                                   parent=self)
            return
        if not self._peak_guesses:
            messagebox.showwarning("No peaks",
                                   "Click [+ Add] to add at least one peak guess.",
                                   parent=self)
            return
        # Prefer normalized y if available, and (optionally) normalize within fit range
        y_in = s.get("y_norm", s["y"])
        if getattr(self, "_norm_var", None) is not None and self._norm_var.get():
            # Range-normalize to 0–1 (same logic as _get_fit_data)
            fr = self._parse_range()
            x_all = np.asarray(s["x"], dtype=float)
            y_all = np.asarray(y_in, dtype=float)
            if fr is not None:
                lo, hi = fr
                m = (x_all >= lo) & (x_all <= hi)
                y_sel = y_all[m]
            else:
                y_sel = y_all
            if len(y_sel):
                y0 = float(np.nanmin(y_sel))
                y1 = float(np.nanmax(y_sel))
                if (y1 - y0) > 0:
                    y_in = (y_all - y0) / (y1 - y0)

        result = fit_xps(
            s["x"], y_in,
            peak_guesses=self._peak_guesses,
            bg_type=self._bg_var.get(),
            fit_range=self._parse_range(),
            spin_orbit_mode=self._so_var.get() if getattr(self, "_so_var", None) else None,
            fix_centers=self._fix_centers_var.get() if getattr(self, "_fix_centers_var", None) else False,
        )
        self._fit_result = result

        if not result["success"]:
            messagebox.showerror("Fit failed", result["message"], parent=self)
            return

        # cache by spectrum name for later stacked export
        try:
            self._fit_results_by_spec[self._spec_var.get()] = result
        except Exception:
            pass

        self._populate_results(result)
        self._refresh_canvas()

    @staticmethod
    def _compute_peak_areas(xf, params):
        """计算各峰面积及总面积。返回 (areas, total_area)。"""
        areas = []
        for fp in params:
            area = float(_trapezoid(
                pseudo_voigt(xf, fp["center"], fp["amplitude"],
                             fp["fwhm"], fp["eta"]), xf))
            areas.append(abs(area))
        total_area = float(np.sum(areas)) if areas else 0.0
        return areas, total_area

    def _populate_results(self, r):
        for row in self._res_tree.get_children():
            self._res_tree.delete(row)
        xf = r["x_fit"]
        areas, total_area = self._compute_peak_areas(xf, r["params"])

        for i, (fp, fe, area_abs) in enumerate(zip(r["params"], r["errors"], areas)):
            pct = (100.0 * area_abs / total_area) if total_area > 0 else 0.0
            self._res_tree.insert("", "end", values=(
                f"{i + 1}",
                f"{fp['center']:.3f} ±{fe['center']:.3f}",
                f"{fp['fwhm']:.3f} ±{fe['fwhm']:.3f}",
                f"{fp['eta']:.3f} ±{fe['eta']:.3f}",
                f"{area_abs:.2f}",
                f"{pct:.1f}",
            ))
        self._r2_var.set(f"R²  =  {r['r_squared']:.5f}")

    # ---------------------------------------------------------------- #
    #  导出                                                              #
    # ---------------------------------------------------------------- #

    def _save_params(self):
        path = filedialog.asksaveasfilename(
            title="Save fit parameters",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile=f"{self._spec_var.get()}_fit_params.json",
            parent=self,
        )
        if not path:
            return

        payload = {
            "schema": "xps_fit_params_v1",
            "spectrum": self._spec_var.get(),
            "fit_range": self._parse_range(),
            "bg_type": self._bg_var.get(),
            "spin_orbit_mode": self._so_var.get() if getattr(self, "_so_var", None) else "None",
            "peak_guesses": list(self._peak_guesses),
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showerror("Save failed", str(e), parent=self)
            return
        messagebox.showinfo("Saved", f"Parameters saved to:\n{path}", parent=self)

    def _load_params(self):
        path = filedialog.askopenfilename(
            title="Load fit parameters",
            filetypes=[("JSON files", "*.json")],
            parent=self,
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            messagebox.showerror("Load failed", str(e), parent=self)
            return

        if not isinstance(payload, dict):
            messagebox.showerror("Load failed", "Invalid JSON format.", parent=self)
            return

        spec_name = payload.get("spectrum")
        if spec_name and any(s["base"] == spec_name for s in self.spectra):
            self._spec_var.set(spec_name)

        fr = payload.get("fit_range")
        if isinstance(fr, (list, tuple)) and len(fr) == 2:
            try:
                self._range_lo.set(f"{float(fr[0]):.2f}")
                self._range_hi.set(f"{float(fr[1]):.2f}")
            except Exception:
                pass

        bg = payload.get("bg_type")
        if bg in ("none", "linear", "shirley"):
            self._bg_var.set(bg)

        so = payload.get("spin_orbit_mode")
        from peak_fit import _SPIN_ORBIT_PRESETS
        valid_so = {"None", ""} | set(_SPIN_ORBIT_PRESETS.keys()) | {None}
        if getattr(self, "_so_var", None) is not None and so in valid_so:
            self._so_var.set("None" if so in (None, "") else so)

        peaks = payload.get("peak_guesses")
        if isinstance(peaks, list):
            self._clear_peak_list()
            for pg in peaks:
                try:
                    cen = float(pg.get("center"))
                    fwhm = float(pg.get("fwhm", 1.0))
                    eta = float(pg.get("eta", 0.3))
                    self._append_peak_guess({"center": cen, "fwhm": fwhm, "eta": eta})
                except Exception:
                    continue

        self._fit_result = None
        self._refresh_canvas()
        messagebox.showinfo("Loaded", f"Parameters loaded from:\n{path}", parent=self)

    def _export_csv(self):
        if self._fit_result is None or not self._fit_result.get("success"):
            messagebox.showwarning("No fit",
                                   "Run a fit first, then export.", parent=self)
            return
        path = filedialog.asksaveasfilename(
            title="Save Fit Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=f"{self._spec_var.get()}_fit.csv",
            parent=self,
        )
        if not path:
            return
        r = self._fit_result
        xf = r["x_fit"]
        residual = r["y_corrected"] - r["y_total_fit"]
        rmse = float(np.sqrt(np.mean(residual ** 2))) if len(residual) else float("nan")
        mae = float(np.mean(np.abs(residual))) if len(residual) else float("nan")

        areas, total_area = self._compute_peak_areas(xf, r["params"])

        df_dict = {
            "BE_eV":      xf,
            "Intensity":  r["y_raw"],
            "Background": r["y_bg"],
            "Corrected":  r["y_corrected"],
            "TotalFit":   r["y_total_fit"],
            "Residual":   residual,
        }
        for i, py in enumerate(r["peaks_y"]):
            df_dict[f"Peak{i + 1}"] = py
        pd.DataFrame(df_dict).to_csv(path, index=False)
        with open(path, "a", newline="", encoding="utf-8") as f:
            f.write("\nFit Meta\n")
            f.write(f"ExportedAt,{datetime.now().isoformat(timespec='seconds')}\n")
            fr = self._parse_range()
            if fr:
                f.write(f"FitRangeLo_eV,{fr[0]:.6f}\n")
                f.write(f"FitRangeHi_eV,{fr[1]:.6f}\n")
            f.write(f"Background,{self._bg_var.get()}\n")
            f.write(f"SpinOrbitMode,{(r.get('spin_orbit_mode') or 'None')}\n")
            f.write(f"R_squared,{r['r_squared']:.8f}\n")
            f.write(f"RMSE_corrected,{rmse:.8g}\n")
            f.write(f"MAE_corrected,{mae:.8g}\n")

            f.write("\nFit Parameters\n")
            f.write("Peak,Center_eV,Center_err,FWHM_eV,FWHM_err,"
                    "eta,eta_err,Amplitude,Amp_err,Area,AreaPct\n")
            for i, (fp, fe, area_abs) in enumerate(zip(r["params"], r["errors"], areas)):
                pct = (100.0 * area_abs / total_area) if total_area > 0 else 0.0
                f.write(f"{i+1},"
                        f"{fp['center']:.4f},{fe['center']:.4f},"
                        f"{fp['fwhm']:.4f},{fe['fwhm']:.4f},"
                        f"{fp['eta']:.4f},{fe['eta']:.4f},"
                        f"{fp['amplitude']:.4f},{fe['amplitude']:.4f},"
                        f"{area_abs:.4f},{pct:.2f}\n")
        messagebox.showinfo("Saved",
                            f"Fit results exported to:\n{path}", parent=self)

    def _export_fit_json(self):
        if self._fit_result is None or not self._fit_result.get("success"):
            messagebox.showwarning("No fit", "Run a fit first, then export.", parent=self)
            return

        path = filedialog.asksaveasfilename(
            title="Export fit report (JSON)",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile=f"{self._spec_var.get()}_fit_report.json",
            parent=self,
        )
        if not path:
            return

        r = self._fit_result
        xf = r["x_fit"]
        residual_arr = r["y_corrected"] - r["y_total_fit"]
        rmse = float(np.sqrt(np.mean(residual_arr ** 2))) if len(residual_arr) else float("nan")
        mae = float(np.mean(np.abs(residual_arr))) if len(residual_arr) else float("nan")

        areas, total_area = self._compute_peak_areas(xf, r["params"])

        peaks = []
        for i, (fp, fe, area_abs) in enumerate(zip(r["params"], r["errors"], areas), start=1):
            pct = (100.0 * area_abs / total_area) if total_area > 0 else 0.0
            peaks.append(
                {
                    "peak": i,
                    "params": fp,
                    "errors_1sigma": fe,
                    "area": area_abs,
                    "area_pct": pct,
                }
            )

        payload = {
            "schema": "xps_fit_report_v1",
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "spectrum": self._spec_var.get(),
            "fit_range": self._parse_range(),
            "bg_type": self._bg_var.get(),
            "spin_orbit_mode": r.get("spin_orbit_mode") or "None",
            "metrics": {
                "r_squared": r.get("r_squared"),
                "rmse_corrected": rmse,
                "mae_corrected": mae,
                "n_points": int(len(xf)),
            },
            "peaks": peaks,
            "curves": {
                "x_fit": xf.tolist(),
                "y_raw": r["y_raw"].tolist(),
                "y_bg": r["y_bg"].tolist(),
                "y_corrected": r["y_corrected"].tolist(),
                "y_total_fit": r["y_total_fit"].tolist(),
                "residual": residual_arr.tolist(),
                "peaks_y": [py.tolist() for py in r["peaks_y"]],
            },
        }

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showerror("Export failed", str(e), parent=self)
            return
        messagebox.showinfo("Saved", f"Fit report exported to:\n{path}", parent=self)

    def _export_stacked_figure(self):
        """
        导出上下两幅 stacked panel，对比两条谱的拟合结果。
        需要两条谱都已成功拟合（会从缓存里取 fit result）。
        """
        names = [s["base"] for s in self.spectra]
        if len(names) < 2:
            messagebox.showwarning("Need two spectra", "Load at least two spectra first.", parent=self)
            return

        # 默认：当前谱为 Top，第二条为 Bottom
        top_default = self._spec_var.get() if self._spec_var.get() in names else names[0]
        bottom_default = next((n for n in names if n != top_default), names[1])
        dlg = _StackExportDialog(self, names, default_a=top_default, default_b=bottom_default)
        if not dlg.result:
            return
        top_name, bottom_name = dlg.result

        r_top = self._fit_results_by_spec.get(top_name)
        r_bot = self._fit_results_by_spec.get(bottom_name)
        if not (r_top and r_top.get("success")):
            messagebox.showwarning("Missing fit", f"Top spectrum '{top_name}' has no successful fit yet.", parent=self)
            return
        if not (r_bot and r_bot.get("success")):
            messagebox.showwarning("Missing fit", f"Bottom spectrum '{bottom_name}' has no successful fit yet.", parent=self)
            return

        path = filedialog.asksaveasfilename(
            title="Export stacked figure",
            defaultextension=".tif",
            filetypes=[("TIFF files", "*.tif;*.tiff"), ("PNG files", "*.png")],
            initialfile=f"{top_name}__{bottom_name}_stacked.tif",
            parent=self,
        )
        if not path:
            return

        _apply_plot_style()
        fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.8), dpi=_PLOT_STYLE.get("figure.dpi", 120), sharex=True)
        fig.patch.set_facecolor("white")

        def _draw_panel(ax, name: str, r: dict):
            ax.set_facecolor("white")
            xf = r["x_fit"]
            ax.plot(
                xf,
                r["y_raw"],
                "o",
                ms=5.0,
                mec="none",
                color="#6b7280",
                alpha=0.9,
                label="Measured",
                zorder=2,
                rasterized=True,
            )
            ax.plot(xf, r["y_bg"], "--", color="#9ca3af", lw=1.6, label="_nolegend_", zorder=3)

            colors = self._PEAK_COLORS
            for i, (py, fp) in enumerate(zip(r["peaks_y"], r["params"])):
                c = colors[i % 10]
                ax.fill_between(xf, r["y_bg"], r["y_bg"] + py, alpha=0.22, color=c, linewidth=0)
                ax.plot(xf, r["y_bg"] + py, lw=2.2, color=c, label=f"Peak {i + 1}", zorder=4)
                ax.axvline(fp["center"], color="#111827", lw=1.1, ls="--", alpha=0.55, zorder=1)

            ax.plot(
                xf,
                r["y_bg"] + r["y_total_fit"],
                color="#111827",
                lw=2.8,
                label="Cumulative fitted line",
                zorder=5,
            )

            # Element label: upper-right corner
            xf = r["x_fit"]
            core_lbl = guess_core_from_range(float(np.min(xf)), float(np.max(xf)))
            if core_lbl:
                pretty_lbl = core_lbl[0] + " " + core_lbl[1:]
                ax.text(
                    0.97, 0.97,
                    pretty_lbl,
                    transform=ax.transAxes,
                    ha="right", va="top",
                    fontsize=15,
                    fontweight="bold",
                    color="#111827",
                )

            ax.tick_params(direction="out", length=6, width=1.4, colors="#111827")
            for spine in ax.spines.values():
                spine.set_color("#111827")
                spine.set_linewidth(1.4)
            ax.grid(False)
            ax.legend(
                loc="upper left",
                frameon=False,
                borderpad=0.6,
                handlelength=2.6,
                labelspacing=0.45,
            )

        _draw_panel(axes[0], top_name, r_top)
        _draw_panel(axes[1], bottom_name, r_bot)

        # Shared axes labels / limits (XPS convention: high -> low)
        x_max = max(float(np.max(r_top["x_fit"])), float(np.max(r_bot["x_fit"])))
        x_min = min(float(np.min(r_top["x_fit"])), float(np.min(r_bot["x_fit"])))
        axes[1].set_xlabel("Binding Energy (eV)")
        fig.text(0.03, 0.5, "Intensity (arb. units)", va="center", rotation="vertical")
        axes[0].set_xlim(x_max, x_min)

        fig.tight_layout(rect=(0.06, 0.04, 1.0, 1.0))

        try:
            ext = str(path).lower().split(".")[-1]
            if ext in ("png", "tif", "tiff"):
                fig.savefig(path, dpi=300, bbox_inches="tight")
            else:
                fig.savefig(path, bbox_inches="tight")
        finally:
            plt.close(fig)

        messagebox.showinfo("Saved", f"Stacked figure exported to:\n{path}", parent=self)

    def _export_figure(self):
        path = filedialog.asksaveasfilename(
            title="Export figure",
            defaultextension=".tif",
            filetypes=[
                ("TIFF files", "*.tif;*.tiff"),
                ("SVG files", "*.svg"),
                ("PNG files", "*.png"),
            ],
            initialfile=f"{self._spec_var.get()}_fit.tif",
            parent=self,
        )
        if not path:
            return
        try:
            ext = str(path).lower().split(".")[-1]
            if ext in ("tif", "tiff", "png"):
                self._fig.savefig(path, dpi=300, bbox_inches="tight")
            else:
                # SVG 为矢量；不关闭窗口的 figure
                self._fig.savefig(path, bbox_inches="tight")
        except Exception as e:
            messagebox.showerror("Export failed", str(e), parent=self)
            return
        messagebox.showinfo("Saved", f"SVG exported to:\n{path}", parent=self)

    # ---------------------------------------------------------------- #
    #  清理                                                              #
    # ---------------------------------------------------------------- #

    def _on_close(self):
        try:
            if self._mpl_click_cid is not None:
                self._fig.canvas.mpl_disconnect(self._mpl_click_cid)
        except Exception:
            pass
        plt.close(self._fig)
        self.destroy()
