# ui_theme.py
# ============================================================
# UPS 工具 UI 配色、字体与 TTK 主题
# ============================================================

import tkinter as tk
from tkinter import ttk

# ── 配色（基于 Tailwind CSS 色板，科研风）────────────────────
COLORS = {
    # 背景层
    "bg":            "#f0f2f5",   # 主窗口背景（微蓝灰）
    "card":          "#ffffff",   # 卡片/区块
    "card_border":   "#dde1e7",   # 卡片边框

    # 主色 - 青蓝（专业、科研感）
    "primary":       "#0e7490",   # 主操作色 (cyan-700)
    "primary_hover": "#0c6275",   # 悬停加深
    "primary_light": "#ecfeff",   # 极浅青，用于徽章/标签背景

    # 文字
    "text":          "#111827",   # 主文字（近黑，更清晰）
    "secondary":     "#6b7280",   # 次要文字（灰）

    # 语义色（用于 log 彩色标注）
    "success":       "#16a34a",   # 绿：加载成功、导出完成
    "error":         "#dc2626",   # 红：失败、错误
    "warning":       "#d97706",   # 琥珀：加载中、导出中

    # 控件
    "button_bg":     "#e5e7eb",   # 次要按钮背景
    "button_hover":  "#d1d5db",   # 次要按钮悬停

    # 列表/输入框
    "list_bg":       "#f9fafb",
    "list_fg":       "#111827",
    "list_select":   "#cffafe",   # 选中行（浅青）

    # 分割线
    "divider":       "#e5e7eb",
}

# ── 字体 ────────────────────────────────────────────────────
FONTS = {
    "title":    ("Segoe UI", 15, "bold"),
    "subtitle": ("Segoe UI", 9),
    "section":  ("Segoe UI", 10, "bold"),
    "body":     ("Segoe UI", 10),
    "mono":     ("Consolas", 10),
    "small":    ("Segoe UI", 9),
    "badge":    ("Segoe UI", 9, "bold"),
}


def apply_ttk_theme(root):
    """将 TTK 控件（Treeview、Combobox、Progressbar）统一为与 COLORS 匹配的风格。"""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")   # clam 是自定义性最好的基础主题
    except Exception:
        pass

    # ── Treeview ──────────────────────────────────────────
    style.configure("Treeview",
        background=COLORS["list_bg"],
        foreground=COLORS["text"],
        fieldbackground=COLORS["list_bg"],
        rowheight=24,
        font=FONTS["body"],
        borderwidth=0,
        relief="flat",
    )
    style.configure("Treeview.Heading",
        background=COLORS["divider"],
        foreground=COLORS["secondary"],
        font=FONTS["section"],
        relief="flat",
        padding=(6, 4),
    )
    style.map("Treeview",
        background=[("selected", COLORS["list_select"])],
        foreground=[("selected", COLORS["text"])],
    )
    style.map("Treeview.Heading",
        background=[("active", COLORS["card_border"])],
        relief=[("active", "flat")],
    )

    # ── Progressbar ────────────────────────────────────────
    style.configure("TProgressbar",
        background=COLORS["primary"],
        troughcolor=COLORS["divider"],
        thickness=4,
        borderwidth=0,
    )

    # ── Combobox ───────────────────────────────────────────
    style.configure("TCombobox",
        fieldbackground=COLORS["list_bg"],
        background=COLORS["card_border"],
        foreground=COLORS["text"],
        selectbackground=COLORS["list_select"],
        selectforeground=COLORS["text"],
        arrowcolor=COLORS["secondary"],
        font=FONTS["body"],
        padding=(4, 2),
    )
    style.map("TCombobox",
        fieldbackground=[("readonly", COLORS["list_bg"])],
        selectbackground=[("readonly", COLORS["list_select"])],
        arrowcolor=[("active", COLORS["primary"])],
    )

    # ── Scrollbar ──────────────────────────────────────────
    style.configure("Vertical.TScrollbar",
        background=COLORS["divider"],
        arrowcolor=COLORS["secondary"],
        troughcolor=COLORS["list_bg"],
        borderwidth=0,
        relief="flat",
        width=10,
    )
    style.map("Vertical.TScrollbar",
        background=[("active", COLORS["card_border"])],
    )


# ── 共享 UI 组件工厂 ─────────────────────────────────────

def create_button(parent, text, command, primary=False, danger=False, **kw):
    """统一按钮工厂：primary=蓝绿主色，danger=红色，默认=灰色。"""
    if primary:
        bg, fg, hover = COLORS["primary"], "#ffffff", COLORS["primary_hover"]
    elif danger:
        bg, fg, hover = "#fef2f2", COLORS["error"], "#fee2e2"
    else:
        bg, fg, hover = COLORS["button_bg"], COLORS["text"], COLORS["button_hover"]

    padx = kw.pop("padx", 14)
    pady = kw.pop("pady", 6)

    btn = tk.Button(parent, text=text, command=command, font=FONTS["body"],
                    bg=bg, fg=fg, activebackground=hover, activeforeground=fg,
                    relief="flat", cursor="hand2",
                    borderwidth=0, highlightthickness=0,
                    padx=padx, pady=pady, **kw)
    btn.bind("<Enter>", lambda _e: btn.configure(bg=hover))
    btn.bind("<Leave>", lambda _e: btn.configure(bg=bg))
    return btn


def create_section(parent, title, **pack_kw):
    """创建带左侧彩条的卡片式分区。返回 inner Frame。"""
    outer = tk.Frame(parent, bg=COLORS["card"],
                     highlightbackground=COLORS["card_border"],
                     highlightthickness=1)
    outer.pack(**pack_kw)

    tk.Frame(outer, bg=COLORS["primary"], width=3).pack(side="left", fill="y")

    right = tk.Frame(outer, bg=COLORS["card"])
    right.pack(side="left", fill="both", expand=True)

    title_bar = tk.Frame(right, bg=COLORS["card"])
    title_bar.pack(fill="x", padx=14, pady=(8, 4))
    tk.Label(title_bar, text=title, font=FONTS["section"],
             fg=COLORS["primary"], bg=COLORS["card"]).pack(side="left")

    tk.Frame(right, bg=COLORS["divider"], height=1).pack(fill="x", padx=14)

    inner = tk.Frame(right, bg=COLORS["card"])
    inner.pack(fill="both", expand=True, padx=14, pady=(8, 10))
    return inner


def create_log_panel(parent):
    """创建带滚动条和颜色标签的日志 Text 面板。返回 Text widget。"""
    log_wrap = tk.Frame(parent, bg=COLORS["card"])
    log_wrap.pack(fill="both", expand=True)
    log_sb = ttk.Scrollbar(log_wrap, orient="vertical")
    log_sb.pack(side="right", fill="y")
    log = tk.Text(
        log_wrap, height=12,
        font=FONTS["mono"],
        bg=COLORS["list_bg"], fg=COLORS["list_fg"],
        relief="flat", highlightthickness=0,
        padx=10, pady=8,
        wrap="word",
        yscrollcommand=log_sb.set,
    )
    log.pack(side="left", fill="both", expand=True)
    log_sb.config(command=log.yview)

    log.tag_configure("ok",   foreground=COLORS["success"])
    log.tag_configure("err",  foreground=COLORS["error"])
    log.tag_configure("warn", foreground=COLORS["warning"])
    log.tag_configure("phi",  foreground=COLORS["primary"])
    log.tag_configure("dim",  foreground=COLORS["secondary"])
    log.tag_configure("bold", font=FONTS["section"])
    return log
