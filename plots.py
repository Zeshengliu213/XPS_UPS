# plots.py
# ============================================================
# UPS 绘图：叠加/分开、双 zoom 面板、SECO/功函数、保存 PNG
# ============================================================

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# He I 光子能量 (eV)，功函数 φ = HV_HEI - BE_cutoff
HV_HEI = 21.22


def find_seco(x, y_norm, search_region=(15.0, 18.0), hv=HV_HEI, min_slope=0.1):
    """
    在给定 BE 范围内找二次电子截止边 (SECO)，并计算功函数 φ = hv - BE_cutoff。
    正确做法：在上升沿最陡处做切线，取切线与基线的交点作为 SECO。

    x, y_norm   : 结合能 (eV) 与归一化强度。
    search_region: (lo, hi) eV。
    min_slope   : 最陡点梯度绝对值的最小阈值（eV⁻¹）。低于此值时认为区域内无有效截止边
                  （可能是噪声或纯平台），默认 0.1 eV⁻¹。

    返回 (BE_cutoff, work_function, aux)：
      - 成功：BE_cutoff 与 work_function 为浮点数，aux 包含绘图所需参数和 reason="ok"。
      - 失败：BE_cutoff 与 work_function 为 None，aux 为含 "reason" 字符串的 dict，
              说明失败原因，供调用方记录日志。
    """
    lo, hi = min(search_region), max(search_region)
    mask = (x >= lo) & (x <= hi)
    n_pts = int(np.sum(mask))
    if not np.any(mask) or n_pts < 5:
        return None, None, {"reason": f"搜索区间 {lo}–{hi} eV 内仅有 {n_pts} 个数据点（至少需要 5 个）"}
    xr = np.asarray(x[mask], dtype=float)
    yr = np.asarray(y_norm[mask], dtype=float)
    idx = np.argsort(xr)
    xr = xr[idx]
    yr = yr[idx]
    grad = np.gradient(yr, xr)
    i_edge = np.argmin(grad)
    x0 = float(xr[i_edge])
    y0 = float(yr[i_edge])
    m = float(grad[i_edge])
    # 斜率过小：区域内无明显截止边（噪声或平台），拒绝检测
    if abs(m) < min_slope:
        return None, None, {
            "reason": (
                f"最陡梯度 |m| = {abs(m):.4f} eV⁻¹ 低于阈值 {min_slope} eV⁻¹，"
                f"搜索区间 {lo}–{hi} eV 内无有效截止边（可能为噪声或平台区）"
            )
        }
    y_baseline = float(np.min(yr))
    BE_cutoff = x0 + (y_baseline - y0) / m
    BE_cutoff = float(np.clip(BE_cutoff, lo, hi))
    work_function = hv - BE_cutoff
    aux = {"x0": x0, "y0": y0, "slope": m, "y_baseline": y_baseline, "lo": lo, "hi": hi, "reason": "ok"}
    return BE_cutoff, work_function, aux


def find_fermi_edge(x, y_norm, search_region=(-1.0, 3.0), min_slope=0.1):
    """
    在给定 BE 范围内找费米边的“第一个截止边”（-1~3 eV 一般覆盖 EF 附近）。
    算法与 find_seco 一致：取最陡处做切线，切线与基线交点作为 edge 位置。

    返回 (BE_edge, aux)：
      - 成功：BE_edge 为浮点数，aux 含绘图所需参数和 reason="ok"
      - 失败：BE_edge 为 None，aux 含 reason 说明
    """
    lo, hi = min(search_region), max(search_region)
    mask = (x >= lo) & (x <= hi)
    n_pts = int(np.sum(mask))
    if not np.any(mask) or n_pts < 5:
        return None, {"reason": f"搜索区间 {lo}–{hi} eV 内仅有 {n_pts} 个数据点（至少需要 5 个）"}

    xr = np.asarray(x[mask], dtype=float)
    yr = np.asarray(y_norm[mask], dtype=float)
    idx = np.argsort(xr)
    xr = xr[idx]
    yr = yr[idx]

    grad = np.gradient(yr, xr)
    # 费米边在该窗口通常表现为“从基线往上升”的最陡上升沿：取最大正斜率
    i_edge = int(np.argmax(grad))
    x0 = float(xr[i_edge])
    y0 = float(yr[i_edge])
    m = float(grad[i_edge])
    if abs(m) < min_slope:
        return None, {
            "reason": (
                f"最陡梯度 |m| = {abs(m):.4f} eV⁻¹ 低于阈值 {min_slope} eV⁻¹，"
                f"搜索区间 {lo}–{hi} eV 内无有效截止边（可能为噪声或平台区）"
            )
        }

    y_baseline = float(np.min(yr))
    BE_edge = x0 + (y_baseline - y0) / m
    BE_edge = float(np.clip(BE_edge, lo, hi))
    aux = {"x0": x0, "y0": y0, "slope": m, "y_baseline": y_baseline, "lo": lo, "hi": hi, "reason": "ok"}
    return BE_edge, aux


def _draw_edge_on_ax(ax, BE_edge, aux, title="Edge", color="C2"):
    """在指定 ax 上画：短基线、短切线、竖虚线与文字标注。"""
    if aux is None:
        return
    lo, hi = aux["lo"], aux["hi"]
    x0, y0 = aux["x0"], aux["y0"]
    m = aux["slope"]
    y_baseline = aux["y_baseline"]

    dx_short = 0.25
    x_lo = max(lo, BE_edge - dx_short)
    x_hi = min(hi, BE_edge + dx_short)
    ax.hlines(y_baseline, x_lo, x_hi, colors=color, linestyles="-", linewidth=1.0, alpha=0.85, zorder=6)
    x_tan = np.linspace(x_lo, x_hi, 20)
    y_tan = y0 + m * (x_tan - x0)
    ax.plot(x_tan, y_tan, color=color, linestyle="-", linewidth=1.0, alpha=0.85, zorder=6)
    ax.axvline(BE_edge, color=color, linestyle=":", linewidth=1.1, alpha=0.9, zorder=6)
    ax.text(
        BE_edge,
        0.98,
        f"{title} = {BE_edge:.2f} eV",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=9,
        color=color,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9),
        zorder=7,
    )


def find_edge_two_line_fit(
    x,
    y_norm,
    search_region=(2.0, -1.0),
    bg_frac=0.22,
    edge_frac=0.18,
    min_pts=8,
):
    """
    在给定窗口内自动选“背景区 + 上升区”，做两条直线拟合并求交点（截止边）。

    选段规则（无需手动指定）：
    - 先裁剪到 search_region，并按 x 升序排序
    - 找 |dy/dx| 最大的点作为 edge 中心，在其附近取 edge_frac*N 个点做“上升区”
    - 背景区从窗口两端中“平均强度更低”的那一端取 bg_frac*N 个点
      （适配不同谱：到底是左端还是右端更像背景不预设）

    返回 (x_intersect, aux)
    aux 含：a1,b1,a2,b2,x_bg,y_bg,x_edge,y_edge,lo,hi,reason
    """
    lo, hi = min(search_region), max(search_region)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y_norm, dtype=float)
    m = (x >= lo) & (x <= hi)
    n = int(np.sum(m))
    if n < max(min_pts, 6):
        return None, {"reason": f"窗口 {lo}–{hi} eV 内点数不足（{n} < {max(min_pts, 6)}）"}

    xr = x[m]
    yr = y[m]
    idx = np.argsort(xr)
    xr = xr[idx]
    yr = yr[idx]

    # 简单平滑一下梯度（不改变曲线形状太多）
    if len(yr) >= 7:
        k = 5
        ker = np.ones(k) / k
        ys = np.convolve(yr, ker, mode="same")
    else:
        ys = yr

    grad = np.gradient(ys, xr)
    i0 = int(np.argmax(np.abs(grad)))

    n_bg = max(min_pts, int(round(bg_frac * n)))
    n_edge = max(min_pts, int(round(edge_frac * n)))
    n_bg = min(n_bg, n // 2)
    n_edge = min(n_edge, n)

    # 背景端：两端各取 n_bg 点，取平均强度更低的一端
    left_mean = float(np.nanmean(yr[:n_bg]))
    right_mean = float(np.nanmean(yr[-n_bg:]))
    if left_mean <= right_mean:
        bg_slice = slice(0, n_bg)
    else:
        bg_slice = slice(n - n_bg, n)

    # 上升区：以最大斜率点为中心取 n_edge 点
    half = n_edge // 2
    i1 = max(0, i0 - half)
    i2 = min(n, i1 + n_edge)
    i1 = max(0, i2 - n_edge)
    edge_slice = slice(i1, i2)

    x_bg = xr[bg_slice]
    y_bg = yr[bg_slice]
    x_edge = xr[edge_slice]
    y_edge = yr[edge_slice]

    if len(x_bg) < 2 or len(x_edge) < 2:
        return None, {"reason": "自动选段后点数不足，无法拟合直线"}

    # 线性拟合：y = a x + b
    a1, b1 = np.polyfit(x_bg, y_bg, 1)
    a2, b2 = np.polyfit(x_edge, y_edge, 1)
    denom = (a2 - a1)
    if np.isclose(denom, 0.0):
        return None, {"reason": "两条拟合直线斜率几乎相同，交点不稳定"}

    x_int = float((b1 - b2) / denom)
    x_int = float(np.clip(x_int, lo, hi))

    aux = {
        "a1": float(a1),
        "b1": float(b1),
        "a2": float(a2),
        "b2": float(b2),
        "x_bg": x_bg,
        "y_bg": y_bg,
        "x_edge": x_edge,
        "y_edge": y_edge,
        "lo": lo,
        "hi": hi,
        "reason": "ok",
    }
    return x_int, aux


def _draw_two_line_fit_on_ax(ax, x_int, aux, title="Edge", color="C2"):
    """把两条拟合线（背景/上升）和交点画在 ax 上，便于核对自动选段是否合理。"""
    if aux is None or aux.get("reason") != "ok" or x_int is None:
        return
    lo, hi = aux["lo"], aux["hi"]
    a1, b1, a2, b2 = aux["a1"], aux["b1"], aux["a2"], aux["b2"]

    # 画短线段：各在线段中点附近显示
    def _short_line(x_seg, a, b, alpha=0.85, lw=1.2):
        if x_seg is None or len(x_seg) < 2:
            return
        x0 = float(np.nanmin(x_seg))
        x1 = float(np.nanmax(x_seg))
        xs = np.linspace(x0, x1, 30)
        ys = a * xs + b
        ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, zorder=6)

    _short_line(aux.get("x_bg"), a1, b1, alpha=0.65, lw=1.0)
    _short_line(aux.get("x_edge"), a2, b2, alpha=0.9, lw=1.2)

    ax.axvline(x_int, color=color, linestyle=":", linewidth=1.1, alpha=0.9, zorder=7)
    ax.text(
        x_int,
        0.98,
        f"{title} = {x_int:.2f} eV",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=9,
        color=color,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9),
        zorder=8,
    )


def _draw_seco_on_main_ax(ax, BE_cutoff, work_function, aux, label=None, color="C1", line_offset=0):
    """
    在 UPS 主图（全谱，20 eV 起始）上画出：短基线、短切线、功函数标注（不画交点竖线）。
    """
    if aux is None:
        return
    lo, hi = aux["lo"], aux["hi"]
    x0, y0 = aux["x0"], aux["y0"]
    m = aux["slope"]
    y_baseline = aux["y_baseline"]
    # 基线和切线都只画很短一段（交点两侧约 ±0.2 eV）
    dx_short = 0.2
    x_lo = max(lo, BE_cutoff - dx_short)
    x_hi = min(hi, BE_cutoff + dx_short)
    # 基线：短水平线
    ax.hlines(y_baseline, x_lo, x_hi, colors=color, linestyles="-", linewidth=1.0, alpha=0.85, zorder=4)
    # 切线：短线段，且到 y=0 就截止（不往下画）
    x_tan = np.linspace(x_lo, x_hi, 20)
    y_tan = y0 + m * (x_tan - x0)
    y_tan = np.maximum(y_tan, 0.0)
    ax.plot(x_tan, y_tan, color=color, linestyle="-", linewidth=1.0, alpha=0.85, zorder=4)
    # 功函数文字（放在交点上方）
    ylim = ax.get_ylim()
    y_pos = ylim[1] - (0.06 + 0.10 * line_offset) * (ylim[1] - ylim[0])
    text = f"φ = {work_function:.2f} eV"
    if label:
        text = f"{label}: {text}"
    ax.text(BE_cutoff, y_pos, text, fontsize=9, color=color, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9), zorder=6)


def add_zoom_inset(ax, x_left, x_right, loc="upper right", width="38%", height="38%"):
    """
    Add an inset zoom view to an existing axes.

    English:
    - x_left/x_right define the zoom window in Binding Energy.
    - UPS usually displayed high->low, so we set inset xlim as (max, min).
    - y-limits are auto-scaled using data inside zoom window.

    中文：
    - x_left/x_right 定义放大窗口的 BE 范围
    - UPS 显示大到小，因此 inset 的 xlim 设置为 (大, 小)
    - inset 的 y 轴范围会根据窗口内数据自动缩放
    """
    lo = min(x_left, x_right)
    hi = max(x_left, x_right)

    axins = inset_axes(ax, width=width, height=height, loc=loc, borderpad=1.2)

    for line in ax.lines:
        axins.plot(line.get_xdata(), line.get_ydata(),
                   lw=max(1.0, float(line.get_linewidth())))

    axins.set_xlim(hi, lo)

    y_vals = []
    for line in ax.lines:
        xdata = np.asarray(line.get_xdata())
        ydata = np.asarray(line.get_ydata())
        mask = (xdata >= lo) & (xdata <= hi)
        if np.any(mask):
            y_vals.append(ydata[mask])

    if y_vals:
        y_all = np.concatenate(y_vals)
        y0 = float(np.nanmin(y_all))
        y1 = float(np.nanmax(y_all))
        pad = 0.05 * (y1 - y0) if (y1 - y0) != 0 else 0.05
        axins.set_ylim(y0 - pad, y1 + pad)

    axins.grid(alpha=0.2)
    axins.tick_params(labelsize=8)

    try:
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.4")
    except Exception:
        pass

    return axins


def plot_overlay(spectra, zoom_enable=True, zoomA=(18.0, 15.0), zoomB=(-1.0, 2.0)):
    """叠加多条谱于一张图；仅当 zoom_enable 为 True 时右侧显示 18–15 eV、-1–2 eV 两块 zoom。"""
    all_x_min = min(np.min(s["x"]) for s in spectra)
    all_x_max = max(np.max(s["x"]) for s in spectra)

    if zoom_enable:
        fig = plt.figure(figsize=(12, 5.5), dpi=140)
        gs = GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1])
        ax_main = fig.add_subplot(gs[:, 0])
        ax_zoom1 = fig.add_subplot(gs[0, 1])
        ax_zoom2 = fig.add_subplot(gs[1, 1])
    else:
        fig = plt.figure(figsize=(8, 5), dpi=140)
        ax_main = fig.add_subplot(111)

    for s in spectra:
        ax_main.plot(s["x"], s["y_norm"], lw=1.3, label=s["base"])
    ax_main.set_xlim(all_x_max, all_x_min)
    ax_main.set_xlabel("Binding Energy (eV)")
    ax_main.set_ylabel("Normalized Intensity (0–1)")
    ax_main.grid(alpha=0.25)
    ax_main.legend(frameon=False, fontsize=8)

    if zoom_enable:
        loA, hiA = min(zoomA), max(zoomA)
        # SECO 画在主图（20 eV 起始的全谱）上：切线、基线、交点竖线、φ
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(spectra), 1)))
        for i, s in enumerate(spectra):
            BE_cut, phi, aux = find_seco(s["x"], s["y_norm"], search_region=(loA, hiA), hv=HV_HEI)
            if BE_cut is not None:
                _draw_seco_on_main_ax(ax_main, BE_cut, phi, aux, label=s["base"], color=colors[i % 10], line_offset=i)

        for s in spectra:
            ax_zoom1.plot(s["x"], s["y_norm"], lw=1.2)
        ax_zoom1.set_xlim(hiA, loA)
        ax_zoom1.set_title("Zoom 18–15 eV", fontsize=10)
        ax_zoom1.grid(alpha=0.2)
        y_vals = []
        for s in spectra:
            mask = (s["x"] >= loA) & (s["x"] <= hiA)
            if np.any(mask):
                y_vals.append(s["y_norm"][mask])
        if y_vals:
            y = np.concatenate(y_vals)
            ax_zoom1.set_ylim(np.min(y), np.max(y) * 1.05)

        loB, hiB = min(zoomB), max(zoomB)
        for s in spectra:
            ax_zoom2.plot(s["x"], s["y_norm"], lw=1.2)
        ax_zoom2.set_xlim(hiB, loB)
        ax_zoom2.set_title(f"Zoom {loB:g}–{hiB:g} eV", fontsize=10)
        ax_zoom2.grid(alpha=0.2)
        y_vals = []
        for s in spectra:
            mask = (s["x"] >= loB) & (s["x"] <= hiB)
            if np.any(mask):
                y_vals.append(s["y_norm"][mask])
        if y_vals:
            y = np.concatenate(y_vals)
            ax_zoom2.set_ylim(np.min(y), np.max(y) * 1.05)

        # Fermi edge detection on Zoom B (use user-provided range, e.g. -1~3)
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(spectra), 1)))
        for i, s in enumerate(spectra):
            BE_edge, aux = find_fermi_edge(s["x"], s["y_norm"], search_region=(loB, hiB), min_slope=0.005)
            if BE_edge is not None and aux and aux.get("reason") == "ok":
                _draw_edge_on_ax(ax_zoom2, BE_edge, aux, title="EF edge", color=colors[i % 10])

    fig.tight_layout()
    return fig


def _slice_and_renorm(x, y, lo, hi):
    """裁剪并局部归一化到 0–1，按 UPS 惯例 x 从高到低排序。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = (x >= lo) & (x <= hi)
    if not np.any(m):
        return None, None
    xs = x[m]
    ys = y[m]
    y0 = float(np.nanmin(ys))
    y1 = float(np.nanmax(ys))
    if (y1 - y0) > 0:
        yn = (ys - y0) / (y1 - y0)
    else:
        yn = ys * 0.0
    idx = np.argsort(xs)[::-1]
    return xs[idx], yn[idx]


def plot_preview_two_regions_overlay(
    spectra,
    region_seco=(18.0, 15.0),
    region_ef=(2.0, -1.0),
    annotate_edges=True,
):
    """
    Preview-only figure: show ONLY two regions side-by-side (1×2).
    - Left: SECO region (default 18–15 eV)
    - Right: EF region (default 2–-1 eV)
    No full-spectrum panel.
    """
    loA, hiA = min(region_seco), max(region_seco)
    loB, hiB = min(region_ef), max(region_ef)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.6, 4.2), dpi=150)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(spectra), 1)))
    for i, s in enumerate(spectra):
        c = colors[i % 10]
        xA, yA = _slice_and_renorm(s["x"], s["y_norm"], loA, hiA)
        xB, yB = _slice_and_renorm(s["x"], s["y_norm"], loB, hiB)
        if xA is not None:
            ax_a.plot(xA, yA, lw=1.5, label=s["base"], color=c)
        if xB is not None:
            ax_b.plot(xB, yB, lw=1.5, label=s["base"], color=c)

        if annotate_edges:
            # SECO: keep tangent-based cutoff on locally renormalized trace
            if xA is not None:
                # find_seco expects x ascending; feed sorted ascending slice
                xa = np.asarray(xA)[::-1]
                ya = np.asarray(yA)[::-1]
                BE_cut, _phi, aux = find_seco(xa, ya, search_region=(loA, hiA), hv=HV_HEI, min_slope=0.05)
                if BE_cut is not None and aux and aux.get("reason") == "ok":
                    _draw_edge_on_ax(ax_a, BE_cut, aux, title="SECO", color=c)

            # EF: auto background + rising region, two-line fit intersection
            if xB is not None:
                xb = np.asarray(xB)[::-1]
                yb = np.asarray(yB)[::-1]
                x_int, aux2 = find_edge_two_line_fit(xb, yb, search_region=(loB, hiB))
                if x_int is not None and aux2 and aux2.get("reason") == "ok":
                    _draw_two_line_fit_on_ax(ax_b, x_int, aux2, title="EF", color=c)

    ax_a.set_xlim(hiA, loA)
    ax_a.set_title(f"SECO  {hiA:g}–{loA:g} eV", fontsize=11)
    ax_a.set_xlabel("Binding Energy (eV)")
    ax_a.set_ylabel("Local-normalized Intensity (0–1)")
    ax_a.grid(alpha=0.25)

    ax_b.set_xlim(hiB, loB)
    ax_b.set_title(f"EF  {hiB:g}–{loB:g} eV", fontsize=11)
    ax_b.set_xlabel("Binding Energy (eV)")
    ax_b.grid(alpha=0.25)

    ax_a.set_ylim(-0.03, 1.03)
    ax_b.set_ylim(-0.03, 1.03)

    ax_a.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def plot_preview_two_regions_separate(
    spectra,
    region_seco=(18.0, 15.0),
    region_ef=(2.0, -1.0),
    annotate_edges=True,
):
    """
    Per-file preview: each spectrum gets a 1×2 figure with only two regions.
    Returns list of (base, fig).
    """
    loA, hiA = min(region_seco), max(region_seco)
    loB, hiB = min(region_ef), max(region_ef)
    figs = []

    for s in spectra:
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.2, 4.2), dpi=150)
        xA, yA = _slice_and_renorm(s["x"], s["y_norm"], loA, hiA)
        xB, yB = _slice_and_renorm(s["x"], s["y_norm"], loB, hiB)
        if xA is not None:
            ax_a.plot(xA, yA, lw=1.8, color="C0")
        if xB is not None:
            ax_b.plot(xB, yB, lw=1.8, color="C0")

        ax_a.set_xlim(hiA, loA)
        ax_a.set_title(f"{s['base']}  ·  SECO {hiA:g}–{loA:g} eV", fontsize=11)
        ax_a.set_xlabel("Binding Energy (eV)")
        ax_a.set_ylabel("Local-normalized Intensity (0–1)")
        ax_a.grid(alpha=0.25)

        ax_b.set_xlim(hiB, loB)
        ax_b.set_title(f"EF {hiB:g}–{loB:g} eV", fontsize=11)
        ax_b.set_xlabel("Binding Energy (eV)")
        ax_b.grid(alpha=0.25)

        ax_a.set_ylim(-0.03, 1.03)
        ax_b.set_ylim(-0.03, 1.03)

        if annotate_edges:
            if xA is not None:
                xa = np.asarray(xA)[::-1]
                ya = np.asarray(yA)[::-1]
                BE_cut, _phi, aux = find_seco(xa, ya, search_region=(loA, hiA), hv=HV_HEI, min_slope=0.05)
                if BE_cut is not None and aux and aux.get("reason") == "ok":
                    _draw_edge_on_ax(ax_a, BE_cut, aux, title="SECO", color="C1")

            if xB is not None:
                xb = np.asarray(xB)[::-1]
                yb = np.asarray(yB)[::-1]
                x_int, aux2 = find_edge_two_line_fit(xb, yb, search_region=(loB, hiB))
                if x_int is not None and aux2 and aux2.get("reason") == "ok":
                    _draw_two_line_fit_on_ax(ax_b, x_int, aux2, title="EF", color="C2")

        fig.tight_layout()
        figs.append((s["base"], fig))

    return figs


def plot_separate(spectra, zoom_enable=True, zoomA=(18.0, 15.0), zoomB=(-1.0, 2.0)):
    """
    每条谱一张图。zoom_enable 为 True 时每张图 3 面板：左全谱，右上 18–15 eV，右下 -1–2 eV；否则仅主图。
    """
    figs = []
    loA, hiA = min(zoomA), max(zoomA)
    loB, hiB = min(zoomB), max(zoomB)

    for s in spectra:
        if zoom_enable:
            fig = plt.figure(figsize=(10.2, 5.6), dpi=140)
            gs = GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1])
            ax_main = fig.add_subplot(gs[:, 0])
            ax_z1 = fig.add_subplot(gs[0, 1])
            ax_z2 = fig.add_subplot(gs[1, 1])
        else:
            fig = plt.figure(figsize=(7, 5), dpi=140)
            ax_main = fig.add_subplot(111)

        ax_main.plot(s["x"], s["y_norm"], lw=1.4)
        ax_main.set_xlim(np.max(s["x"]), np.min(s["x"]))
        ax_main.set_xlabel("Binding Energy (eV)")
        ax_main.set_ylabel("Normalized Intensity (0–1)")
        ax_main.set_title(s["base"])
        ax_main.grid(alpha=0.25)

        if zoom_enable:
            BE_cut, phi, aux = find_seco(s["x"], s["y_norm"], search_region=(loA, hiA), hv=HV_HEI)
            if BE_cut is not None:
                _draw_seco_on_main_ax(ax_main, BE_cut, phi, aux, color="C1")

            ax_z1.plot(s["x"], s["y_norm"], lw=1.2)
            ax_z1.set_xlim(hiA, loA)
            ax_z1.set_title("Zoom 18–15 eV", fontsize=10)
            ax_z1.grid(alpha=0.2)
            mA = (s["x"] >= loA) & (s["x"] <= hiA)
            if np.any(mA):
                yA = s["y_norm"][mA]
                y0, y1 = float(np.min(yA)), float(np.max(yA))
                pad = 0.08 * (y1 - y0) if (y1 - y0) != 0 else 0.08
                ax_z1.set_ylim(y0 - pad, y1 + pad)

            ax_z2.plot(s["x"], s["y_norm"], lw=1.2)
            ax_z2.set_xlim(hiB, loB)
            ax_z2.set_title(f"Zoom {loB:g}–{hiB:g} eV", fontsize=10)
            ax_z2.grid(alpha=0.2)
            mB = (s["x"] >= loB) & (s["x"] <= hiB)
            if np.any(mB):
                yB = s["y_norm"][mB]
                y0, y1 = float(np.min(yB)), float(np.max(yB))
                pad = 0.08 * (y1 - y0) if (y1 - y0) != 0 else 0.08
                ax_z2.set_ylim(y0 - pad, y1 + pad)

            BE_edge, aux = find_fermi_edge(s["x"], s["y_norm"], search_region=(loB, hiB), min_slope=0.005)
            if BE_edge is not None and aux and aux.get("reason") == "ok":
                _draw_edge_on_ax(ax_z2, BE_edge, aux, title="EF edge", color="C2")

        fig.tight_layout()
        figs.append((s["base"], fig))

    return figs


def save_figure(fig, path):
    """
    保存图并关闭 figure。
    - PNG：300 dpi
    - SVG/PDF：矢量导出
    """
    ext = os.path.splitext(str(path))[-1].lower()
    if ext == ".png":
        fig.savefig(path, dpi=300, bbox_inches="tight")
    else:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_png(fig, path):
    """兼容旧接口：保存 PNG 并关闭 figure。"""
    return save_figure(fig, path)


def plot_homo_stitched(s, zoomA=(18.0, 15.0), homo_range=(0.0, 5.0), hv=HV_HEI):
    """
    生成用于读 HOMO 的“断轴拼接图”：
    - 左侧：费米边附近（Zoom B）
    - 右侧：二次电子截止边附近（Zoom A）

    备注：通常电离能 IE = hv - (SECO - HOMO_onset) = φ + HOMO_onset。
    这里会标出 SECO/φ，并在图上提示 IE 公式，HOMO_onset 由用户从左图读取。
    """
    loA, hiA = min(zoomA), max(zoomA)
    loH, hiH = min(homo_range), max(homo_range)

    # 连续拼接到同一横轴：
    # - 左侧 VB：把 BE (0→5) 映射到 x' = -BE（0→-5），以 EF=0 为分界
    # - 右侧 SECO：把 BE (20→15) 映射到 x' = hiA - BE（20→0, 15→5）
    fig, ax = plt.subplots(1, 1, figsize=(10.8, 4.0), dpi=160)

    x = np.asarray(s["x"])
    y = np.asarray(s["y_norm"])
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # 左侧：VB（homo_range，默认 0~5 eV），在该范围内重新归一化到 0–1，然后映射到负轴
    mH = (x >= loH) & (x <= hiH)
    if np.any(mH):
        xH = x[mH]
        yH = y[mH]
        y0 = float(np.nanmin(yH))
        y1 = float(np.nanmax(yH))
        yHn = (yH - y0) / (y1 - y0) if (y1 - y0) > 0 else (yH * 0.0)
        xVB = -xH  # EF=0 在 0，VB 在负侧
        ax.fill_between(xVB, yHn, 0, color="C0", alpha=0.25, linewidth=0)
        ax.plot(xVB, yHn, lw=1.7, color="C0")

    # 右侧：SECO（zoomA，默认 20~15 eV），映射到 0~5，并单独归一化
    mA = (x >= loA) & (x <= hiA)
    if np.any(mA):
        xA = x[mA]
        yA = y[mA]
        y0 = float(np.nanmin(yA))
        y1 = float(np.nanmax(yA))
        yAn = (yA - y0) / (y1 - y0) if (y1 - y0) > 0 else (yA * 0.0)
        xSE = hiA - xA  # hiA(≈20)->0, loA(≈15)->5
        ax.fill_between(xSE, yAn, 0, color="C0", alpha=0.25, linewidth=0)
        ax.plot(xSE, yAn, lw=1.7, color="C0")

    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(-hiH, hiA - loA)
    ax.grid(alpha=0.25)
    ax.axvline(0.0, color="#0f172a", lw=1.2, ls="--", alpha=0.85)
    ax.text(0.0, 0.98, "EF = 0", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9, color="#0f172a")

    # 标注 SECO / φ（使用 ZoomA 作为搜索区间），并把 SECO 位置映射到 x' 上
    BE_cut, phi, aux = find_seco(x, y, search_region=(loA, hiA), hv=hv)
    if BE_cut is not None and aux and aux.get("reason") == "ok":
        x_seco = hiA - BE_cut
        ax.axvline(x_seco, color="C1", lw=1.0, ls=":", alpha=0.9)
        ax.text(x_seco, 0.02, f"SECO = {BE_cut:.2f} eV", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9, color="C1",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))
        ax.text(0.99, 0.97, f"φ = {phi:.2f} eV", transform=ax.transAxes,
                ha="right", va="top", fontsize=10, color="C1",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9))

    # 统一标签/标题
    fig.suptitle(f"{s.get('base', 'Spectrum')}  ·  VB (left) + SECO (right)", fontsize=12, y=0.98)
    ax.set_xlabel("Energy axis (eV): left = −BE (VB), right = (hiA − BE) (SECO)")
    ax.set_ylabel("Intensity (0–1, renormalized separately)")
    fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.95])
    return fig
