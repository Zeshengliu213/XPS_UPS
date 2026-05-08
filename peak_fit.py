# peak_fit.py
# ============================================================
# XPS 峰拟合后端
#   - 背景扣除：Linear / Shirley
#   - 峰型：伪 Voigt（Gaussian + Lorentzian 线性混合）
#   - 多峰拟合：scipy.optimize.curve_fit（含参数边界）
# ============================================================

import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid

# NumPy 2.0 renamed trapz → trapezoid
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")


# ------------------------------------------------------------------ #
#  Background subtraction                                              #
# ------------------------------------------------------------------ #

def linear_background(x, y):
    """
    两端点连线作为线性背景（x 可任意顺序）。
    返回与 x 等长的背景数组。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]
    # 在全程上线性插值
    bg = np.interp(x, [xs[0], xs[-1]], [ys[0], ys[-1]])
    return bg


def shirley_background(x, y, max_iter=50, tol=1e-6):
    """
    Shirley 迭代背景：阶梯形背景，用于 XPS 峰下方的非弹性散射贡献。

    算法：
        B[i] = y_lo + (y_hi - y_lo) * ∫[x[0]→x[i]] (y−B) dx
                                      ────────────────────────
                                      ∫[x[0]→x[-1]] (y−B) dx

    其中 x[0] = low BE 端，x[-1] = high BE 端。
    内部升序处理，完成后还原到原始顺序。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    idx_sort = np.argsort(x)
    xs = x[idx_sort]
    ys = y[idx_sort]
    y_lo = ys[0]
    y_hi = ys[-1]

    # 初始猜测：线性
    bg = np.linspace(y_lo, y_hi, len(ys))

    for _ in range(max_iter):
        net = np.maximum(ys - bg, 0.0)
        total = _trapezoid(net, xs)
        if total < 1e-30:
            break
        cum = cumulative_trapezoid(net, xs, initial=0.0)
        bg_new = y_lo + (y_hi - y_lo) * cum / total
        if np.max(np.abs(bg_new - bg)) < tol:
            bg = bg_new
            break
        bg = bg_new

    result = np.empty_like(x)
    result[idx_sort] = bg
    return result


# ------------------------------------------------------------------ #
#  Peak model                                                          #
# ------------------------------------------------------------------ #

def pseudo_voigt(x, center, amplitude, fwhm, eta):
    """
    伪 Voigt 峰型：Gaussian 与 Lorentzian 的线性混合。

    参数
    ----
    center    : 峰中心 (eV)
    amplitude : 峰高（背景扣除后）
    fwhm      : 半高全宽 (eV)
    eta       : Lorentzian 比例（0 = 纯 Gaussian，1 = 纯 Lorentzian）
    """
    fwhm = max(float(fwhm), 1e-6)
    eta = float(np.clip(eta, 0.0, 1.0))
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gamma = fwhm / 2.0
    G = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    L = 1.0 / (1.0 + ((x - center) / gamma) ** 2)
    return amplitude * (eta * L + (1.0 - eta) * G)


def _multi_peak_func(x, *params):
    """内部：n 个伪 Voigt 之和，参数依次为 [cen, amp, fwhm, eta, ...]。"""
    n = len(params) // 4
    y = np.zeros_like(x, dtype=float)
    for i in range(n):
        cen, amp, fwhm, eta = params[i * 4: (i + 1) * 4]
        y += pseudo_voigt(x, cen, amp, fwhm, eta)
    return y


# ------------------------------------------------------------------ #
#  Main fitting function                                               #
# ------------------------------------------------------------------ #

_SPIN_ORBIT_PRESETS = {
    # Values are typical; users can still tweak by switching to unconstrained.
    "S2p":  {"delta_e": 1.18, "area_ratio_main_to_minor": 2.0 / 1.0},
    "Au4f": {"delta_e": 3.67, "area_ratio_main_to_minor": 4.0 / 3.0},
    "Ti2p": {"delta_e": 5.54, "area_ratio_main_to_minor": 2.0 / 1.0},
    "Cu2p": {"delta_e": 19.75, "area_ratio_main_to_minor": 2.0 / 1.0},
    "Zn2p": {"delta_e": 23.05, "area_ratio_main_to_minor": 2.0 / 1.0},
    "Fe2p": {"delta_e": 13.6, "area_ratio_main_to_minor": 2.0 / 1.0},
    "Ni2p": {"delta_e": 17.3, "area_ratio_main_to_minor": 2.0 / 1.0},
    "Co2p": {"delta_e": 15.0, "area_ratio_main_to_minor": 2.0 / 1.0},
    "Mn2p": {"delta_e": 11.5, "area_ratio_main_to_minor": 2.0 / 1.0},
    "Mo3d": {"delta_e": 3.13, "area_ratio_main_to_minor": 3.0 / 2.0},
    "W4f":  {"delta_e": 2.18, "area_ratio_main_to_minor": 4.0 / 3.0},
    "Pt4f": {"delta_e": 3.33, "area_ratio_main_to_minor": 4.0 / 3.0},
}


def _spin_orbit_pair_func(mode: str, x, center_main, amp_main, fwhm, eta):
    """
    返回受约束的双峰和：主峰(center_main, amp_main, fwhm, eta) + 次峰(中心偏移, 按面积比缩放 amp)。
    这里用 amplitude 比例近似面积比例（共享 fwhm/eta 时面积与峰高成正比）。
    """
    preset = _SPIN_ORBIT_PRESETS.get(mode)
    if preset is None:
        raise ValueError(f"Unknown spin orbit mode: {mode}")
    de = float(preset["delta_e"])
    ratio = float(preset["area_ratio_main_to_minor"])
    center_minor = center_main + de
    amp_minor = amp_main / ratio
    return (
        pseudo_voigt(x, center_main, amp_main, fwhm, eta)
        + pseudo_voigt(x, center_minor, amp_minor, fwhm, eta)
    )


def fit_xps(x, y, peak_guesses, bg_type="shirley", fit_range=None, spin_orbit_mode=None, fix_centers=False):
    """
    XPS 多峰拟合主函数。

    参数
    ----
    x, y         : 原始谱（可任意顺序，使用 raw y，非归一化）
    peak_guesses : list[dict]，每项 {"center": float, "fwhm": float, "eta": float}
                   amplitude 自动从背景扣除后的谱中估算
    bg_type      : "none" | "linear" | "shirley"
    fit_range    : (lo, hi) eV 截取范围；None 则用全谱
    spin_orbit_mode : None | "S2p" | "Au4f"
    fix_centers  : bool — True 时严格锁定峰中心（center 只允许 ±0.001 eV 浮动）

    返回 dict
    ----------
    success       : bool
    message       : str（失败原因 或 "ok"）
    x_fit         : 升序 x 数组（裁剪后）
    y_raw         : 对应原始 y
    y_bg          : 背景
    y_corrected   : 背景扣除后
    y_total_fit   : 多峰模型拟合值（背景扣除空间）
    peaks_y       : list[ndarray]，每峰单独的 y（背景扣除空间）
    params        : list[dict]，拟合得到的峰参数 {center, amplitude, fwhm, eta}
    errors        : list[dict]，各参数 1σ 不确定度
    r_squared     : float
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # 裁剪到拟合范围
    if fit_range is not None:
        lo, hi = min(fit_range), max(fit_range)
        mask = (x >= lo) & (x <= hi)
        if np.sum(mask) < 10:
            return {
                "success": False,
                "message": f"拟合范围 {lo:.2f}–{hi:.2f} eV 内仅有 {int(np.sum(mask))} 个点（至少需要 10 个）",
            }
        x = x[mask]
        y = y[mask]

    # 升序排列（拟合与背景算法均要求）
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    # 背景扣除
    if bg_type == "shirley":
        y_bg = shirley_background(x, y)
    elif bg_type == "linear":
        y_bg = linear_background(x, y)
    else:
        y_bg = np.zeros_like(y)

    y_corr = y - y_bg

    if spin_orbit_mode in (None, "None", "", False):
        spin_orbit_mode = None
    if spin_orbit_mode is not None and spin_orbit_mode not in _SPIN_ORBIT_PRESETS:
        return {"success": False, "message": f"未知自旋轨道约束模式：{spin_orbit_mode}"}

    if not peak_guesses and spin_orbit_mode is None:
        return {"success": False, "message": "请至少添加一个峰"}

    y_max = float(np.max(y_corr)) if np.max(y_corr) > 0 else 1.0
    if spin_orbit_mode is None:
        # 初始参数与边界（常规多峰）
        p0, lo_b, hi_b = [], [], []
        for pg in peak_guesses:
            cen = float(pg["center"])
            fwhm = float(pg.get("fwhm", 1.0))
            eta = float(pg.get("eta", 0.3))
            # 在 cen 处插值估算初始幅度，至少为谱高的 1%
            amp = float(np.interp(cen, x, y_corr))
            amp = max(amp, 0.01 * y_max)
            p0 += [cen, amp, fwhm, eta]
            cen_tol = 0.001 if fix_centers else 3.0
            lo_b += [cen - cen_tol, 0.0, 0.05, 0.0]
            hi_b += [cen + cen_tol, y_max * 3.0, 8.0, 1.0]
        model_func = _multi_peak_func
    else:
        # 受约束双峰：用一组参数（主峰）拟合
        if peak_guesses:
            cen0 = float(peak_guesses[0].get("center", x[int(np.argmax(y_corr))]))
            fwhm0 = float(peak_guesses[0].get("fwhm", 1.0))
            eta0 = float(peak_guesses[0].get("eta", 0.3))
        else:
            cen0 = float(x[int(np.argmax(y_corr))])
            fwhm0 = 1.0
            eta0 = 0.3

        amp0 = float(np.interp(cen0, x, y_corr))
        amp0 = max(amp0, 0.05 * y_max)
        p0 = [cen0, amp0, fwhm0, eta0]
        cen_tol0 = 0.001 if fix_centers else 3.0
        lo_b = [cen0 - cen_tol0, 0.0, 0.05, 0.0]
        hi_b = [cen0 + cen_tol0, y_max * 3.0, 8.0, 1.0]
        model_func = lambda xx, cen, amp, fwhm, eta: _spin_orbit_pair_func(spin_orbit_mode, xx, cen, amp, fwhm, eta)

    try:
        popt, pcov = curve_fit(
            model_func, x, y_corr,
            p0=p0, bounds=(lo_b, hi_b),
            maxfev=20000,
        )
    except Exception as e:
        return {"success": False, "message": f"scipy curve_fit 失败：{e}"}

    # 参数不确定度
    try:
        perr = np.sqrt(np.diag(pcov))
    except Exception:
        perr = np.zeros_like(popt)

    fitted_params, fitted_errors, peaks_y = [], [], []
    if spin_orbit_mode is None:
        n = len(peak_guesses)
        for i in range(n):
            cen, amp, fwhm, eta = popt[i * 4: (i + 1) * 4]
            err = perr[i * 4: (i + 1) * 4]
            fitted_params.append(
                {"center": float(cen), "amplitude": float(amp), "fwhm": float(fwhm), "eta": float(eta)}
            )
            fitted_errors.append(
                {"center": float(err[0]), "amplitude": float(err[1]), "fwhm": float(err[2]), "eta": float(err[3])}
            )
            peaks_y.append(pseudo_voigt(x, cen, amp, fwhm, eta))
        y_total = _multi_peak_func(x, *popt)
    else:
        cen, amp, fwhm, eta = popt
        err = perr if len(perr) == 4 else np.zeros(4)
        preset = _SPIN_ORBIT_PRESETS[spin_orbit_mode]
        de = float(preset["delta_e"])
        ratio = float(preset["area_ratio_main_to_minor"])

        cen_minor = float(cen + de)
        amp_minor = float(amp / ratio)

        fitted_params = [
            {"center": float(cen), "amplitude": float(amp), "fwhm": float(fwhm), "eta": float(eta)},
            {"center": cen_minor, "amplitude": amp_minor, "fwhm": float(fwhm), "eta": float(eta)},
        ]
        fitted_errors = [
            {"center": float(err[0]), "amplitude": float(err[1]), "fwhm": float(err[2]), "eta": float(err[3])},
            {
                "center": float(err[0]),
                "amplitude": float(err[1] / ratio),
                "fwhm": float(err[2]),
                "eta": float(err[3]),
            },
        ]
        peaks_y = [
            pseudo_voigt(x, float(cen), float(amp), float(fwhm), float(eta)),
            pseudo_voigt(x, cen_minor, amp_minor, float(fwhm), float(eta)),
        ]
        y_total = model_func(x, *popt)
    ss_res = float(np.sum((y_corr - y_total) ** 2))
    ss_tot = float(np.sum((y_corr - np.mean(y_corr)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "success":     True,
        "message":     "ok",
        "x_fit":       x,
        "y_raw":       y,
        "y_bg":        y_bg,
        "y_corrected": y_corr,
        "y_total_fit": y_total,
        "peaks_y":     peaks_y,
        "params":      fitted_params,
        "errors":      fitted_errors,
        "r_squared":   r2,
        "spin_orbit_mode": spin_orbit_mode,
    }
