# autofit_engine.py
# ============================================================
# 自动拟合引擎：DL (CNN1Dv2) 初猜 + lmfit 精修
#
# 用法（在 fit_window 中）：
#   from autofit_engine import auto_fit, AutoFitResult
#   res = auto_fit(be=x_array, y=y_array, fit_range=(lo, hi),
#                  models_dir=Path(__file__).parent / "models")
#   res.peak_guesses  -> List[dict{"center","fwhm","eta"}] 可直接喂给 FitWindow
# ============================================================

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# torch / lmfit 是可选依赖；缺失时自动降级
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False
    nn = None  # type: ignore

try:
    from lmfit import Parameters, minimize
    _HAS_LMFIT = True
except Exception:
    _HAS_LMFIT = False


# ------------------------------------------------------------------ #
#  常量 / 默认 BE 窗口（与训练侧保持一致）                              #
# ------------------------------------------------------------------ #

DEFAULT_BE_WINDOWS = {
    "C1s": (278.0, 296.0),
    "N1s": (393.0, 408.0),
    "S2p": (157.0, 175.0),
    "O1s": (525.0, 542.0),
    "F1s": (678.0, 695.0),
}

ORBITAL_BY_BE = [
    ("C1s", 276.0, 300.0),
    ("N1s", 393.0, 408.0),
    ("S2p", 157.0, 175.0),
    ("O1s", 525.0, 542.0),
    ("F1s", 678.0, 695.0),
]

DOUBLET_BY_ORBITAL = {
    "S2p":  (1.18, 0.50),
    "Si2p": (0.60, 0.50),
    "P2p":  (0.84, 0.50),
    "Au4f": (3.65, 0.75),
}


# ------------------------------------------------------------------ #
#  CNN 架构定义（与 train_c1s_v2.py 保持一致）                          #
# ------------------------------------------------------------------ #

if _HAS_TORCH:

    class _ConvBlock(nn.Module):
        def __init__(self, c_in: int, c_out: int, k: int = 7, stride: int = 1):
            super().__init__()
            self.conv = nn.Conv1d(c_in, c_out, k, stride=stride, padding=k // 2)
            self.bn = nn.BatchNorm1d(c_out)

        def forward(self, x):  # type: ignore[override]
            return F.relu(self.bn(self.conv(x)))

    class CNN1Dv2(nn.Module):
        """V2 架构：~1.4M 参数。训练侧 train_c1s_v2.py。"""

        def __init__(self, max_peaks: int = 4, param_dim: int = 7):
            super().__init__()
            self.stem = _ConvBlock(1, 48, k=11)
            self.b1 = nn.Sequential(_ConvBlock(48, 96, k=9, stride=2),  _ConvBlock(96, 96, k=9))
            self.b2 = nn.Sequential(_ConvBlock(96, 192, k=7, stride=2), _ConvBlock(192, 192, k=7))
            self.b3 = nn.Sequential(_ConvBlock(192, 256, k=5, stride=2), _ConvBlock(256, 256, k=5))
            self.b4 = nn.Sequential(_ConvBlock(256, 320, k=3, stride=2), _ConvBlock(320, 320, k=3))
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Sequential(
                nn.Flatten(), nn.Linear(320, 384), nn.ReLU(), nn.Dropout(0.15),
                nn.Linear(384, 384), nn.ReLU(), nn.Dropout(0.10),
                nn.Linear(384, max_peaks * param_dim),
            )
            self.max_peaks = max_peaks
            self.param_dim = param_dim

        def forward(self, x):  # type: ignore[override]
            x = self.stem(x); x = self.b1(x); x = self.b2(x)
            x = self.b3(x); x = self.b4(x); x = self.pool(x)
            out = self.head(x)
            return out.view(-1, self.max_peaks, self.param_dim)

    class CNN1D(nn.Module):
        """V1 架构（兼容旧 ckpt）。"""

        def __init__(self, max_peaks: int = 5, param_dim: int = 7):
            super().__init__()
            self.stem = _ConvBlock(1, 32, k=9)
            self.b1 = nn.Sequential(_ConvBlock(32, 64, k=7, stride=2), _ConvBlock(64, 64, k=7))
            self.b2 = nn.Sequential(_ConvBlock(64, 128, k=5, stride=2), _ConvBlock(128, 128, k=5))
            self.b3 = nn.Sequential(_ConvBlock(128, 192, k=3, stride=2), _ConvBlock(192, 192, k=3))
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Sequential(
                nn.Flatten(), nn.Linear(192, 256), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(256, max_peaks * param_dim),
            )
            self.max_peaks = max_peaks
            self.param_dim = param_dim

        def forward(self, x):  # type: ignore[override]
            x = self.stem(x); x = self.b1(x); x = self.b2(x)
            x = self.b3(x); x = self.pool(x)
            out = self.head(x)
            return out.view(-1, self.max_peaks, self.param_dim)


# ------------------------------------------------------------------ #
#  数学工具                                                           #
# ------------------------------------------------------------------ #

def _pseudo_voigt(x: np.ndarray, mu: float, fwhm_g: float, fwhm_l: float, eta: float) -> np.ndarray:
    sigma = fwhm_g / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    gauss = np.exp(-((x - mu) ** 2) / (2.0 * sigma * sigma))
    gamma = fwhm_l / 2.0
    lorentz = (gamma * gamma) / ((x - mu) ** 2 + gamma * gamma)
    return eta * lorentz + (1.0 - eta) * gauss


def _shirley_iter(x: np.ndarray, y: np.ndarray, max_iter: int = 80, tol: float = 1e-5) -> np.ndarray:
    """与 autofit.py 的 shirley_iter 一致：内部按 BE 降序处理。"""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if x[0] < x[-1]:
        x = x[::-1]; y = y[::-1]
        rev = True
    else:
        rev = False
    yL, yR = float(y[0]), float(y[-1])
    bg = np.linspace(yL, yR, len(y))
    for _ in range(max_iter):
        diff = y - bg
        diff[diff < 0] = 0
        cum = np.cumsum(diff)
        total = float(cum[-1]) if cum[-1] > 0 else 1.0
        bg_new = yR + (yL - yR) * (1.0 - cum / total)
        if np.max(np.abs(bg_new - bg)) < tol * max(yL - yR, 1.0):
            bg = bg_new
            break
        bg = bg_new
    return bg[::-1] if rev else bg


def _resample(be: np.ndarray, y: np.ndarray, be_lo: float, be_hi: float, n_grid: int = 360) -> np.ndarray:
    be = np.asarray(be, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if be[0] > be[-1]:
        be = be[::-1]; y = y[::-1]
    grid = np.linspace(be_lo, be_hi, n_grid, dtype=np.float32)
    return np.interp(grid, be, y).astype(np.float32)


def _normalize(y: np.ndarray):
    base = float(np.percentile(y, 5))
    yc = y - base
    peak = max(float(yc.max()), 1e-6)
    return (yc / peak).astype(np.float32), base, peak


def detect_orbital(be_min: float, be_max: float) -> str:
    center = (be_min + be_max) / 2.0
    span = be_max - be_min
    if span > 200:
        return "Survey"
    for name, lo, hi in ORBITAL_BY_BE:
        if lo <= center <= hi:
            return name
    return "Unknown"


def pick_model_path(orbital: str, models_dir: Path) -> Optional[Path]:
    candidates = {
        "C1s": ["c1s_v2.pt", "c1s_v1.pt"],
        "N1s": ["n1s_v2.pt", "n1s_v1.pt"],
        "S2p": ["s2p_v2.pt", "s2p_v1.pt"],
        "O1s": ["o1s_v2.pt", "o1s_v1.pt"],
        "F1s": ["f1s_v2.pt", "f1s_v1.pt"],
    }
    for fname in candidates.get(orbital, []):
        p = models_dir / fname
        if p.exists():
            return p
    return None


# ------------------------------------------------------------------ #
#  模型加载 / 推理                                                     #
# ------------------------------------------------------------------ #

_MODEL_CACHE: dict = {}


def _load_model(ckpt_path: Path):
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch 未安装，无法做 DL 自动拟合。请先 `pip install torch`。")
    key = str(ckpt_path.resolve())
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    cls = cfg.get("model_class", "CNN1D")
    model = CNN1Dv2() if cls == "CNN1Dv2" else CNN1D()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    _MODEL_CACHE[key] = (model, cfg)
    return model, cfg


def _dl_infer(model, cfg: dict, be: np.ndarray, y_sub: np.ndarray):
    be_lo = cfg.get("be_lo", 278.0)
    be_hi = cfg.get("be_hi", 296.0)
    n_grid = cfg.get("n_grid", 360)
    if be.min() > be_hi - 5 or be.max() < be_lo + 5:
        return []
    y_rs = _resample(be, y_sub, be_lo, be_hi, n_grid)
    y_norm, _, peak_scale = _normalize(y_rs)
    x = torch.from_numpy(y_norm)[None, None]
    with torch.no_grad():
        pred = model(x).cpu().numpy()[0]

    peaks = []
    for i in range(pred.shape[0]):
        present = 1.0 / (1.0 + math.exp(-pred[i, 0]))
        if present < 0.5:
            continue
        mu = float(pred[i, 1] * (be_hi - be_lo) + be_lo)
        peaks.append({
            "mu": mu,
            "fwhm_g": float(math.exp(pred[i, 2])),
            "fwhm_l": float(math.exp(pred[i, 3])),
            "eta": float(np.clip(pred[i, 4], 0.0, 1.0)),
            "amp": float(math.exp(pred[i, 5]) * peak_scale),
            "presence": float(present),
        })
    return peaks


def _lmfit_refine(be: np.ndarray, y_sub: np.ndarray, peaks_dl: list, orbital: str):
    if not _HAS_LMFIT or not peaks_dl:
        return peaks_dl, float("nan"), float("nan")

    doublet = DOUBLET_BY_ORBITAL.get(orbital)
    peak_scale = float(np.max(y_sub)) if len(y_sub) else 1.0

    p = Parameters()
    for i, pk in enumerate(peaks_dl):
        p.add(f"mu_{i}",  value=pk["mu"],
              min=float(be.min()) + 0.3, max=float(be.max()) - 0.3)
        p.add(f"fg_{i}",  value=max(0.6, pk["fwhm_g"]), min=0.5, max=2.5)
        p.add(f"fl_{i}",  value=max(0.4, pk["fwhm_l"]), min=0.3, max=2.0)
        p.add(f"eta_{i}", value=pk["eta"], min=0.0, max=1.0)
        p.add(f"amp_{i}", value=max(pk["amp"], peak_scale * 0.05),
              min=peak_scale * 1e-3, max=peak_scale * 3.0)

    n = len(peaks_dl)

    def model(params, x):
        out = np.zeros_like(x, dtype=np.float64)
        for i in range(n):
            mu = params[f"mu_{i}"].value
            fg = params[f"fg_{i}"].value
            fl = params[f"fl_{i}"].value
            eta = params[f"eta_{i}"].value
            amp = params[f"amp_{i}"].value
            out += amp * _pseudo_voigt(x, mu, fg, fl, eta)
            if doublet is not None:
                split, ratio = doublet
                out += amp * ratio * _pseudo_voigt(x, mu + split, fg, fl, eta)
        return out

    try:
        out = minimize(lambda pp, xx, yy: model(pp, xx) - yy,
                       p, args=(be, y_sub), method="leastsq", max_nfev=8000)
    except Exception:
        return peaks_dl, float("nan"), float("nan")

    refined = []
    for i in range(n):
        amp_i = float(out.params[f"amp_{i}"].value)
        if amp_i / max(peak_scale, 1e-9) < 0.02:
            continue
        refined.append({
            "mu":     float(out.params[f"mu_{i}"].value),
            "fwhm_g": float(out.params[f"fg_{i}"].value),
            "fwhm_l": float(out.params[f"fl_{i}"].value),
            "eta":    float(out.params[f"eta_{i}"].value),
            "amp":    amp_i,
        })
    refined.sort(key=lambda pk: pk["mu"])

    fit_curve = model(out.params, be)
    res = y_sub - fit_curve
    rms = float(np.sqrt(np.mean(res ** 2)) / max(peak_scale, 1e-9))
    redchi = float(out.redchi) if out.redchi is not None else float("nan")
    return refined, redchi, rms


# ------------------------------------------------------------------ #
#  对外 API                                                           #
# ------------------------------------------------------------------ #

@dataclass
class AutoFitResult:
    success: bool
    message: str = ""
    orbital: str = ""
    model_name: str = ""
    n_peaks: int = 0
    redchi: float = float("nan")
    rms: float = float("nan")
    # 直接喂给 FitWindow._peak_guesses 的格式
    peak_guesses: list = field(default_factory=list)
    # 完整峰参数（含 amp / fwhm_g / fwhm_l），便于调试
    peaks_full: list = field(default_factory=list)


def auto_fit(
    be: np.ndarray,
    y: np.ndarray,
    fit_range: Optional[tuple] = None,
    models_dir: Optional[Path] = None,
    refine: bool = True,
) -> AutoFitResult:
    """
    自动拟合：返回 peak_guesses，可直接填入 FitWindow 的峰列表。

    Parameters
    ----------
    be, y       : 整段谱（可超出 fit_range；函数内部会裁剪）
    fit_range   : (lo, hi)，单位 eV；None 则用整段
    models_dir  : 模型 .pt 所在目录；默认 <脚本同级>/models
    refine      : 是否做 lmfit 精修（False 时只返回 DL 初猜）
    """
    if not _HAS_TORCH:
        return AutoFitResult(success=False,
                             message="未安装 PyTorch，无法做 DL 自动拟合。")

    if models_dir is None:
        models_dir = Path(__file__).resolve().parent / "models"
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return AutoFitResult(success=False, message=f"模型目录不存在：{models_dir}")

    be = np.asarray(be, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if be.size < 8 or y.size != be.size:
        return AutoFitResult(success=False, message="谱数据点太少。")

    # 裁剪到 fit_range 内
    if fit_range is not None:
        lo, hi = float(min(fit_range)), float(max(fit_range))
        m = (be >= lo) & (be <= hi)
        if m.sum() < 8:
            return AutoFitResult(success=False,
                                 message=f"区域内数据点过少（{int(m.sum())}）。")
        be = be[m]; y = y[m]

    # BE 降序（与训练侧约定一致）
    order = np.argsort(-be)
    be = be[order]; y = y[order]

    orbital = detect_orbital(float(be.min()), float(be.max()))
    if orbital in ("Unknown", "Survey"):
        return AutoFitResult(success=False, orbital=orbital,
                             message=f"区域 [{be.min():.1f},{be.max():.1f}] eV "
                                     f"对应 {orbital}，目前不支持自动拟合。")
    model_path = pick_model_path(orbital, models_dir)
    if model_path is None:
        return AutoFitResult(success=False, orbital=orbital,
                             message=f"未找到 {orbital} 模型（在 {models_dir} 下）。")

    try:
        model, cfg = _load_model(model_path)
    except Exception as e:
        return AutoFitResult(success=False, orbital=orbital,
                             message=f"模型加载失败：{e}")

    # Shirley 背景扣除（仅给 DL / lmfit 使用；FitWindow 自己还会再算一次）
    bg = _shirley_iter(be, y)
    y_sub = y - bg

    peaks_dl = _dl_infer(model, cfg, be, y_sub)
    if not peaks_dl:
        return AutoFitResult(success=False, orbital=orbital,
                             model_name=model_path.name,
                             message="DL 未检测到峰。")

    if refine and _HAS_LMFIT:
        peaks_final, redchi, rms = _lmfit_refine(be, y_sub, peaks_dl, orbital=orbital)
        if not peaks_final:
            peaks_final = peaks_dl
    else:
        peaks_final = peaks_dl
        redchi = rms = float("nan")

    # 转成 FitWindow 的 {center, fwhm, eta} 格式
    peak_guesses = []
    for pk in peaks_final:
        fwhm_avg = (float(pk["fwhm_g"]) + float(pk["fwhm_l"])) / 2.0
        peak_guesses.append({
            "center": float(pk["mu"]),
            "fwhm":   max(0.3, min(3.0, fwhm_avg)),
            "eta":    float(np.clip(pk["eta"], 0.0, 1.0)),
        })
    peak_guesses.sort(key=lambda d: d["center"])

    msg = f"{orbital}: {len(peak_guesses)} peaks"
    if rms == rms:
        msg += f", rms={rms:.3f}"
    return AutoFitResult(
        success=True,
        message=msg,
        orbital=orbital,
        model_name=model_path.name,
        n_peaks=len(peak_guesses),
        redchi=redchi,
        rms=rms,
        peak_guesses=peak_guesses,
        peaks_full=peaks_final,
    )
