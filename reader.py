# reader.py
# ============================================================
# UPS IBW 读取：从 IGOR .ibw 构建能量轴并归一化
# ============================================================

import os
import numpy as np


def read_ibw_ups(file_path: str):
    """
    Read UPS spectrum from IGOR IBW.

    English:
    - sfB: start value of x-axis
    - sfA: step size (can be negative)
    - x = sfB + sfA * arange(N)
    - Sort x descending for UPS plotting (high -> low BE)
    - Normalize y to 0–1 using min-max

    中文：
    - sfB：横轴起点
    - sfA：横轴步长（可能为负）
    - x = sfB + sfA * np.arange(N)
    - UPS 画图习惯：结合能从大到小显示
    - y 使用 min-max 归一化到 0–1
    """
    # Python 3.12 tightened struct.Struct.__new__, breaking igor2.struct.Structure
    # subclassing (TypeError: Struct() takes at most 1 keyword argument).
    # igor2/__init__.py eagerly imports binarywave, so we can't go through the
    # package. Load igor2.struct directly from disk, patch Structure.__new__,
    # register it in sys.modules under its expected name, *then* let igor2's
    # package import find the already-patched module.
    import sys as _sys
    import importlib.util as _importlib_util
    import os as _os
    if "igor2.binarywave" not in _sys.modules:
        # Locate igor2 package directory without importing it (importing
        # triggers the broken __init__.py).
        _igor2_dir = None
        for _p in _sys.path:
            _cand = _os.path.join(_p, "igor2", "struct.py")
            if _os.path.isfile(_cand):
                _igor2_dir = _os.path.dirname(_cand); break

        if _igor2_dir:
            try:
                # Pre-create an empty 'igor2' package entry so 'igor2.struct'
                # can be registered before igor2/__init__.py runs.
                if "igor2" not in _sys.modules:
                    _pkg_spec = _importlib_util.spec_from_file_location(
                        "igor2",
                        _os.path.join(_igor2_dir, "__init__.py"),
                        submodule_search_locations=[_igor2_dir],
                    )
                    _pkg_mod = _importlib_util.module_from_spec(_pkg_spec)
                    _sys.modules["igor2"] = _pkg_mod
                    # Don't exec_module yet; we just need the namespace.

                _spec = _importlib_util.spec_from_file_location(
                    "igor2.struct", _os.path.join(_igor2_dir, "struct.py"))
                _igor2_struct_mod = _importlib_util.module_from_spec(_spec)
                _sys.modules["igor2.struct"] = _igor2_struct_mod
                _spec.loader.exec_module(_igor2_struct_mod)  # type: ignore[union-attr]

                import struct as _stdstruct
                _orig_new = _stdstruct.Struct.__new__

                # Python 3.12 made _struct.Struct immutable. igor2's Structure
                # subclasses Struct and tries to (a) construct with kwargs
                # (rejected by __new__) and (b) reset its format later via
                # super().__init__(format=...) (rejected by __init__).
                # Replace inheritance with composition: keep an inner Struct
                # and delegate the binary ops to it.
                def _patched_new(cls, *a, **kw):
                    return _orig_new(cls, b"")
                _igor2_struct_mod.Structure.__new__ = staticmethod(_patched_new)  # type: ignore[assignment]

                def _patched_get_format(self):
                    fmt = self.byte_order + "".join(self.sub_format())
                    fmt = fmt.replace("P", "I")
                    self._inner = _stdstruct.Struct(fmt)
                    return fmt
                _igor2_struct_mod.Structure.get_format = _patched_get_format  # type: ignore[assignment]

                def _ensure_inner(self):
                    if not hasattr(self, "_inner"):
                        self.get_format()
                    return self._inner

                def _patched_pack(self, data):
                    args = list(self._pack_item(data))
                    try:
                        return _ensure_inner(self).pack(*args)
                    except BaseException:
                        raise ValueError(self._inner.format)

                def _patched_pack_into(self, buffer, offset, data):
                    args = list(self._pack_item(data))
                    return _ensure_inner(self).pack_into(buffer, offset, *args)

                def _patched_unpack(self, *args, **kwargs):
                    args = _ensure_inner(self).unpack(*args, **kwargs)
                    return self._unpack_item(args)

                def _patched_unpack_from(self, buffer, offset=0, **kwargs):
                    args = _ensure_inner(self).unpack_from(buffer, offset, **kwargs)
                    return self._unpack_item(args)

                _igor2_struct_mod.Structure.pack = _patched_pack
                _igor2_struct_mod.Structure.pack_into = _patched_pack_into
                _igor2_struct_mod.Structure.unpack = _patched_unpack
                _igor2_struct_mod.Structure.unpack_from = _patched_unpack_from
                _igor2_struct_mod.Structure.format = property(lambda s: _ensure_inner(s).format)
                _igor2_struct_mod.Structure.size = property(lambda s: _ensure_inner(s).size)
            except Exception:
                _sys.modules.pop("igor2.struct", None)
                _sys.modules.pop("igor2", None)

    try:
        from igor2 import binarywave  # lazy import: igor2 may fail on Python 3.12+
    except Exception as e:
        raise RuntimeError(
            "Failed to import igor2.binarywave. "
            "If you're on Python 3.12+, igor2 may be incompatible. "
            "Please use Python 3.11/3.10, or reinstall a compatible igor2."
        ) from e

    w = binarywave.load(file_path)
    wave = w["wave"]

    y = np.asarray(wave["wData"]).squeeze()
    if y.ndim != 1:
        raise ValueError(f"{os.path.basename(file_path)}: not 1D wave, shape={y.shape}")

    wh = wave["wave_header"]

    start = float(wh["sfB"][0])  # x start / 起点
    step  = float(wh["sfA"][0])  # x step / 步长

    x = start + step * np.arange(len(y))

    # Sort x descending (UPS convention) / 按 BE 从大到小排序
    idx = np.argsort(x)[::-1]
    x = x[idx]
    y = y[idx]

    # Normalize to 0–1 / 归一化到 0–1
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    if abs(y_max - y_min) < 1e-15:
        y_norm = y.copy()
    else:
        y_norm = (y - y_min) / (y_max - y_min)

    meta = {"sfB_start": start, "sfA_step": step, "N": len(y)}
    return x, y, y_norm, meta
