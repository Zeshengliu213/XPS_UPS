# XPS_UPS.py (启动入口)
# ============================================================
# UPS IBW Processor v3.1 - 启动 GUI
# 功能已拆分到：reader.py, plots.py, export_csv.py, app.py
# 首次运行若缺库会自动用 pip 安装
# ============================================================

import sys
import subprocess
import traceback
import importlib.util

# 本程序必须的库（包名与 pip 名一致）
_REQUIRED_COMMON = ["numpy", "matplotlib", "pandas"]
_REQUIRED_UPS = _REQUIRED_COMMON + ["igor2"]
# DL 自动拟合：torch（首次安装 ~200 MB，CPU 版即可）+ lmfit
_REQUIRED_XPS = _REQUIRED_COMMON + ["scipy", "torch", "lmfit"]
_REQUIRED_ALL = sorted(set(_REQUIRED_UPS + _REQUIRED_XPS))


def _ensure_deps(required):
    """若缺少依赖则自动 pip 安装，然后返回是否就绪。"""
    missing = []
    for pkg in required:
        # 用 find_spec 只检查“是否安装”，避免导入 matplotlib 等造成明显启动卡顿
        # igor2 也用 find_spec（它在某些 Python 版本下导入会直接抛非 ImportError）
        if importlib.util.find_spec(pkg) is None:
            missing.append(pkg)
    if not missing:
        return True
    print("正在自动安装缺少的库:", ", ".join(missing))
    print("请稍候…")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing,
            check=True,
        )
        print("安装完成，正在启动程序…")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("自动安装失败，请手动在命令行执行：")
        print("  pip install -r requirements.txt")
        print("然后重新运行本程序。")
        if getattr(e, "returncode", None) is not None:
            input("按回车键退出…")
        return False


if __name__ == "__main__":
    try:
        if not _ensure_deps(_REQUIRED_ALL):
            sys.exit(1)
        from main_app import MainApp
        MainApp().mainloop()
    except TypeError as e:
        # 已知问题：igor2 0.5.x 在 Python 3.12 下导入/解析可能失败
        if sys.version_info >= (3, 12) and "Struct()" in str(e):
            print("检测到 Python 版本为 3.12+，且 igor2 在该版本下可能不兼容。")
            print("建议安装并使用 Python 3.11（或 3.10）后再运行本程序。")
            print("当前错误：")
            print(traceback.format_exc())
            sys.exit(1)
        print(traceback.format_exc())
        sys.exit(1)
    except Exception:
        print(traceback.format_exc())
        sys.exit(1)
