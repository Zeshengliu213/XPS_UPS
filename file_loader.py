# file_loader.py
# ============================================================
# 异步 IBW 文件加载器（带取消机制）
# ============================================================

import os
import threading

from reader import read_ibw_ups


class FileLoader:
    """
    在后台线程中加载 IBW 文件列表，完成后通过 schedule_fn 回调主线程。

    schedule_fn : 将函数调度到主线程执行的函数，典型用法为 widget.after(0, fn)。
    reader_func : 读取单个文件的函数，签名 (file_path) → (x, y, y_norm, meta)。
                  默认为 read_ibw_ups，可替换以支持其他格式。
    on_done     : callback(spectra, ok_count, bad_count, failures)
                  failures 为 [(file_path, error_str), ...]
    """

    def __init__(self, schedule_fn, reader_func=None):
        self._schedule = schedule_fn
        self._reader = reader_func or read_ibw_ups
        self._cancel = threading.Event()

    def load(self, files, on_done):
        """启动后台加载。可在加载过程中调用 cancel() 跳过剩余文件。"""
        self._cancel.clear()

        def run():
            result = self._load_body(files)
            self._schedule(lambda: on_done(*result))

        threading.Thread(target=run, daemon=True).start()

    def cancel(self):
        """请求取消：已在读取中的当前文件不会中断，但后续文件跳过。"""
        self._cancel.set()

    def _load_body(self, files):
        spectra, failures = [], []
        for fp in files:
            if self._cancel.is_set():
                break
            try:
                x, y, y_norm, meta = self._reader(fp)
                spectra.append({
                    "file": fp,
                    "dir": os.path.dirname(fp),
                    "base": os.path.splitext(os.path.basename(fp))[0],
                    "x": x, "y": y, "y_norm": y_norm,
                    "meta": meta,
                })
            except Exception as e:
                failures.append((fp, str(e)))
        return spectra, len(spectra), len(failures), failures
