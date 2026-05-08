import os
import traceback

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ui_theme import COLORS, FONTS, create_button, create_section, create_log_panel
from file_loader import FileLoader
from fit_window import FitWindow


class XPSFrame(tk.Frame):
    """
    XPS 模块主界面（可嵌入 Notebook Tab）：负责选择文件、加载谱、打开分峰拟合窗口。
    复用现有 FitWindow 的交互与拟合/导出逻辑。
    """

    def __init__(self, master):
        super().__init__(master, bg=COLORS["bg"])

        self.files: list[str] = []
        self.spectra: list[dict] = []
        self.out_dir: str | None = None

        self._loader = FileLoader(schedule_fn=lambda fn: self.after(0, fn))
        self._build_ui()

    # ------------------------------ UI build ------------------------------ #

    def _build_ui(self):
        tk.Frame(self, bg=COLORS["primary"], height=3).pack(fill="x")

        header = tk.Frame(self, bg=COLORS["bg"])
        header.pack(fill="x", padx=20, pady=(12, 6))
        tk.Label(header, text="XPS Peak Fitting", font=FONTS["title"], fg=COLORS["text"], bg=COLORS["bg"]).pack(
            side="left"
        )
        tk.Label(
            header,
            text="Core-level peak deconvolution",
            font=FONTS["badge"],
            fg=COLORS["secondary"],
            bg=COLORS["button_bg"],
            padx=8,
            pady=3,
        ).pack(side="left", padx=10)

        file_box = create_section(self, "1) Files", fill="x", padx=20, pady=4)
        row = tk.Frame(file_box, bg=COLORS["card"])
        row.pack(fill="x")
        create_button(row, "Select .ibw files", self.pick_files, primary=True).pack(side="left")
        create_button(row, "Choose output folder", self.pick_out_dir).pack(side="left", padx=8)

        self.out_dir_var = tk.StringVar(value="Default: same folder as first IBW")
        tk.Label(row, textvariable=self.out_dir_var, fg=COLORS["secondary"], bg=COLORS["card"], font=FONTS["small"]).pack(
            side="left", padx=8
        )

        act_box = create_section(self, "2) Actions", fill="x", padx=20, pady=4)
        create_button(act_box, "⚙  Open Peak Fitting", self.open_fit_window, primary=True).pack(side="left")
        create_button(act_box, "Clear list", self.clear_list, danger=True).pack(side="left", padx=8)

        main = tk.Frame(self, bg=COLORS["bg"])
        main.pack(fill="both", expand=True, padx=20, pady=(4, 0))

        left = create_section(main, "Selected Files", side="left", fill="both", expand=True)
        self.listbox = tk.Listbox(
            left,
            height=12,
            font=FONTS["body"],
            bg=COLORS["list_bg"],
            fg=COLORS["list_fg"],
            selectbackground=COLORS["list_select"],
            selectforeground=COLORS["text"],
            relief="flat",
            highlightthickness=0,
            activestyle="none",
        )
        self.listbox.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(left, orient="vertical", command=self.listbox.yview)
        sb.pack(side="left", fill="y")
        self.listbox.config(yscrollcommand=sb.set)

        right = create_section(main, "Log", side="left", fill="both", expand=True, padx=(8, 0))
        self.log = create_log_panel(right)

    # ------------------------------ actions ------------------------------ #

    def _log(self, msg: str, tag: str = ""):
        self.log.insert("end", msg + "\n", tag)
        self.log.see("end")

    def pick_files(self):
        files = filedialog.askopenfilenames(title="Select .ibw files", filetypes=[("IGOR Binary Wave", "*.ibw")])
        if not files:
            return
        self.files = list(files)
        self.listbox.delete(0, "end")
        for f in self.files:
            self.listbox.insert("end", os.path.basename(f))
        self._log(f"Selected {len(self.files)} file(s).", "ok")

        self._log("Loading spectra in background…", "dim")
        self._loader.load(self.files, self._on_loaded)

    def _on_loaded(self, spectra, ok_count, bad_count, failures):
        self.spectra = spectra
        self._log(f"Loaded: {ok_count}  Failed: {bad_count}", "ok" if bad_count == 0 else "warn")
        for fp, err in failures:
            self._log(f"  {os.path.basename(fp)}: {err}", "err")

    def pick_out_dir(self):
        d = filedialog.askdirectory(title="Choose output folder")
        if not d:
            return
        self.out_dir = d
        self.out_dir_var.set(d)

    def clear_list(self):
        self.files = []
        self.spectra = []
        self.listbox.delete(0, "end")
        self._log("Cleared.", "ok")

    def get_out_dir(self):
        if self.out_dir:
            return self.out_dir
        if self.files:
            return os.path.dirname(self.files[0])
        return os.getcwd()

    def open_fit_window(self):
        if not self.spectra:
            messagebox.showwarning("Not Loaded", "Please select and load at least one .ibw file first.")
            return
        try:
            FitWindow(self, self.spectra)
        except Exception:
            self._log("Open fit window failed:\n" + traceback.format_exc(), "err")
            messagebox.showerror("Error", "Failed to open fit window. Check Log.")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("XPS Peak Fitting")
    root.geometry("980x680")
    root.minsize(900, 620)
    XPSFrame(root).pack(fill="both", expand=True)
    root.mainloop()

