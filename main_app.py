import tkinter as tk
from tkinter import ttk

from ui_theme import COLORS, FONTS, apply_ttk_theme
from app import UPSFrame
from xps_app import XPSFrame


class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("XPS / UPS IBW Processor")
        self.geometry("1080x740")
        self.minsize(980, 680)

        self.configure(bg=COLORS["bg"])
        self.option_add("*Font", FONTS["body"])
        self.option_add("*Background", COLORS["bg"])
        apply_ttk_theme(self)

        self._build_ui()

    def _build_ui(self):
        tk.Frame(self, bg=COLORS["primary"], height=3).pack(fill="x")

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=10)

        ups = UPSFrame(nb)
        xps = XPSFrame(nb)

        nb.add(ups, text="UPS")
        nb.add(xps, text="XPS")


if __name__ == "__main__":
    MainApp().mainloop()

