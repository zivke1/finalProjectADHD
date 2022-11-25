
# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer
import tkinter
from pathlib import Path

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage


import build_load_dataset.gui as load_EEG_dataSet_win
import build_generate_graphs.gui as generate_graphs_win
import build_analyze_data.gui as analyze_data_win



OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("./assets")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

window = Tk()
window.title("Main Page")
class win:
    def __init__(self, *args, **kwargs):
        window.geometry("1170x687")
        window.configure(bg="#FFFFFF")

        canvas = Canvas(
            window,
            bg="#FFFFFF",
            height=687,
            width=1170,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )

        canvas.place(x=0, y=0)
        canvas.create_rectangle(
            0.0,
            0.0,
            234.0,
            684.0,
            fill="#9272EB",
            outline="")

        button_image_1 = PhotoImage(
            file=relative_to_assets("button_1.png"))
        button_1 = Button(
            image=button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: {
                window.destroy():
                    generate_graphs_win.win()
            },
            relief="flat"
        )
        button_1.place(
            x=34.0,
            y=92.0,
            width=168.0,
            height=37.0
        )

        button_image_2 = PhotoImage(
            file=relative_to_assets("button_2.png"))
        button_2 = Button(
            image=button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: {
                window.destroy():
                    load_EEG_dataSet_win.win()
            },
            relief="flat"
        )
        button_2.place(
            x=34.0,
            y=154.0,
            width=168.0,
            height=37.0
        )

        canvas.create_text(
            27.0,
            33.0,
            anchor="nw",
            text="EEG Recordings Analyzer",
            fill="#FFFAFA",
            font=("JejuMyeongjo", 14 * -1)
        )

        canvas.create_rectangle(
            234.0,
            0.0,
            1167.0,
            684.0,
            fill="#E2D8EF",
            outline="")

        canvas.create_text(
            515.0,
            74.0,
            anchor="nw",
            text="Welcome To EEG Recordings Analyzer",
            fill="#000000",
            font=("JejuMyeongjo", 24 * -1)
        )
        window.resizable(False, False)
        window.mainloop()

if __name__ == "__main__":
    win()