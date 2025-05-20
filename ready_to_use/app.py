import tkinter as tk
from model import DeepFakeModel
from view import DeepFakeView
from presenter import DeepFakePresenter

if __name__ == "__main__":
    root = tk.Tk()
    view = DeepFakeView(root)
    model = DeepFakeModel("itw_light_split_svm.pkl", "itw_light_split_scaler.pkl")
    presenter = DeepFakePresenter(view, model)
    root.mainloop()