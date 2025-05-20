import tkinter as tk
from tkinter import filedialog, messagebox

class DeepFakeView:
    def __init__(self, root):
        self.root = root
        self.root.title("SyntSpeech Detector")

        self.path_label = tk.Label(root, text="Audio File Path:")
        self.path_label.pack()

        self.path_entry = tk.Entry(root, width=50)
        self.path_entry.pack()

        self.browse_button = tk.Button(root, text="Browse")
        self.browse_button.pack()

        self.check_button = tk.Button(root, text="Check")
        self.check_button.pack()

        self.about_button = tk.Button(root, text="About")
        self.about_button.pack()

        self.result_label = tk.Label(root, text="", fg="blue")
        self.result_label.pack()

    def get_audio_path(self):
        return self.path_entry.get()

    def set_audio_path(self, path):
        self.path_entry.delete(0, tk.END)
        self.path_entry.insert(0, path)

    def set_result(self, result):
        self.result_label.config(text=f"Prediction: {result}")

    def show_about(self):
        messagebox.showinfo("About", "Author: K. Antropov\nWebsite: https://github.com/noescapefromreality/deepfake_audio_detection_svm_tk")