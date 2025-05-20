from tkinter import filedialog
from model import DeepFakeModel
from view import DeepFakeView

class DeepFakePresenter:
    def __init__(self, view: DeepFakeView, model: DeepFakeModel):
        self.view = view
        self.model = model

        self.view.browse_button.config(command=self.on_browse)
        self.view.check_button.config(command=self.on_check)
        self.view.about_button.config(command=self.view.show_about)

    def on_browse(self):
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if file_path:
            self.view.set_audio_path(file_path)

    def on_check(self):
        path = self.view.get_audio_path()
        try:
            result = self.model.predict(path)
            self.view.set_result(result)
        except Exception as e:
            self.view.set_result("Error: " + str(e))