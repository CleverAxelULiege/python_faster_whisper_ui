import time
import tkinter as tk
from pathlib import Path
import re
import json
WHISPER_LANGUAGES = {
    "Anglais": "en",
    "Chinois": "zh",
    "Allemand": "de",
    "Espagnol": "es",
    "Russe": "ru",
    "Coréen": "ko",
    "Français": "fr",
    "Japonais": "ja",
    "Portugais": "pt",
    "Turc": "tr",
    "Polonais": "pl",
    "Néerlandais": "nl",
    "Suédois": "sv",
    "Italien": "it",
    "Indonésien": "id",
    "Finnois": "fi",
    "Ukrainien": "uk",
    "Grec": "el",
    "Tchèque": "cs",
    "Roumain": "ro",
    "Danois": "da",
    "Hongrois": "hu",
    "Norvégien": "no"
}


class WhisperUIInput:
    def __init__(self, root):
        self.allowed_extensions = [".txt"]
        self.save_directory_path = tk.StringVar(master=root, value=Path.home())
        self.save_file_extension = tk.StringVar(master=root, value=self.allowed_extensions[0])
        self.save_filename = tk.StringVar(master=root, value=str(int(time.time())))
        self.save_transcript_file_full_path = tk.StringVar(master=root)
        self.audio_file_path = tk.StringVar(master=root)
        self.language = tk.StringVar(master=root)
        self.should_add_timestamp = tk.BooleanVar(master=root, value=False)
        self.transcript_quality_list = ("Rapide (moins précis)", "Équilibré (recommandé)", "Précis (plus lent)")
        self.transcript_quality = tk.StringVar(master=root, value=self.transcript_quality_list[0])

        self.progress = tk.IntVar(master=root, value=0)
        self.normalize_save_full_path()
        self.__init_event_listeners()

        self.model = tk.StringVar(master=root, value="")
        self.models_names = []
        self.models = {}
        self.default_model = None

    def normalize_save_full_path(self, *args):
        self.save_filename.set(self.__sanitize_filename())

        pathObject = Path(
            self.save_directory_path.get() + 
            "/" + 
            self.save_filename.get() + 
            self.save_file_extension.get()
        )

        self.save_transcript_file_full_path.set(pathObject.resolve())
    
    def __init_event_listeners(self):
        self.save_filename.trace_add("write", self.normalize_save_full_path)
        self.save_file_extension.trace_add("write", self.normalize_save_full_path)
        self.save_directory_path.trace_add("write", self.normalize_save_full_path)

    def __sanitize_filename(self):
        filename = self.save_filename.get().replace(" ", "_")
        filename = re.sub(r'[\\/:*?"<>|]', '', filename)
        filename = re.sub(r'[\0]', '', filename)
        
        #substr
        return filename[:255]
    
    def load_config(self, config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            models = json.load(f)

        valid_models = [
            model for model in models
            if model.get("model_name") and model.get("model_path")
        ]

        self.models = valid_models
        self.models_names = [model["model_name"] for model in valid_models]

        self.default_model = next(
            (model["model_name"] for model in valid_models if model.get("default") is True),
            None
        )

    def get_model_path(self):
        selected_name = self.model.get()

        for model in self.models:
            if model["model_name"] == selected_name:
                return model["model_path"]

        return None