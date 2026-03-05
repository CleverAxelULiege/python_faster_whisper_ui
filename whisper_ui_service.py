import json
import threading
import logging
from datetime import timedelta
import librosa
import os

from faster_whisper import WhisperModel


class WhisperUIService:
    def __init__(self, root, progress_bar_status):
        self.transcript_quality_beam_size = {
            "Rapide (moins précis)": 1,
            "Équilibré (recommandé)": 3,
            "Précis (plus lent)": 5
        }
        self.audio_path = ""
        self.root = root
        self.progress_bar_status = progress_bar_status

        print("Device : cpu")

        logging.basicConfig(
            filename="app.log",
            level=logging.ERROR,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

        self.model = None


    def load_config(self, config_file_path):
        with open(config_file_path, "r") as f:
            models = json.load(f)

        for model in models:
            if model.get("default") is True:
                self.model_id = model.get("model_path")

        # load Faster-Whisper model
        # self.model = WhisperModel(
        #     r"",
        #     device="cpu",
        #     compute_type="int8",
        #     cpu_threads=6,
        #     num_workers=2
        # )
        self.model = None

    

    def setModel(self, model_path):
        cpu_count = os.cpu_count()
        self.model = WhisperModel(
            model_path,
            device="cpu",
            compute_type="int8",
            cpu_threads=  int(cpu_count / 2) if cpu_count != None else 4,
            num_workers=2
        )


    def transcribe(self, on_finish, language, output_file, should_add_timestamp, transcription_quality):
        beam_size = self.transcript_quality_beam_size[transcription_quality]
        t = threading.Thread(
            target=self.__thread,
            daemon=True,
            args=(on_finish, language, output_file, should_add_timestamp, beam_size),
        )
        t.start()


    def __thread(self, on_finish, language, output_file, should_add_timestamp, beam_size):
        try:
            # get audio duration for progress tracking
            audio, sr = librosa.load(self.audio_path, sr=None)
            duration = librosa.get_duration(y=audio, sr=sr)

            segments, info = self.model.transcribe(
                self.audio_path,
                language=language,
                task="transcribe",
                beam_size=beam_size,
                vad_filter=True
            )

            with open(output_file, "w", encoding="utf-8") as file:

                for segment in segments:

                    text = segment.text.strip()

                    if should_add_timestamp:
                        start_str = self.__format_time(segment.start)
                        end_str = self.__format_time(segment.end)

                        line = f"[De : {start_str}; à : {end_str}] {text}\n"
                    else:
                        line = f"{text}\n"

                    # STREAM write immediately
                    file.write(line)
                    file.flush()

                    progress = (segment.end / duration) * 100
                    self.progress_bar_status["percentage_done"] = progress

                    self.root.event_generate(
                        "<<update_progress_bar_event>>",
                        when="tail",
                        state=0
                    )

            on_finish(output_file)

        except Exception as e:
            print(e)
            logging.exception("Unhandled exception occurred")
            self.root.event_generate(
                "<<update_progress_bar_event>>",
                when="tail",
                state=1
            )


    def __format_time(self, seconds: float) -> str:
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        minutes, seconds = divmod(total_seconds, 60)
        return f"{minutes:02d}:{seconds:02d}"