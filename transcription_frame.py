import wx
import threading
import time
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import queue
import logging
import os
from transcription_panel import TranscriptionPanel
import torch

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    import whisper
except ImportError:
    logging.error("Error: Unable to import whisper. Make sure it's installed correctly.")
    logging.error("Try running: pip install whisper openai-whisper sounddevice scipy")
    import sys
    sys.exit(1)

class TranscriptionFrame(wx.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.init_ui()
        self.init_menu()
        self.init_toolbar()
        self.init_bindings()
        self.is_recording = False
        self.recorded_frames = []
        self.SetTitle("Transcription")
        self.SetSize(800, 600)
        self.Centre()
        self.Show()

    def init_ui(self):
        self.panel = TranscriptionPanel(self)

    def init_menu(self):
        menu_bar = wx.MenuBar()
        file_menu = wx.Menu()
        self.open_item = file_menu.Append(wx.ID_OPEN, "Open Audio File")
        self.save_item = file_menu.Append(wx.ID_SAVE, "Save Transcription")
        file_menu.Append(wx.ID_EXIT, "Exit")
        menu_bar.Append(file_menu, "File")
        model_menu = wx.Menu()
        self.base_item = model_menu.Append(wx.ID_ANY, "Base")
        self.small_item = model_menu.Append(wx.ID_ANY, "Small")
        self.large_item = model_menu.Append(wx.ID_ANY, "Large")
        menu_bar.Append(model_menu, "Model")
        record_menu = wx.Menu()
        self.start_record_item = record_menu.Append(wx.ID_ANY, "Start Recording")
        self.stop_record_item = record_menu.Append(wx.ID_ANY, "Stop Recording")
        menu_bar.Append(record_menu, "Record")
        self.SetMenuBar(menu_bar)

    def init_toolbar(self):
        self.toolbar = self.CreateToolBar()
        self.transcription_model = wx.Choice(self.toolbar, choices=["base", "small", "large"])
        self.transcription_model.SetSize((120, -1))
        self.toolbar.AddControl(self.transcription_model)
        self.open_tool = self.toolbar.AddTool(wx.ID_OPEN, "Open", wx.ArtProvider.GetBitmap(wx.ART_FILE_OPEN))
        self.save_tool = self.toolbar.AddTool(wx.ID_SAVE, "Save", wx.ArtProvider.GetBitmap(wx.ART_FILE_SAVE))
        self.toolbar.Realize()

    def init_bindings(self):
        self.Bind(wx.EVT_MENU, self.on_open, self.open_item)
        self.Bind(wx.EVT_MENU, self.on_save, self.save_item)
        self.Bind(wx.EVT_MENU, self.on_exit, id=wx.ID_EXIT)
        self.Bind(wx.EVT_MENU, self.on_base, self.base_item)
        self.Bind(wx.EVT_MENU, self.on_small, self.small_item)
        self.Bind(wx.EVT_MENU, self.on_large, self.large_item)
        self.Bind(wx.EVT_MENU, self.on_start_record, self.start_record_item)
        self.Bind(wx.EVT_MENU, self.on_stop_record, self.stop_record_item)
        self.Bind(wx.EVT_TOOL, self.on_open, self.open_tool)
        self.Bind(wx.EVT_TOOL, self.on_save, self.save_tool)
        self.panel.play_pause_button.Bind(wx.EVT_BUTTON, self.on_play_pause)

    def on_exit(self, event):
        self.Close()
    
    def on_base(self, event):
        self.transcription_model.SetStringSelection("base")

    def on_small(self, event):
        self.transcription_model.SetStringSelection("small")

    def on_large(self, event):
        self.transcription_model.SetStringSelection("large")

    def on_play_pause(self, event):
        if self.panel.play_pause_button.GetLabel() == "Play":
            self.panel.play_pause_button.SetLabel("Pause")
            # TODO: Start playing the audio
        else:
            self.panel.play_pause_button.SetLabel("Play")
            # TODO: Pause the audio

    def on_save(self, event):
        with wx.FileDialog(self, "Save transcription", wildcard="Text files (*.txt)|*.txt",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as file_dialog:
            if file_dialog.ShowModal() == wx.ID_CANCEL:
                return
            with open(file_dialog.GetPath(), "w", encoding='utf-8') as file:
                file.write(self.panel.transcription_text.GetValue())

    def on_open(self, event):
        if not self.transcription_model.GetStringSelection():
            wx.MessageBox("Please select a model before opening a file.", "No Model Selected", wx.OK | wx.ICON_ERROR)
            return
        with wx.FileDialog(self, "Open file for transcription", wildcard="Audio files (*.mp3;*.wav;*.m4a)|*.mp3;*.wav;*.m4a",
                           style=wx.FD_OPEN) as file_dialog:
            if file_dialog.ShowModal() == wx.ID_CANCEL:
                return
            filepath = file_dialog.GetPath()
            logging.info(f"Selected file: {filepath}")
            print(f"Selected file: {filepath}")  # Print to console for immediate feedback
            self.transcribe_file(filepath)

    def transcribe_file(self, filepath):
        model_name = self.transcription_model.GetStringSelection()
        if not model_name:
            wx.MessageBox("Please select a model before transcribing.", "No Model Selected", wx.OK | wx.ICON_ERROR)
            return

        if not os.path.exists(filepath):
            wx.MessageBox(f"The file '{filepath}' does not exist.", "File Not Found", wx.OK | wx.ICON_ERROR)
            return

        self.progress_dialog = wx.ProgressDialog("Transcribing", "Please wait while the audio is being transcribed...",
                                                 maximum=100, parent=self, 
                                                 style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_ELAPSED_TIME | wx.PD_SMOOTH | wx.PD_CAN_ABORT)
        
        self.transcription_done = False
        self.transcription_result = ""
        self.result_queue = queue.Queue()
        
        def transcribe_thread():
            try:
                logging.info(f"Loading model: {model_name}")
                model = whisper.load_model(model_name)
                logging.info(f"Transcribing file: {filepath}")
                
                audio = whisper.load_audio(filepath)
                audio = whisper.pad_or_trim(audio)
                mel = whisper.log_mel_spectrogram(audio).to(model.device)
                
                # Detect the spoken language
                _, probs = model.detect_language(mel)
                detected_lang = max(probs, key=probs.get)
                
                # Transcribe the entire audio at once
                options = whisper.DecodingOptions(language=detected_lang, fp16=False)
                result = model.transcribe(audio, **options.__dict__)
                
                logging.info("Transcription complete")
                logging.debug(f"Transcription result: {result}")
                self.result_queue.put((result["text"], 100))
            except Exception as e:
                logging.error(f"Transcription error: {e}", exc_info=True)
                self.result_queue.put((None, 100))  # Signal error
            finally:
                self.transcription_done = True

        thread = threading.Thread(target=transcribe_thread)
        thread.start()

        self.start_time = time.time()
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.update_progress_and_transcription, self.timer)
        self.timer.Start(100)  # Update every 100ms

    def update_progress_and_transcription(self, event):
        if self.transcription_done:
            self.timer.Stop()
            wx.CallAfter(self.progress_dialog.Destroy)
            final_result, _ = self.result_queue.get()
            if final_result is not None:
                logging.info("Transcription successful, updating UI")
                wx.CallAfter(self.update_transcription, final_result)
            else:
                logging.error("Transcription failed")
                wx.CallAfter(wx.MessageBox, "Transcription failed. Please try again.", "Error", wx.OK | wx.ICON_ERROR)
            return

        elapsed_time = time.time() - self.start_time
        progress = min(int((elapsed_time / 60) * 100), 99)  # Assume 1 minute for transcription
        message = f"Transcribing... {progress}% (Elapsed time: {int(elapsed_time)} seconds)"
        logging.debug(f"Progress: {message}")
        continue_flag, skip_flag = self.progress_dialog.Update(progress, message)

        if not continue_flag:  # User cancelled
            logging.info("User cancelled transcription")
            self.transcription_done = True
            return

    def update_transcription(self, transcription):
        logging.info("Updating transcription in UI")
        self.panel.transcription_text.SetValue(transcription)
        self.panel.transcription_text.ShowPosition(self.panel.transcription_text.GetLastPosition())

    def on_start_record(self, event):
        if not self.is_recording:
            self.is_recording = True
            self.recorded_frames = []
            
            def callback(indata, frames, time, status):
                if status:
                    print(status)
                self.recorded_frames.append(indata.copy())

            self.stream = sd.InputStream(callback=callback, channels=2, samplerate=44100)
            self.stream.start()
            wx.MessageBox("Recording started. Use 'Stop Recording' to finish.", "Recording", wx.OK | wx.ICON_INFORMATION)

    def on_stop_record(self, event):
        if self.is_recording:
            self.stream.stop()
            self.stream.close()
            self.is_recording = False
            
            recorded_data = np.concatenate(self.recorded_frames, axis=0)
            filename = wx.FileSelector("Save recorded audio", default_extension="wav",
                                       wildcard="WAV files (*.wav)|*.wav",
                                       flags=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
            if filename:
                write(filename, 44100, recorded_data)
                wx.MessageBox(f"Recording saved as {filename}", "Recording Saved", wx.OK | wx.ICON_INFORMATION)