import wx
import threading
import time

class TranscriptionPanel(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)

        # Create a text control to display the transcription
        self.transcription_text = wx.TextCtrl(self, value="", style=wx.TE_MULTILINE | wx.TE_READONLY)

        # Create play/pause button
        self.play_pause_button = wx.Button(self, label="Play")
        self.playing = False  # To track play/pause status

        # Bind the play/pause button event
        self.play_pause_button.Bind(wx.EVT_BUTTON, self.on_play_pause)

        # Create a progress bar
        self.progress = wx.Gauge(self, range=100)

        # Add controls to the layout
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.transcription_text, 1, wx.EXPAND | wx.ALL, 10)
        sizer.Add(self.play_pause_button, 0, wx.ALIGN_CENTER | wx.ALL, 10)
        sizer.Add(self.progress, 0, wx.EXPAND | wx.ALL, 10)
        self.SetSizer(sizer)

    def on_play_pause(self, event):
        if not self.playing:
            self.playing = True
            self.play_pause_button.SetLabel("Pause")
            # Start transcription in a background thread
            self.transcription_thread = threading.Thread(target=self.transcribe_audio)
            self.transcription_thread.start()
        else:
            self.playing = False
            self.play_pause_button.SetLabel("Play")

    def transcribe_audio(self):
        for i in range(100):
            if not self.playing:
                break  # Pause transcription if play is stopped
            wx.CallAfter(self.transcription_text.AppendText, f"Transcription line {i + 1}\n")
            wx.CallAfter(self.update_progress, i + 1)
            time.sleep(0.1)  # Simulate processing time

        # Reset once transcription is done
        wx.CallAfter(self.play_pause_button.SetLabel, "Play")
        wx.CallAfter(self.progress.SetValue, 0)
        self.playing = False

    def update_progress(self, value):
        """Update the progress bar in a thread-safe manner"""
        if value <= 100:
            self.progress.SetValue(value)


class TranscriptionApp(wx.App):
    def OnInit(self):
        frame = wx.Frame(None, title="Transcription App", size=(400, 300))
        TranscriptionPanel(frame)
        frame.Show(True)
        return True


if __name__ == "__main__":
    app = TranscriptionApp()
    app.MainLoop()
