import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import wx
import logging
from transcription_frame import TranscriptionFrame

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    logging.info("Starting WhisperUI application")
    app = wx.App()
    frame = TranscriptionFrame(None)
    app.MainLoop()