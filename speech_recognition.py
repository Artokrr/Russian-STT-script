import os
import time
import torch
import librosa
import json
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

with open('config/stt_config.json', 'r') as f:
    config = json.load(f)

dir_path = config['dir_path']

# Ensure the directory exists
os.makedirs(dir_path, exist_ok=True)

print("Model loading...")

# Initialize the wav2vec2 model and processor
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

print("Model loaded, waiting for new files...")

# Assume profane_words is a list of profane words
profane_words = ["хуй", "пидорас", "сука"]

def handle_new_file(path):
    # Load the .wav file
    try:
        speech_array, sampling_rate = librosa.load(path, sr=16_000)
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    print("File loaded, transcribing...")

    # Preprocess the audio
    inputs = processor(speech_array, sampling_rate=16_000, return_tensors="pt", padding=True)

    # Transcribe the audio
    with torch.no_grad():
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    filename = os.path.basename(path)
    nickname = filename.split('_')[0]
    print(f"Player's nickname: {nickname} Transcription: {transcription}")

    # Check for profanity
    if any(word in transcription.lower().split() for word in profane_words):
        print("Profanity detected!")
        # Mute the player who used profanity (to be implemented)
        pass

# Define the event handler
class AudioFileHandler(PatternMatchingEventHandler):
    patterns = ["*.wav"]

    def on_created(self, event):
        handle_new_file(event.src_path)

# Start the observer

print(f"Monitoring directory: {dir_path}")

observer = Observer()
observer.schedule(AudioFileHandler(), path=dir_path, recursive=True)
observer.start()

# Keep the script running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()