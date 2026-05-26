import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import pandas as pd
import os
import glob
import argparse
import matplotlib.pyplot as plt
import librosa.display

# Suppress TF warnings
import logging
tf.get_logger().setLevel(logging.ERROR)

# Set Matplotlib to non-interactive mode (for servers without screens)
plt.switch_backend('Agg')

def load_model():
    print("Loading YAMNet model...")
    # Load the model from TensorFlow Hub
    yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
    model = hub.load(yamnet_model_handle)
    
    # Load the class map
    class_map_path = model.class_map_path().numpy()
    
    # FIX: Decode bytes to string for Pandas compatibility
    if isinstance(class_map_path, bytes):
        class_map_path = class_map_path.decode('utf-8')
        
    print(f"Class map path: {class_map_path}")
    class_names = pd.read_csv(class_map_path)['display_name'].tolist()
    return model, class_names

def load_wav_for_model(filename):
    """Loads a wav file, resamples to 16kHz, and normalizes it for YAMNet."""
    try:
        # Load with librosa (automatically handles resampling and mono conversion)
        wav, sr = librosa.load(filename, sr=16000, mono=True)
        return wav
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def save_event_spectrogram(waveform, sr, timestamp, label, confidence, filename, output_dir="detection_images"):
    """
    Saves a spectrogram of the 2-second window around the detected event.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define window: 0.5s before to 1.5s after the timestamp
    start_time = max(0, timestamp - 0.5)
    end_time = min(len(waveform)/sr, timestamp + 1.5)
    
    # Convert time to samples
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    
    snippet = waveform[start_sample:end_sample]
    
    if len(snippet) == 0:
        return

    # Create plot
    plt.figure(figsize=(10, 4))
    
    # Compute spectrogram
    S = librosa.feature.melspectrogram(y=snippet, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # Display
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Detected: {label} ({confidence:.2f}) at {timestamp:.2f}s\nFile: {filename}")
    plt.tight_layout()
    
    # Save file
    safe_filename = filename.replace(".WAV", "").replace(".wav", "")
    save_path = os.path.join(output_dir, f"{safe_filename}_{timestamp:.0f}s_{label}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"   >>> [IMAGE SAVED] {save_path}")

def probe_specific_timestamp(model, class_names, filepath, minute, second):
    """
    Loads a specific tiny chunk of the file to see exactly what the model thinks 
    of that specific moment.
    """
    # Calculate offset
    offset = (minute * 60) + second
    print(f"\n--- PROBE MODE ACTIVATED ---")
    print(f"Probing file: {os.path.basename(filepath)}")
    print(f"Timestamp: {minute}m {second}s (Offset: {offset}s)")
    print(f"Loading 3-second clip around this time...")
    
    # Load just 3 seconds around that time (faster than loading whole file)
    try:
        # Start 1 second before to capture context
        wav, sr = librosa.load(filepath, sr=16000, mono=True, offset=max(0, offset-1), duration=3.0)
    except Exception as e:
        print(f"Error loading file snippet: {e}")
        return

    # Run model
    scores, embeddings, spectrogram = model(wav)
    scores_np = scores.numpy()
    
    # Average scores across the 3 seconds to get the general vibe
    mean_scores = np.mean(scores_np, axis=0)
    
    # Get top 5 predictions
    top_n_indices = np.argsort(mean_scores)[::-1][:5]
    
    print("\nTop 5 Model Predictions for this moment:")
    print("-" * 40)
    for i in top_n_indices:
        print(f"{class_names[i]:<25} : {mean_scores[i]:.4f} ({mean_scores[i]*100:.1f}%)")
    print("-" * 40)
    
    # Generate Probe Image
    top_label = class_names[top_n_indices[0]]
    top_score = mean_scores[top_n_indices[0]]
    # We pass '1.0' as timestamp because we loaded the clip such that the event is roughly in the middle
    save_event_spectrogram(wav, sr, 1.0, f"PROBE_{top_label}", top_score, f"PROBE_{minute}m{second}s.png")

def analyze_audio(model, class_names, filepath, target_indices, threshold=0.2):
    """
    Runs the model on a file and returns max confidence if a gunshot is detected.
    Returns (is_gunshot, max_confidence, top_class_name, timestamp_seconds, waveform)
    """
    waveform = load_wav_for_model(filepath)
    if waveform is None: return False, 0.0, "Error", 0.0, None

    # Run the model
    scores, embeddings, spectrogram = model(waveform)
    
    detected = False
    best_score = 0.0
    best_label = ""
    best_timestamp = 0.0
    
    scores_np = scores.numpy() # Convert to numpy for easier indexing
    
    # Check if any target class exceeds threshold in any frame
    for idx in target_indices:
        class_scores = scores_np[:, idx]
        max_class_score = np.max(class_scores)
        
        if max_class_score > best_score:
            best_score = max_class_score
            best_label = class_names[idx]
            frame_index = np.argmax(class_scores)
            best_timestamp = frame_index * 0.48
            
    if best_score > threshold:
        detected = True
        
    return detected, best_score, best_label, best_timestamp, waveform

def main():
    # --- CONFIGURATION ---
    SEARCH_DIR = "/Volumes/aid_elephants_interaction/Audio Data/2025/05_FR53/02-11-25 to 02-25-25/20250211/"
    SPECIFIC_FILE = "20250211_190000.WAV"
    OUTPUT_FILE = "detected_gunshots.csv"
    THRESHOLD = 0.25
    
    # --- PROBE SETTINGS ---
    # Set PROBE_MODE to True to investigate a specific timestamp
    PROBE_MODE = True
    PROBE_MIN = 37
    PROBE_SEC = 24  # UPDATED to 24s based on your request
    
    # Target classes
    target_classes = ['Gunshot, gunfire', 'Cap gun', 'Explosion', 'Fireworks']
    
    # 1. Load Model
    try:
        model, class_names = load_model()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Find indices for targets
    target_indices = []
    for t in target_classes:
        if t in class_names:
            target_indices.append(class_names.index(t))
    
    # Locate File
    full_path = os.path.join(SEARCH_DIR, SPECIFIC_FILE)
    if not os.path.exists(full_path):
        # Fallback for case sensitivity
        if os.path.exists(os.path.join(SEARCH_DIR, SPECIFIC_FILE.lower())):
             full_path = os.path.join(SEARCH_DIR, SPECIFIC_FILE.lower())
        else:
            print(f"ERROR: File not found: {full_path}")
            return

    # 2. RUN PROBE IF ACTIVATED
    if PROBE_MODE:
        probe_specific_timestamp(model, class_names, full_path, PROBE_MIN, PROBE_SEC)
        return  # Stop here, don't run the full analysis

    # 3. Standard Analysis (If Probe Mode is False)
    files = [full_path]
    results = []
    print(f"Starting analysis on single file...")
    
    for i, file_path in enumerate(files):
        try:
            print(f"Analyzing: {os.path.basename(file_path)}...") 
            is_gunshot, confidence, label, timestamp, waveform = analyze_audio(model, class_names, file_path, target_indices, THRESHOLD)
            
            if is_gunshot:
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                
                print(f"   >>> [MATCH] {label} ({confidence:.2f}) at {timestamp:.2f}s ({minutes}m {seconds}s) : {os.path.basename(file_path)}")
                save_event_spectrogram(waveform, 16000, timestamp, label, confidence, os.path.basename(file_path))

                results.append({
                    "filepath": file_path,
                    "filename": os.path.basename(file_path),
                    "predicted_label": label,
                    "confidence": confidence,
                    "timestamp_seconds": timestamp,
                    "timestamp_readable": f"{minutes}m {seconds}s"
                })
            else:
                 print(f"No gunshots detected (Highest score below {THRESHOLD}).")
                
        except KeyboardInterrupt:
            print("\nStopping early...")
            break
        except Exception as e:
            print(f"Error on file {file_path}: {e}")

if __name__ == "__main__":
    main()