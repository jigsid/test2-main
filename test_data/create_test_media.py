import cv2
import numpy as np
import wave
import struct

# Create a test video
def create_test_video():
    output_path = 'test_data/sample_video.mp4'
    height, width = 480, 640
    fps = 30
    duration = 5  # seconds
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for i in range(fps * duration):
        # Create a frame with changing colors
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = i % 255  # Blue channel
        frame[:, :, 1] = (i * 2) % 255  # Green channel
        frame[:, :, 2] = (i * 3) % 255  # Red channel
        
        # Add some text
        cv2.putText(frame, 'Test Video', (width//4, height//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Created test video: {output_path}")

# Create a test audio file
def create_test_audio():
    output_path = 'test_data/sample_music.wav'
    duration = 5  # seconds
    sample_rate = 44100
    frequency = 440  # A4 note
    
    # Generate samples
    samples = []
    for i in range(int(duration * sample_rate)):
        sample = int(32767 * np.sin(2 * np.pi * frequency * i / sample_rate))
        samples.append(sample)
    
    # Save as WAV file
    with wave.open(output_path, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(struct.pack('h' * len(samples), *samples))
    
    print(f"Created test audio: {output_path}")

if __name__ == '__main__':
    # Create test_data directory if it doesn't exist
    import os
    os.makedirs('test_data', exist_ok=True)
    
    create_test_video()
    create_test_audio() 