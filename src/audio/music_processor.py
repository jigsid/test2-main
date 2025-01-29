import librosa
import numpy as np
import soundfile as sf
import tempfile
import logging
from typing import Dict, Any, Tuple
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MusicProcessor:
    def __init__(self):
        """Initialize the music processor."""
        self.supported_formats = ['.mp3', '.wav', '.ogg', '.flac']
        self.sample_rate = 44100  # Standard sample rate
        
    def process_audio_file(self, audio_path: str) -> Dict[str, Any]:
        """Process an audio file and extract features for video generation.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            Dict: Dictionary containing audio features
        """
        try:
            # Validate audio file
            self._validate_audio_file(audio_path)
            
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract features
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            energy_profile = self._calculate_energy_profile(y)
            
            # Get beat times
            beat_times = librosa.frames_to_time(beats, sr=sr)
            
            # Calculate sync points (beat times with strength)
            sync_points = []
            for beat_time in beat_times:
                frame = librosa.time_to_frames(beat_time, sr=sr)
                strength = onset_env[min(frame, len(onset_env)-1)]
                sync_points.append((float(beat_time), float(strength)))
            
            return {
                'duration': float(len(y) / sr),
                'tempo': float(tempo),
                'beat_times': beat_times.tolist(),
                'energy_profile': energy_profile,
                'sync_points': sync_points
            }
            
        except Exception as e:
            logger.error(f"Error processing audio file: {str(e)}")
            raise
            
    def _validate_audio_file(self, audio_path: str) -> None:
        """Validate the audio file format and existence.
        
        Args:
            audio_path (str): Path to the audio file
            
        Raises:
            ValueError: If the file format is not supported or file doesn't exist
        """
        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")
            
        file_ext = os.path.splitext(audio_path)[1].lower()
        if file_ext not in self.supported_formats:
            raise ValueError(
                f"Unsupported audio format: {file_ext}. "
                f"Supported formats: {self.supported_formats}"
            )
            
    def _calculate_energy_profile(self, y: np.ndarray) -> np.ndarray:
        """Calculate the energy profile of the audio signal.
        
        Args:
            y (np.ndarray): Audio signal
            
        Returns:
            np.ndarray: Energy profile
        """
        # Calculate the RMS energy for each frame
        hop_length = 512
        frame_length = 2048
        
        # Compute RMS energy for each frame
        energy = librosa.feature.rms(
            y=y,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        # Normalize energy to [0, 1] range
        energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-6)
        
        return energy
        
    def trim_audio(self, audio_path: str,
                  start_time: float,
                  end_time: float) -> str:
        """Trim the audio file to specified duration.
        
        Args:
            audio_path (str): Path to the audio file
            start_time (float): Start time in seconds
            end_time (float): End time in seconds
            
        Returns:
            str: Path to the trimmed audio file
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Convert times to samples
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            # Trim audio
            y_trimmed = y[start_sample:end_sample]
            
            # Save trimmed audio
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            sf.write(temp_file.name, y_trimmed, sr)
            
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Error trimming audio: {str(e)}")
            raise
            
    def adjust_volume(self, audio_path: str, volume_factor: float) -> str:
        """Adjust the volume of the audio file.
        
        Args:
            audio_path (str): Path to the audio file
            volume_factor (float): Volume adjustment factor (1.0 = original volume)
            
        Returns:
            str: Path to the volume-adjusted audio file
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Adjust volume
            y_adjusted = y * volume_factor
            
            # Save adjusted audio
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            sf.write(temp_file.name, y_adjusted, sr)
            
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Error adjusting volume: {str(e)}")
            raise 