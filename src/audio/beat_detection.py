import librosa
import numpy as np
from typing import Tuple, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BeatDetector:
    def __init__(self, sample_rate: int = 22050):
        """Initialize the beat detector.
        
        Args:
            sample_rate (int): Sample rate for audio processing
        """
        self.sample_rate = sample_rate
        
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, float]:
        """Load audio file and return the time series and sample rate.
        
        Args:
            file_path (str): Path to the audio file
            
        Returns:
            Tuple containing the audio time series and sample rate
        """
        try:
            y, sr = librosa.load(file_path, sr=self.sample_rate)
            return y, sr
        except Exception as e:
            logger.error(f"Error loading audio file: {str(e)}")
            raise
            
    def detect_beats(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Detect beats and rhythm features in the audio.
        
        Args:
            y (np.ndarray): Audio time series
            sr (int): Sample rate
            
        Returns:
            Dictionary containing beat frames, tempo, and onset strength
        """
        try:
            # Get onset strength
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            
            # Detect tempo and beat frames
            tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            
            # Convert beat frames to time
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            
            # Get onset strength at beat times
            beat_strengths = onset_env[beat_frames]
            
            return {
                'beat_frames': beat_frames,
                'beat_times': beat_times,
                'beat_strengths': beat_strengths,
                'tempo': tempo,
                'onset_envelope': onset_env
            }
        except Exception as e:
            logger.error(f"Error detecting beats: {str(e)}")
            raise
            
    def analyze_rhythm_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract additional rhythm features for enhanced synchronization.
        
        Args:
            y (np.ndarray): Audio time series
            sr (int): Sample rate
            
        Returns:
            Dictionary containing rhythm features
        """
        try:
            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
            
            # Compute tempogram
            tempogram = librosa.feature.tempogram(onset_envelope=librosa.onset.onset_strength(y=y, sr=sr), sr=sr)
            
            # Compute spectral novelty
            spectral_novelty = librosa.onset.onset_strength(y=y, sr=sr)
            
            return {
                'mel_spectrogram': mel_spec,
                'tempogram': tempogram,
                'spectral_novelty': spectral_novelty
            }
        except Exception as e:
            logger.error(f"Error analyzing rhythm features: {str(e)}")
            raise
            
    def get_sync_points(self, beat_times: np.ndarray, beat_strengths: np.ndarray, 
                       threshold: float = 0.5) -> List[Tuple[float, float]]:
        """Generate synchronization points for video effects.
        
        Args:
            beat_times (np.ndarray): Array of beat times
            beat_strengths (np.ndarray): Array of beat strengths
            threshold (float): Threshold for beat strength
            
        Returns:
            List of tuples containing (time, strength) for sync points
        """
        # Normalize beat strengths
        normalized_strengths = (beat_strengths - beat_strengths.min()) / (beat_strengths.max() - beat_strengths.min())
        
        # Filter strong beats
        sync_points = [(time, strength) for time, strength in zip(beat_times, normalized_strengths)
                      if strength > threshold]
        
        return sync_points

    def get_sections(self, y: np.ndarray, sr: int) -> List[Dict[str, float]]:
        """Detect musical sections for video scene transitions.
        
        Args:
            y (np.ndarray): Audio time series
            sr (int): Sample rate
            
        Returns:
            List of dictionaries containing section boundaries and characteristics
        """
        # Compute harmonic and percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Compute chromagram
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
        
        # Detect section boundaries
        boundaries = librosa.segment.detect_onsets(librosa.onset.onset_strength(y=y, sr=sr))
        boundary_times = librosa.frames_to_time(boundaries, sr=sr)
        
        sections = []
        for i in range(len(boundary_times) - 1):
            sections.append({
                'start': boundary_times[i],
                'end': boundary_times[i + 1],
                'duration': boundary_times[i + 1] - boundary_times[i]
            })
            
        return sections 