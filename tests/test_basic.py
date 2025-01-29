import unittest
import os
import sys
import tempfile
import numpy as np
from PIL import Image

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.audio.music_processor import MusicProcessor
from src.video.generator import VideoGenerator
from src.models.text_to_script import ScriptGenerator
from src.models.visual_generator import VisualGenerator

class TestBasicFunctionality(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.music_processor = MusicProcessor()
        self.video_generator = VideoGenerator()
        self.script_generator = ScriptGenerator()
        self.visual_generator = VisualGenerator()
        
        # Create test audio file
        self.test_audio = self._create_test_audio()
        
        # Create test image
        self.test_image = self._create_test_image()
        
    def tearDown(self):
        """Clean up test environment."""
        # Remove test files
        if hasattr(self, 'test_audio') and os.path.exists(self.test_audio):
            os.remove(self.test_audio)
        if hasattr(self, 'test_image') and os.path.exists(self.test_image):
            os.remove(self.test_image)
            
    def _create_test_audio(self):
        """Create a test audio file."""
        # Create a simple sine wave
        duration = 5  # seconds
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Save to WAV file
        import scipy.io.wavfile as wav
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        wav.write(temp_file.name, sample_rate, audio.astype(np.float32))
        
        return temp_file.name
        
    def _create_test_image(self):
        """Create a test image."""
        # Create a simple gradient image
        width, height = 1080, 1920
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                image[y, x] = [
                    int(255 * x / width),
                    int(255 * y / height),
                    128
                ]
                
        # Save to PNG file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        Image.fromarray(image).save(temp_file.name)
        
        return temp_file.name
        
    def test_audio_processing(self):
        """Test basic audio processing functionality."""
        # Test audio loading
        try:
            audio_features = self.music_processor.process_audio_file(self.test_audio)
            self.assertIsNotNone(audio_features)
            self.assertIn('beat_info', audio_features)
            self.assertIn('rhythm_features', audio_features)
        except Exception as e:
            self.fail(f"Audio processing failed: {str(e)}")
            
    def test_video_generation(self):
        """Test basic video generation functionality."""
        try:
            # Test background generation
            video_file = self.video_generator.generate_background(
                duration=5,
                theme='abstract',
                audio_features={
                    'beat_info': {'tempo': 120},
                    'energy_profile': np.random.random(100)
                }
            )
            self.assertTrue(os.path.exists(video_file))
            
            # Clean up
            os.remove(video_file)
        except Exception as e:
            self.fail(f"Video generation failed: {str(e)}")
            
    def test_script_generation(self):
        """Test script generation functionality."""
        try:
            script_data = self.script_generator.generate_script(
                "Create a travel vlog about Paris",
                template='travel_vlog',
                duration=60
            )
            self.assertIsNotNone(script_data)
            self.assertIn('script', script_data)
            self.assertIn('metadata', script_data)
        except Exception as e:
            self.fail(f"Script generation failed: {str(e)}")
            
    def test_visual_generation(self):
        """Test visual generation functionality."""
        try:
            scene = {
                'description': 'A beautiful sunset over Paris',
                'duration': 5
            }
            
            image_file = self.visual_generator.generate_scene_visuals(
                scene,
                style='realistic'
            )
            self.assertTrue(os.path.exists(image_file))
            
            # Clean up
            os.remove(image_file)
        except Exception as e:
            self.fail(f"Visual generation failed: {str(e)}")
            
if __name__ == '__main__':
    unittest.main() 