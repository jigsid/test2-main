import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, ColorClip, ImageClip, VideoClip
import tempfile
import os
from PIL import Image, ImageDraw, ImageFilter
import random
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoGenerator:
    def __init__(self, width: int = 1080, height: int = 1920):
        """Initialize the video generator with specified dimensions.
        
        Args:
            width (int): Width of the output video
            height (int): Height of the output video
        """
        self.width = width
        self.height = height
        self.fps = 30
        self.supported_themes = ['realistic', 'animated', 'abstract', 'cinematic']
        
    def generate_background(self, duration: float, theme: str,
                          audio_features: Dict[str, any]) -> str:
        """Generate a dynamic background video based on audio features.
        
        Args:
            duration (float): Duration of the video in seconds
            theme (str): Visual theme for the background
            audio_features (Dict): Dictionary containing audio analysis results
            
        Returns:
            str: Path to the generated video file
        """
        try:
            if theme.lower() not in self.supported_themes:
                raise ValueError(f"Unsupported theme. Supported themes: {self.supported_themes}")
            
            # Create base video clip with optimized settings
            base_clip = ColorClip((self.width, self.height), 
                                color=(0, 0, 0),
                                duration=duration)
            
            # Generate effects based on audio features
            effects = self._generate_effects(audio_features, theme)
            
            # Combine base clip with effects
            final_clip = CompositeVideoClip([base_clip] + effects)
            
            # Export video with optimized settings
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            logger.info(f"Generating video to {temp_file.name}")
            
            try:
                final_clip.write_videofile(
                    temp_file.name, 
                    fps=self.fps,
                    codec='libx264',
                    audio=False,
                    preset='ultrafast',
                    threads=4,
                    bitrate='2000k',
                    logger=None  # Disable progress bar
                )
            finally:
                # Clean up clips
                final_clip.close()
                base_clip.close()
                for effect in effects:
                    if hasattr(effect, 'close'):
                        effect.close()
            
            return temp_file.name
        except Exception as e:
            logger.error(f"Error generating background video: {str(e)}")
            raise
            
    def _generate_effects(self, audio_features: Dict[str, any], 
                         theme: str) -> List[VideoFileClip]:
        """Generate visual effects based on audio features and theme.
        
        Args:
            audio_features (Dict): Dictionary containing audio analysis results
            theme (str): Visual theme for the effects
            
        Returns:
            List[VideoFileClip]: List of effect clips
        """
        effects = []
        
        # Get sync points and energy profile
        sync_points = audio_features['sync_points']
        energy_profile = audio_features['energy_profile']
        
        if theme.lower() == 'abstract':
            effects.extend(self._generate_abstract_effects(sync_points, energy_profile))
        elif theme.lower() == 'realistic':
            effects.extend(self._generate_realistic_effects(sync_points, energy_profile))
        elif theme.lower() == 'animated':
            effects.extend(self._generate_animated_effects(sync_points, energy_profile))
        elif theme.lower() == 'cinematic':
            effects.extend(self._generate_cinematic_effects(sync_points, energy_profile))
            
        return effects
        
    def _generate_abstract_effects(self, sync_points: List[Tuple[float, float]],
                                 energy_profile: np.ndarray) -> List[VideoFileClip]:
        """Generate abstract visual effects (e.g., geometric patterns, particles).
        
        Args:
            sync_points (List[Tuple]): List of (time, strength) tuples
            energy_profile (np.ndarray): Audio energy profile
            
        Returns:
            List[VideoFileClip]: List of abstract effect clips
        """
        effects = []
        
        # Generate particle system
        particles = self._create_particle_system(sync_points, energy_profile)
        effects.append(particles)
        
        # Generate geometric patterns
        patterns = self._create_geometric_patterns(sync_points, energy_profile)
        effects.append(patterns)
        
        return effects
        
    def _generate_realistic_effects(self, sync_points: List[Tuple[float, float]],
                                  energy_profile: np.ndarray) -> List[VideoFileClip]:
        """Generate realistic visual effects (e.g., light flares, bokeh).
        
        Args:
            sync_points (List[Tuple]): List of (time, strength) tuples
            energy_profile (np.ndarray): Audio energy profile
            
        Returns:
            List[VideoFileClip]: List of realistic effect clips
        """
        effects = []
        
        # Generate light flares
        flares = self._create_light_flares(sync_points)
        effects.append(flares)
        
        # Generate bokeh effects
        bokeh = self._create_bokeh_effects(energy_profile)
        effects.append(bokeh)
        
        return effects
        
    def _create_particle_system(self, sync_points: List[Tuple[float, float]],
                              energy_profile: np.ndarray) -> VideoFileClip:
        """Create a particle system animation synchronized with the music."""
        duration = len(energy_profile) / self.fps
        num_particles = 50  # Reduced number of particles
        
        def make_frame(t):
            frame = np.zeros((self.height, self.width, 4), dtype=np.uint8)
            frame_idx = int(t * self.fps)
            energy = energy_profile[min(frame_idx, len(energy_profile) - 1)]
            
            # Generate particles for this frame
            for _ in range(num_particles):
                x = random.randint(0, self.width)
                y = random.randint(0, self.height)
                size = int(random.randint(2, 8) * (1 + energy))
                
                # Draw particle
                cv2.circle(frame, 
                         (int(x), int(y)), 
                         size,
                         (255, 255, 255, int(255 * energy)),
                         -1)
            
            return frame

        # Create clip directly from frame generator
        clip = VideoClip(make_frame, duration=duration)
        clip = clip.set_position(('center', 'center'))
        return clip

    def _create_geometric_patterns(self, sync_points: List[Tuple[float, float]],
                                 energy_profile: np.ndarray) -> VideoFileClip:
        """Create geometric pattern animations synchronized with the music."""
        duration = len(energy_profile) / self.fps
        pattern_types = ['circles', 'triangles', 'squares']
        
        def make_frame(t):
            frame = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(frame)
            
            frame_idx = int(t * self.fps)
            energy = energy_profile[min(frame_idx, len(energy_profile) - 1)]
            
            # Determine pattern type based on time
            current_pattern = int((t / duration) * len(pattern_types)) % len(pattern_types)
            
            # Draw patterns based on energy
            if pattern_types[current_pattern] == 'circles':
                num_shapes = int(5 * energy)  # Reduced number of shapes
                for _ in range(num_shapes):
                    size = random.randint(50, 200)
                    x = random.randint(-size, self.width + size)
                    y = random.randint(-size, self.height + size)
                    draw.ellipse([x-size, y-size, x+size, y+size],
                               outline=(255, 255, 255, int(255 * energy)))
            
            elif pattern_types[current_pattern] == 'triangles':
                num_shapes = int(4 * energy)  # Reduced number of shapes
                for _ in range(num_shapes):
                    size = random.randint(50, 150)
                    x = random.randint(0, self.width)
                    y = random.randint(0, self.height)
                    points = [
                        (x, y-size),
                        (x-size, y+size),
                        (x+size, y+size)
                    ]
                    draw.polygon(points, outline=(255, 255, 255, int(255 * energy)))
            
            else:  # squares
                num_shapes = int(3 * energy)  # Reduced number of shapes
                for _ in range(num_shapes):
                    size = random.randint(40, 120)
                    x = random.randint(-size, self.width + size)
                    y = random.randint(-size, self.height + size)
                    draw.rectangle([x-size, y-size, x+size, y+size],
                                 outline=(255, 255, 255, int(255 * energy)))
            
            return np.array(frame)

        # Create clip directly from frame generator
        clip = VideoClip(make_frame, duration=duration)
        clip = clip.set_position(('center', 'center'))
        return clip

    def _create_light_flares(self, sync_points: List[Tuple[float, float]]) -> VideoFileClip:
        """Create light flare effects synchronized with strong beats."""
        duration = sync_points[-1][0] if sync_points else 5.0
        
        def make_frame(t):
            frame = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
            
            # Check for nearby sync points
            for time, strength in sync_points:
                time_diff = abs(t - time)
                if time_diff < 0.1:  # Within 100ms of beat
                    # Create light flare
                    flare_size = int(300 * strength)  # Reduced size
                    x = random.randint(0, self.width)
                    y = random.randint(0, self.height)
                    
                    # Create simplified radial gradient
                    for radius in range(flare_size, 0, -4):  # Larger step size
                        opacity = int(255 * (1 - time_diff*10) * (radius/flare_size) * strength)
                        ImageDraw.Draw(frame).ellipse(
                            [x-radius, y-radius, x+radius, y+radius],
                            fill=(255, 255, 255, opacity)
                        )
                    
                    # Add simplified lens streaks
                    for angle in range(0, 360, 90):  # Reduced number of streaks
                        streak_length = random.randint(50, 150)  # Reduced length
                        end_x = x + streak_length * math.cos(math.radians(angle))
                        end_y = y + streak_length * math.sin(math.radians(angle))
                        ImageDraw.Draw(frame).line(
                            [x, y, end_x, end_y],
                            fill=(255, 255, 255, int(127 * strength)),
                            width=2
                        )
            
            # Apply minimal blur
            frame = frame.filter(ImageFilter.GaussianBlur(radius=3))
            return np.array(frame)
        
        # Create clip directly from frame generator
        clip = VideoClip(make_frame, duration=duration)
        clip = clip.set_position(('center', 'center'))
        return clip

    def _create_bokeh_effects(self, energy_profile: np.ndarray) -> VideoFileClip:
        """Create bokeh effects that respond to audio energy."""
        duration = len(energy_profile) / self.fps
        num_bokeh = 15  # Reduced number of bokeh points
        
        def make_frame(t):
            frame = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
            frame_idx = int(t * self.fps)
            energy = energy_profile[min(frame_idx, len(energy_profile) - 1)]
            
            # Generate bokeh points for this frame
            for _ in range(num_bokeh):
                x = random.randint(0, self.width)
                y = random.randint(0, self.height)
                size = int(random.randint(20, 60) * (0.5 + energy))  # Reduced size
                color = (
                    random.randint(200, 255),
                    random.randint(200, 255),
                    random.randint(200, 255)
                )
                
                # Create simplified bokeh circle
                bokeh = Image.new('RGBA', (size*2, size*2), (0, 0, 0, 0))
                draw = ImageDraw.Draw(bokeh)
                
                # Draw simplified gradient circle
                for radius in range(size, 0, -2):  # Larger step size
                    opacity = int(255 * (radius/size) * 0.5)
                    color_with_alpha = tuple(list(color) + [opacity])
                    draw.ellipse([size-radius, size-radius,
                                size+radius, size+radius],
                               fill=color_with_alpha)
                
                # Apply minimal blur
                bokeh = bokeh.filter(ImageFilter.GaussianBlur(radius=2))
                
                # Paste bokeh onto frame
                frame.paste(bokeh, (int(x-size), int(y-size)), bokeh)
            
            return np.array(frame)
        
        # Create clip directly from frame generator
        clip = VideoClip(make_frame, duration=duration)
        clip = clip.set_position(('center', 'center'))
        return clip
        
    def combine_with_audio(self, video_path: str, audio_path: str) -> str:
        """Combine video with audio track.
        
        Args:
            video_path (str): Path to the video file
            audio_path (str): Path to the audio file
            
        Returns:
            str: Path to the combined video file
        """
        try:
            # Load video and audio with optimized settings
            video = VideoFileClip(video_path)
            audio = AudioFileClip(audio_path)
            
            # Set audio
            final_video = video.set_audio(audio)
            
            # Export combined video with optimized settings
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            logger.info(f"Combining video and audio to {temp_file.name}")
            
            try:
                final_video.write_videofile(
                    temp_file.name,
                    fps=self.fps,
                    codec='libx264',
                    preset='ultrafast',
                    threads=4,
                    bitrate='2000k',
                    logger=None,  # Disable progress bar
                    ffmpeg_params=['-movflags', 'faststart']  # Optimize for streaming
                )
            finally:
                # Clean up
                video.close()
                audio.close()
                if hasattr(final_video, 'close'):
                    final_video.close()
            
            # Remove temporary input video
            if os.path.exists(video_path):
                try:
                    os.unlink(video_path)
                except:
                    pass
            
            return temp_file.name
        except Exception as e:
            logger.error(f"Error combining video and audio: {str(e)}")
            raise
            
    def apply_effects(self, video_path: str, effects: List[Dict[str, any]]) -> str:
        """Apply post-processing effects to the video.
        
        Args:
            video_path (str): Path to the video file
            effects (List[Dict]): List of effect configurations
            
        Returns:
            str: Path to the processed video file
        """
        try:
            # Load video
            video = VideoFileClip(video_path)
            
            # Apply each effect
            for effect in effects:
                if effect['type'] == 'color_grading':
                    video = self._apply_color_grading(video, effect['params'])
                elif effect['type'] == 'blur':
                    video = self._apply_blur(video, effect['params'])
                # Add more effect types as needed
            
            # Export processed video
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            video.write_videofile(temp_file.name,
                                fps=self.fps,
                                codec='libx264')
            
            return temp_file.name
        except Exception as e:
            logger.error(f"Error applying effects: {str(e)}")
            raise
            
    def _apply_color_grading(self, video: VideoFileClip, 
                           params: Dict[str, float]) -> VideoFileClip:
        """Apply color grading to the video.
        
        Args:
            video (VideoFileClip): Input video
            params (Dict): Color grading parameters
            
        Returns:
            VideoFileClip: Color graded video
        """
        try:
            # Get intensity parameter
            intensity = params.get('intensity', 0.5)
            
            def color_grade_frame(frame):
                # Convert to float for processing
                frame = frame.astype(float)
                
                # Adjust contrast
                frame = frame * (1 + 0.2 * intensity)
                
                # Adjust shadows (make them cooler/bluer)
                shadows_mask = frame < 128
                frame[shadows_mask] = frame[shadows_mask] * [0.9, 0.95, 1.1]  # Blue tint
                
                # Adjust highlights (make them warmer/orange)
                highlights_mask = frame >= 128
                frame[highlights_mask] = frame[highlights_mask] * [1.1, 1.0, 0.9]  # Orange tint
                
                # Clip values to valid range
                frame = np.clip(frame, 0, 255)
                
                return frame.astype(np.uint8)
            
            # Apply color grading to each frame
            return video.fl_image(color_grade_frame)
            
        except Exception as e:
            logger.error(f"Error applying color grading: {str(e)}")
            raise
            
    def _apply_blur(self, video: VideoFileClip, 
                   params: Dict[str, float]) -> VideoFileClip:
        """Apply blur effect to the video.
        
        Args:
            video (VideoFileClip): Input video
            params (Dict): Blur parameters
            
        Returns:
            VideoFileClip: Blurred video
        """
        try:
            # Get blur parameters
            radius = int(params.get('radius', 5))
            
            def blur_frame(frame):
                # Apply Gaussian blur
                blurred = cv2.GaussianBlur(frame, (radius*2+1, radius*2+1), 0)
                return blurred
            
            # Apply blur to each frame
            return video.fl_image(blur_frame)
            
        except Exception as e:
            logger.error(f"Error applying blur: {str(e)}")
            raise 