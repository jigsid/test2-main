import cv2
import numpy as np
from typing import List, Tuple, Dict
import logging
from moviepy.editor import VideoClip, ImageClip
from moviepy.video.fx.all import fadein, fadeout, resize
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParticleSystem:
    def __init__(self, width: int, height: int, num_particles: int = 100):
        """Initialize particle system.
        
        Args:
            width (int): Video width
            height (int): Video height
            num_particles (int): Number of particles
        """
        self.width = width
        self.height = height
        self.num_particles = num_particles
        self.particles = self._initialize_particles()
        
    def _initialize_particles(self) -> List[Dict]:
        """Initialize particle positions and properties.
        
        Returns:
            List[Dict]: List of particle properties
        """
        particles = []
        for _ in range(self.num_particles):
            particle = {
                'x': np.random.randint(0, self.width),
                'y': np.random.randint(0, self.height),
                'size': np.random.randint(2, 10),
                'speed': np.random.random() * 2,
                'angle': np.random.random() * 2 * np.pi,
                'color': np.random.randint(0, 255, 3)
            }
            particles.append(particle)
        return particles
        
    def update(self, energy: float):
        """Update particle positions and properties based on audio energy.
        
        Args:
            energy (float): Current audio energy level (0-1)
        """
        for particle in self.particles:
            # Update position
            particle['x'] += math.cos(particle['angle']) * particle['speed'] * (1 + energy * 2)
            particle['y'] += math.sin(particle['angle']) * particle['speed'] * (1 + energy * 2)
            
            # Update size based on energy
            particle['size'] = max(2, min(10, particle['size'] * (1 + energy)))
            
            # Wrap around screen
            particle['x'] = particle['x'] % self.width
            particle['y'] = particle['y'] % self.height
            
    def render(self) -> np.ndarray:
        """Render particles to frame.
        
        Returns:
            np.ndarray: Frame with rendered particles
        """
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        for particle in self.particles:
            cv2.circle(frame,
                      (int(particle['x']), int(particle['y'])),
                      int(particle['size']),
                      particle['color'].tolist(),
                      -1)
            
        return frame

class GeometricPatterns:
    def __init__(self, width: int, height: int):
        """Initialize geometric pattern generator.
        
        Args:
            width (int): Video width
            height (int): Video height
        """
        self.width = width
        self.height = height
        
    def generate_pattern(self, pattern_type: str, energy: float) -> np.ndarray:
        """Generate geometric pattern based on type and energy.
        
        Args:
            pattern_type (str): Type of pattern to generate
            energy (float): Current audio energy level (0-1)
            
        Returns:
            np.ndarray: Frame with generated pattern
        """
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        if pattern_type == 'circles':
            frame = self._generate_circles(frame, energy)
        elif pattern_type == 'lines':
            frame = self._generate_lines(frame, energy)
        elif pattern_type == 'waves':
            frame = self._generate_waves(frame, energy)
            
        return frame
        
    def _generate_circles(self, frame: np.ndarray, energy: float) -> np.ndarray:
        """Generate concentric circles pattern.
        
        Args:
            frame (np.ndarray): Base frame
            energy (float): Current audio energy level
            
        Returns:
            np.ndarray: Frame with circle pattern
        """
        center = (self.width // 2, self.height // 2)
        max_radius = min(self.width, self.height) // 2
        num_circles = int(10 * (1 + energy))
        
        for i in range(num_circles):
            radius = int((i + 1) * max_radius / num_circles)
            color = (int(255 * energy), int(128 * energy), int(192 * energy))
            cv2.circle(frame, center, radius, color, 2)
            
        return frame
        
    def _generate_lines(self, frame: np.ndarray, energy: float) -> np.ndarray:
        """Generate dynamic line pattern.
        
        Args:
            frame (np.ndarray): Base frame
            energy (float): Current audio energy level
            
        Returns:
            np.ndarray: Frame with line pattern
        """
        num_lines = int(20 * (1 + energy))
        
        for i in range(num_lines):
            start_point = (
                np.random.randint(0, self.width),
                np.random.randint(0, self.height)
            )
            end_point = (
                np.random.randint(0, self.width),
                np.random.randint(0, self.height)
            )
            color = (int(192 * energy), int(255 * energy), int(128 * energy))
            cv2.line(frame, start_point, end_point, color, 2)
            
        return frame
        
    def _generate_waves(self, frame: np.ndarray, energy: float) -> np.ndarray:
        """Generate wave pattern.
        
        Args:
            frame (np.ndarray): Base frame
            energy (float): Current audio energy level
            
        Returns:
            np.ndarray: Frame with wave pattern
        """
        amplitude = int(50 * (1 + energy))
        frequency = 2 * np.pi / self.width
        phase = energy * 2 * np.pi
        
        points = []
        for x in range(self.width):
            y = int(self.height/2 + amplitude * np.sin(frequency * x + phase))
            points.append((x, y))
            
        # Draw wave
        for i in range(len(points) - 1):
            color = (int(128 * energy), int(192 * energy), int(255 * energy))
            cv2.line(frame, points[i], points[i+1], color, 2)
            
        return frame

class LightEffects:
    def __init__(self, width: int, height: int):
        """Initialize light effects generator.
        
        Args:
            width (int): Video width
            height (int): Video height
        """
        self.width = width
        self.height = height
        
    def generate_light_flare(self, position: Tuple[int, int], 
                           intensity: float) -> np.ndarray:
        """Generate light flare effect.
        
        Args:
            position (Tuple[int, int]): Position of the flare
            intensity (float): Intensity of the flare (0-1)
            
        Returns:
            np.ndarray: Frame with light flare
        """
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Create radial gradient
        y, x = np.ogrid[-position[1]:self.height-position[1],
                       -position[0]:self.width-position[0]]
        mask = x*x + y*y <= (100 * intensity)**2
        frame[..., 0][mask] = int(255 * intensity)  # Blue channel
        frame[..., 1][mask] = int(255 * intensity)  # Green channel
        frame[..., 2][mask] = int(255 * intensity)  # Red channel
        
        # Apply gaussian blur
        frame = cv2.GaussianBlur(frame, (0, 0), 10 * intensity)
        
        return frame
        
    def generate_bokeh(self, num_points: int, intensity: float) -> np.ndarray:
        """Generate bokeh effect.
        
        Args:
            num_points (int): Number of bokeh points
            intensity (float): Intensity of the effect (0-1)
            
        Returns:
            np.ndarray: Frame with bokeh effect
        """
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        for _ in range(num_points):
            position = (
                np.random.randint(0, self.width),
                np.random.randint(0, self.height)
            )
            size = np.random.randint(10, 30)
            color = (
                np.random.randint(128, 255),
                np.random.randint(128, 255),
                np.random.randint(128, 255)
            )
            
            # Draw bokeh circle
            cv2.circle(frame, position, size, color, -1)
            
        # Apply gaussian blur
        frame = cv2.GaussianBlur(frame, (0, 0), 5 * intensity)
        
        return frame

def create_transition(width: int, height: int, duration: float, 
                     transition_type: str) -> VideoClip:
    """Create a transition effect.
    
    Args:
        width (int): Video width
        height (int): Video height
        duration (float): Duration of transition
        transition_type (str): Type of transition
        
    Returns:
        VideoClip: Transition video clip
    """
    if transition_type == 'fade':
        # Create black frame
        black_frame = np.zeros((height, width, 3), dtype=np.uint8)
        clip = ImageClip(black_frame, duration=duration)
        clip = fadein(clip, duration/2)
        clip = fadeout(clip, duration/2)
        return clip
    elif transition_type == 'wipe':
        # Create wipe transition
        def make_frame(t):
            progress = t / duration
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :int(width * progress)] = 255
            return frame
            
        return VideoClip(make_frame, duration=duration)
    else:
        raise ValueError(f"Unsupported transition type: {transition_type}")

def apply_color_grading(frame: np.ndarray, 
                       params: Dict[str, float]) -> np.ndarray:
    """Apply color grading to a frame.
    
    Args:
        frame (np.ndarray): Input frame
        params (Dict): Color grading parameters
        
    Returns:
        np.ndarray: Color graded frame
    """
    # Convert to float32 for processing
    frame = frame.astype(np.float32) / 255.0
    
    # Apply contrast
    if 'contrast' in params:
        frame = frame * params['contrast']
        
    # Apply brightness
    if 'brightness' in params:
        frame = frame + params['brightness']
        
    # Apply saturation
    if 'saturation' in params:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = hsv[..., 1] * params['saturation']
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
    # Clip values to [0, 1] range
    frame = np.clip(frame, 0, 1)
    
    # Convert back to uint8
    return (frame * 255).astype(np.uint8) 