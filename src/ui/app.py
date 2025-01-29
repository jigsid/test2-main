import gradio as gr
import os
import tempfile
from typing import Tuple, Optional
import logging
from pathlib import Path

# Import our video generation components
from video.generator import VideoGenerator
from audio.music_processor import MusicProcessor
from models.visual_generator import VisualGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoApp:
    def __init__(self):
        """Initialize the video generation application."""
        self.video_generator = VideoGenerator()
        self.music_processor = MusicProcessor()
        self.visual_generator = VisualGenerator()
        
        # Define supported themes and styles
        self.themes = ['realistic', 'animated', 'abstract', 'cinematic']
        self.text_positions = ['top', 'bottom', 'center']
        
    def create_ui(self) -> gr.Blocks:
        """Create the Gradio interface."""
        with gr.Blocks(title="AI Video Generator", theme=gr.themes.Soft()) as app:
            gr.Markdown("""
            # 🎬 AI Video Generator
            Create stunning music videos and TikTok-style content with AI-powered effects!
            """)
            
            with gr.Tabs():
                # Music Video Tab
                with gr.Tab("🎵 Music Video Generator"):
                    with gr.Row():
                        with gr.Column():
                            audio_input = gr.Audio(
                                label="Upload Music",
                                type="filepath"
                            )
                            theme_input = gr.Dropdown(
                                choices=self.themes,
                                value="realistic",
                                label="Visual Theme"
                            )
                            duration_input = gr.Slider(
                                minimum=5,
                                maximum=300,
                                value=60,
                                step=5,
                                label="Duration (seconds)"
                            )
                            generate_music_btn = gr.Button(
                                "🎨 Generate Music Video",
                                variant="primary"
                            )
                        
                        with gr.Column():
                            video_output = gr.Video(
                                label="Generated Video",
                                height=600
                            )
                            
                    generate_music_btn.click(
                        fn=self.generate_music_video,
                        inputs=[audio_input, theme_input, duration_input],
                        outputs=video_output
                    )
                
                # TikTok Video Tab
                with gr.Tab("📱 TikTok Video Generator"):
                    with gr.Row():
                        with gr.Column():
                            prompt_input = gr.Textbox(
                                label="Video Description",
                                placeholder="Describe your video concept...",
                                lines=3
                            )
                            style_input = gr.Dropdown(
                                choices=self.themes,
                                value="cinematic",
                                label="Visual Style"
                            )
                            text_position = gr.Dropdown(
                                choices=self.text_positions,
                                value="bottom",
                                label="Text Position"
                            )
                            tiktok_duration = gr.Slider(
                                minimum=15,
                                maximum=60,
                                value=30,
                                step=5,
                                label="Duration (seconds)"
                            )
                            generate_tiktok_btn = gr.Button(
                                "✨ Generate TikTok Video",
                                variant="primary"
                            )
                        
                        with gr.Column():
                            tiktok_output = gr.Video(
                                label="Generated Video",
                                height=600
                            )
                            hashtags_output = gr.Textbox(
                                label="Suggested Hashtags",
                                lines=2
                            )
                    
                    generate_tiktok_btn.click(
                        fn=self.generate_tiktok_video,
                        inputs=[
                            prompt_input,
                            style_input,
                            text_position,
                            tiktok_duration
                        ],
                        outputs=[tiktok_output, hashtags_output]
                    )
            
            # Add footer with instructions
            gr.Markdown("""
            ### 📝 Instructions
            1. **Music Video**: Upload your music file and select a visual theme
            2. **TikTok Video**: Enter a description and choose your style preferences
            3. Click generate and wait for the magic to happen!
            
            *Note: Video generation may take a few minutes depending on the duration and effects.*
            """)
        
        return app
    
    def generate_music_video(
        self,
        audio_file: str,
        theme: str,
        duration: float
    ) -> str:
        """Generate a music video from the provided audio."""
        try:
            if not audio_file:
                raise ValueError("Please upload an audio file")
            
            logger.info(f"Generating music video with theme: {theme}")
            
            # Process audio file
            audio_features = self.music_processor.process_audio_file(audio_file)
            
            # Generate video
            video_file = self.video_generator.generate_background(
                duration,
                theme,
                audio_features
            )
            
            # Combine audio and video
            final_video = self.video_generator.combine_with_audio(
                video_file,
                audio_file
            )
            
            return final_video
            
        except Exception as e:
            logger.error(f"Error generating music video: {str(e)}")
            raise gr.Error(str(e))
    
    def generate_tiktok_video(
        self,
        prompt: str,
        style: str,
        text_position: str,
        duration: int
    ) -> Tuple[str, str]:
        """Generate a TikTok-style video from the prompt."""
        try:
            if not prompt:
                raise ValueError("Please enter a video description")
            
            logger.info(f"Generating TikTok video with style: {style}")
            
            # Generate visuals
            image_file = self.visual_generator.generate_scene_visuals(
                {"description": prompt},
                style=style
            )
            
            # Add text overlay
            image_with_text = self.visual_generator.create_text_overlay(
                image_file,
                prompt,
                position=text_position
            )
            
            # Convert to video
            video_file = self._image_to_video(image_with_text, duration)
            
            # Generate hashtags (placeholder)
            hashtags = "#AIVideo #TikTok #Trending"
            
            return video_file, hashtags
            
        except Exception as e:
            logger.error(f"Error generating TikTok video: {str(e)}")
            raise gr.Error(str(e))
    
    def _image_to_video(self, image_path: str, duration: float) -> str:
        """Convert an image to a video clip."""
        try:
            # Create temporary file for video
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            
            # Use VideoGenerator to create video from image
            video = self.video_generator.apply_effects(
                image_path,
                [{"type": "color_grading", "params": {"intensity": 0.5}}]
            )
            
            return video
            
        except Exception as e:
            logger.error(f"Error converting image to video: {str(e)}")
            raise 