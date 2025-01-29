from typing import List, Dict, Optional, Tuple
import logging
import os
import requests
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import io
import tempfile
from dotenv import load_dotenv
import numpy as np
import cv2
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualGenerator:
    def __init__(self):
        """Initialize the visual generator with API configurations."""
        load_dotenv()
        self.api_key = os.getenv("HUGGINGFACE_API_KEY", "")  # Optional: for higher rate limits
        self.image_api_url = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
        self.supported_styles = ['realistic', 'animated', 'abstract', 'cinematic']
        self.supported_sizes = [(1080, 1920), (720, 1280)]  # Portrait sizes
        
    def generate_scene_visuals(self, scene: Dict[str, any], 
                             style: str = 'realistic',
                             size: Tuple[int, int] = (1080, 1920)) -> str:
        """Generate visuals for a scene using Stable Diffusion.
        
        Args:
            scene (Dict): Scene description and parameters
            style (str): Visual style to apply
            size (Tuple[int, int]): Output image size
            
        Returns:
            str: Path to the generated image file
        """
        try:
            if style not in self.supported_styles:
                raise ValueError(f"Unsupported style. Supported styles: {self.supported_styles}")
                
            # Create image prompt
            prompt = self._create_image_prompt(scene, style)
            
            # Generate image
            image_data = self._generate_image(prompt)
            
            # Resize image to desired size
            image_data = image_data.resize(size, Image.Resampling.LANCZOS)
            
            # Save image to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            image_data.save(temp_file.name)
            
            return temp_file.name
        except Exception as e:
            logger.error(f"Error generating scene visuals: {str(e)}")
            raise
            
    def _create_image_prompt(self, scene: Dict[str, any], style: str) -> str:
        """Create a detailed prompt for image generation.
        
        Args:
            scene (Dict): Scene description and parameters
            style (str): Visual style to apply
            
        Returns:
            str: Formatted image generation prompt
        """
        base_prompt = scene['description']
        
        # Add style-specific keywords
        style_keywords = {
            'realistic': 'highly detailed, photorealistic, 8k uhd, high quality',
            'animated': '3D animation style, pixar style, vibrant colors, smooth textures',
            'abstract': 'abstract art, geometric shapes, modern art style, minimalist',
            'cinematic': 'cinematic lighting, movie scene, dramatic atmosphere, professional photography'
        }
        
        # Add composition guidelines for vertical video
        composition = "vertical composition, portrait orientation, centered composition"
        
        # Add quality boosters
        quality = "masterpiece, best quality, highly detailed"
        
        # Combine prompts
        full_prompt = f"{base_prompt}, {style_keywords[style]}, {composition}, {quality}"
        
        return full_prompt
        
    def _generate_image(self, prompt: str) -> Image.Image:
        """Generate image using Stable Diffusion through Hugging Face.
        
        Args:
            prompt (str): Image generation prompt
            
        Returns:
            Image.Image: Generated image
        """
        try:
            headers = {
                "Content-Type": "application/json"
            }
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            data = {
                "inputs": prompt,
                "parameters": {
                    "negative_prompt": "blurry, bad quality, distorted, deformed, ugly, bad anatomy",
                    "num_inference_steps": 50,
                    "guidance_scale": 7.5
                }
            }
            
            response = requests.post(self.image_api_url, headers=headers, json=data)
            response.raise_for_status()
            
            # Convert response to image
            image = Image.open(io.BytesIO(response.content))
            
            return image
        except Exception as e:
            logger.error(f"Error in image generation API call: {str(e)}")
            raise
            
    def apply_style_transfer(self, image_path: str, style: str) -> str:
        """Apply style transfer to an image.
        
        Args:
            image_path (str): Path to the input image
            style (str): Style to apply
            
        Returns:
            str: Path to the styled image
        """
        try:
            # Load image
            image = Image.open(image_path)
            
            # Apply style-specific processing
            if style == 'realistic':
                styled_image = self._apply_realistic_style(image)
            elif style == 'animated':
                styled_image = self._apply_animated_style(image)
            elif style == 'abstract':
                styled_image = self._apply_abstract_style(image)
            elif style == 'cinematic':
                styled_image = self._apply_cinematic_style(image)
            else:
                raise ValueError(f"Unsupported style: {style}")
                
            # Save styled image
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            styled_image.save(temp_file.name)
            
            return temp_file.name
        except Exception as e:
            logger.error(f"Error applying style transfer: {str(e)}")
            raise
            
    def _apply_realistic_style(self, image: Image.Image) -> Image.Image:
        """Apply realistic style processing."""
        # Convert to numpy array for OpenCV processing
        img_array = np.array(image)
        
        # Apply various enhancements
        # 1. Increase contrast
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.2)
        
        # 2. Increase sharpness
        sharpness = ImageEnhance.Sharpness(image)
        image = sharpness.enhance(1.3)
        
        # 3. Adjust color
        color = ImageEnhance.Color(image)
        image = color.enhance(1.1)
        
        # 4. Apply subtle vignette
        img_array = np.array(image)
        rows, cols = img_array.shape[:2]
        
        # Generate vignette mask
        X_resultant, Y_resultant = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1, rows))
        mask = np.sqrt(X_resultant**2 + Y_resultant**2)
        mask = 1 - np.clip(mask, 0, 1)
        
        # Apply mask to each channel
        for i in range(3):
            img_array[:,:,i] = img_array[:,:,i] * mask
            
        return Image.fromarray(img_array)
        
    def _apply_animated_style(self, image: Image.Image) -> Image.Image:
        """Apply animated style processing."""
        # 1. Reduce color palette
        image = image.quantize(colors=32, method=2).convert('RGB')
        
        # 2. Apply edge enhancement
        edges = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        
        # 3. Increase saturation
        converter = ImageEnhance.Color(edges)
        image = converter.enhance(1.4)
        
        # 4. Smooth certain areas while keeping edges sharp
        img_array = np.array(image)
        
        # Apply bilateral filter for cartoon-like effect
        img_array = cv2.bilateralFilter(img_array, 9, 75, 75)
        
        # Find edges
        edges = cv2.adaptiveThreshold(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY),
                                    255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 9, 2)
        
        # Combine edges with filtered image
        img_array = cv2.bitwise_and(img_array, img_array, mask=edges)
        
        return Image.fromarray(img_array)
        
    def _apply_abstract_style(self, image: Image.Image) -> Image.Image:
        """Apply abstract style processing."""
        # 1. Create base canvas
        canvas = Image.new('RGB', image.size, (0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        
        # 2. Convert image to numpy array for processing
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # 3. Find edges and contours
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 4. Draw geometric shapes based on contours
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255)]
        
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small contours
                # Simplify contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Convert points to list for PIL
                points = [(p[0][0], p[0][1]) for p in approx]
                if len(points) >= 3:  # Need at least 3 points
                    color = colors[len(points) % len(colors)]
                    draw.polygon(points, outline=color)
        
        # 5. Apply geometric overlay
        overlay = Image.new('RGB', image.size, (0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Draw random geometric shapes
        for _ in range(50):
            shape = random.choice(['triangle', 'rectangle', 'circle'])
            color = colors[random.randint(0, len(colors)-1)]
            x = random.randint(0, image.size[0])
            y = random.randint(0, image.size[1])
            size = random.randint(20, 100)
            
            if shape == 'triangle':
                points = [(x, y-size), (x-size, y+size), (x+size, y+size)]
                draw.polygon(points, outline=color)
            elif shape == 'rectangle':
                draw.rectangle([x-size, y-size, x+size, y+size], outline=color)
            else:  # circle
                draw.ellipse([x-size, y-size, x+size, y+size], outline=color)
        
        # 6. Blend original with geometric overlay
        result = Image.blend(canvas, overlay, 0.5)
        return result
        
    def _apply_cinematic_style(self, image: Image.Image) -> Image.Image:
        """Apply cinematic style processing."""
        # 1. Adjust contrast and color
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.3)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.85)  # Slightly desaturate
        
        # 2. Apply color grading (cinematic teal and orange look)
        img_array = np.array(image)
        
        # Split into channels
        b, g, r = cv2.split(img_array)
        
        # Adjust shadows to be more teal
        shadows_mask = cv2.inRange(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY), 0, 100)
        b = cv2.add(b, 20, mask=shadows_mask)  # Add blue to shadows
        g = cv2.add(g, 10, mask=shadows_mask)  # Add some green to shadows
        
        # Adjust highlights to be more orange
        highlights_mask = cv2.inRange(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY), 150, 255)
        r = cv2.add(r, 20, mask=highlights_mask)  # Add red to highlights
        g = cv2.add(g, 10, mask=highlights_mask)  # Add some green to highlights
        
        img_array = cv2.merge([b, g, r])
        
        # 3. Add cinematic vignette
        rows, cols = img_array.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols/4)
        kernel_y = cv2.getGaussianKernel(rows, rows/4)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()
        
        for i in range(3):
            img_array[:,:,i] = img_array[:,:,i] * mask
        
        # 4. Add film grain
        grain = np.random.normal(0, 2, img_array.shape).astype(np.uint8)
        img_array = cv2.add(img_array, grain)
        
        # 5. Adjust gamma for cinematic look
        gamma = 1.1
        lookUpTable = np.empty((1,256), np.uint8)
        for i in range(256):
            lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        img_array = cv2.LUT(img_array, lookUpTable)
        
        return Image.fromarray(img_array)
        
    def create_text_overlay(self, image_path: str, text: str,
                          position: str = 'bottom') -> str:
        """Add text overlay to an image."""
        try:
            # Load image
            image = Image.open(image_path)
            
            # Create text overlay
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Load a font (you might want to specify a custom font file path)
            try:
                font = ImageFont.truetype("arial.ttf", 60)
            except:
                font = ImageFont.load_default()
            
            # Calculate text size and position
            text_width, text_height = draw.textsize(text, font=font)
            padding = 20
            
            if position == 'bottom':
                text_x = (image.width - text_width) // 2
                text_y = image.height - text_height - padding * 2
            elif position == 'top':
                text_x = (image.width - text_width) // 2
                text_y = padding
            else:  # center
                text_x = (image.width - text_width) // 2
                text_y = (image.height - text_height) // 2
            
            # Draw text background
            bg_bbox = [text_x - padding,
                      text_y - padding,
                      text_x + text_width + padding,
                      text_y + text_height + padding]
            draw.rectangle(bg_bbox, fill=(0, 0, 0, 128))
            
            # Draw text shadow
            shadow_offset = 2
            draw.text((text_x + shadow_offset, text_y + shadow_offset),
                     text, font=font, fill=(0, 0, 0, 200))
            
            # Draw main text
            draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255, 255))
            
            # Combine original image with overlay
            result = Image.alpha_composite(image.convert('RGBA'), overlay)
            
            # Save result
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            result.save(temp_file.name)
            
            return temp_file.name
        except Exception as e:
            logger.error(f"Error creating text overlay: {str(e)}")
            raise 