from typing import List, Dict, Optional
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import json
import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScriptGenerator:
    def __init__(self, model_name: str = "facebook/opt-350m"):
        """Initialize the script generator.
        
        Args:
            model_name (str): Name of the language model to use
        """
        load_dotenv()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device="cuda" if os.getenv("USE_GPU", "false").lower() == "true" else "cpu"
        )
        
        self.scene_templates = {
            'travel_vlog': {
                'structure': [
                    {'type': 'intro', 'duration': 5},
                    {'type': 'location_shots', 'duration': 15},
                    {'type': 'highlights', 'duration': 25},
                    {'type': 'outro', 'duration': 5}
                ],
                'style': 'energetic, fast-paced'
            },
            'product_demo': {
                'structure': [
                    {'type': 'problem_statement', 'duration': 5},
                    {'type': 'product_intro', 'duration': 10},
                    {'type': 'features', 'duration': 30},
                    {'type': 'call_to_action', 'duration': 5}
                ],
                'style': 'professional, clear'
            }
        }
        
    def generate_script(self, prompt: str, template: str = None,
                       duration: int = 60) -> Dict[str, any]:
        """Generate a video script from a text prompt.
        
        Args:
            prompt (str): Text prompt describing the video
            template (str): Optional template name to use
            duration (int): Target video duration in seconds
            
        Returns:
            Dict containing the generated script and metadata
        """
        try:
            # Generate base script
            script_prompt = self._create_script_prompt(prompt, template, duration)
            
            # Generate script using language model
            response = self.generator(
                script_prompt,
                max_length=1000,
                num_return_sequences=1,
                temperature=0.7
            )[0]['generated_text']
            
            # Parse and structure the response
            script = self._parse_script(response, template)
            
            # Add metadata
            metadata = {
                'duration': duration,
                'template': template,
                'original_prompt': prompt
            }
            
            return {
                'script': script,
                'metadata': metadata
            }
        except Exception as e:
            logger.error(f"Error generating script: {str(e)}")
            raise
            
    def _create_script_prompt(self, prompt: str, template: str,
                            duration: int) -> str:
        """Create a detailed prompt for the language model.
        
        Args:
            prompt (str): User's text prompt
            template (str): Template name
            duration (int): Target duration
            
        Returns:
            str: Formatted prompt for the language model
        """
        base_prompt = f"""Create a detailed TikTok-style video script for the following concept:
{prompt}

Target duration: {duration} seconds
Style: Engaging, dynamic, and suitable for vertical video format

Please include:
1. Scene descriptions
2. Narration text
3. Visual effects suggestions
4. Music mood recommendations
5. Transition types between scenes"""

        if template and template in self.scene_templates:
            template_info = self.scene_templates[template]
            base_prompt += f"\n\nFollow this structure:\n"
            for scene in template_info['structure']:
                base_prompt += f"- {scene['type'].replace('_', ' ').title()}: {scene['duration']}s\n"
            base_prompt += f"\nStyle: {template_info['style']}"
            
        return base_prompt
        
    def _parse_script(self, response: str, template: str = None) -> List[Dict[str, any]]:
        """Parse the language model response into structured scenes.
        
        Args:
            response (str): Raw response from language model
            template (str): Template name used
            
        Returns:
            List[Dict]: List of scene dictionaries
        """
        scenes = []
        current_scene = None
        
        # Split response into lines
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for scene markers
            if line.lower().startswith(('scene', 'shot', 'segment')):
                if current_scene:
                    scenes.append(current_scene)
                current_scene = {
                    'description': line,
                    'narration': '',
                    'effects': [],
                    'transitions': [],
                    'duration': 5  # Default duration
                }
            elif current_scene:
                # Add details to current scene
                if line.lower().startswith('narration:'):
                    current_scene['narration'] = line.split(':', 1)[1].strip()
                elif line.lower().startswith('effect:'):
                    current_scene['effects'].append(line.split(':', 1)[1].strip())
                elif line.lower().startswith('transition:'):
                    current_scene['transitions'].append(line.split(':', 1)[1].strip())
                elif line.lower().startswith('duration:'):
                    try:
                        current_scene['duration'] = int(line.split(':', 1)[1].strip().split('s')[0])
                    except ValueError:
                        pass
                        
        # Add last scene
        if current_scene:
            scenes.append(current_scene)
            
        return scenes
        
    def suggest_hashtags(self, script: Dict[str, any]) -> List[str]:
        """Generate relevant hashtags based on the script content.
        
        Args:
            script (Dict): Generated script dictionary
            
        Returns:
            List[str]: List of relevant hashtags
        """
        try:
            # Combine all text content
            content = script['metadata']['original_prompt'] + ' '
            for scene in script['script']:
                content += scene['description'] + ' ' + scene['narration'] + ' '
                
            # Generate hashtag prompt
            hashtag_prompt = f"""Generate 5-7 relevant TikTok hashtags for the following content:
{content}

Format: #hashtag1 #hashtag2 #hashtag3"""
            
            # Generate hashtags
            response = self.generator(
                hashtag_prompt,
                max_length=100,
                num_return_sequences=1,
                temperature=0.7
            )[0]['generated_text']
            
            # Extract hashtags
            hashtags = [tag.strip() for tag in response.split() if tag.startswith('#')]
            
            return hashtags[:7]  # Limit to 7 hashtags
        except Exception as e:
            logger.error(f"Error generating hashtags: {str(e)}")
            return []
            
    def adjust_pacing(self, script: Dict[str, any], target_duration: int) -> Dict[str, any]:
        """Adjust scene durations to match target duration.
        
        Args:
            script (Dict): Generated script dictionary
            target_duration (int): Target total duration in seconds
            
        Returns:
            Dict: Adjusted script dictionary
        """
        scenes = script['script']
        current_total = sum(scene['duration'] for scene in scenes)
        
        if current_total == target_duration:
            return script
            
        # Calculate adjustment factor
        factor = target_duration / current_total
        
        # Adjust durations
        for scene in scenes:
            scene['duration'] = max(1, int(scene['duration'] * factor))
            
        # Fine-tune to exactly match target duration
        total = sum(scene['duration'] for scene in scenes)
        diff = target_duration - total
        
        if diff != 0:
            # Distribute remaining seconds
            for i in range(abs(diff)):
                if diff > 0:
                    scenes[i % len(scenes)]['duration'] += 1
                else:
                    scenes[i % len(scenes)]['duration'] = max(1, scenes[i % len(scenes)]['duration'] - 1)
                    
        return script 