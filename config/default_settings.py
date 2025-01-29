"""Default configuration settings for the AI Video Generator."""

# Audio Processing Settings
AUDIO_SETTINGS = {
    'sample_rate': 22050,
    'hop_length': 512,
    'n_mels': 128,
    'fmin': 20,
    'fmax': 8000
}

# Video Settings
VIDEO_SETTINGS = {
    'width': 1080,
    'height': 1920,
    'fps': 30,
    'codec': 'libx264',
    'bitrate': '4000k'
}

# Visual Effects Settings
EFFECTS_SETTINGS = {
    'particle_count': 100,
    'blur_radius': 5,
    'transition_duration': 0.5
}

# Theme Settings
THEME_SETTINGS = {
    'realistic': {
        'color_grading': {
            'contrast': 1.2,
            'brightness': 0.1,
            'saturation': 1.1
        },
        'effects': ['light_flares', 'bokeh']
    },
    'animated': {
        'color_grading': {
            'contrast': 1.3,
            'brightness': 0.2,
            'saturation': 1.4
        },
        'effects': ['particles', 'geometric']
    },
    'abstract': {
        'color_grading': {
            'contrast': 1.1,
            'brightness': 0.0,
            'saturation': 1.2
        },
        'effects': ['particles', 'waves']
    },
    'cinematic': {
        'color_grading': {
            'contrast': 1.4,
            'brightness': -0.1,
            'saturation': 0.9
        },
        'effects': ['light_flares', 'vignette']
    }
}

# Text Settings
TEXT_SETTINGS = {
    'font_size': 36,
    'font_family': 'Arial',
    'text_color': (255, 255, 255),
    'shadow_color': (0, 0, 0),
    'shadow_offset': 2
}

# Template Settings
TEMPLATE_SETTINGS = {
    'travel_vlog': {
        'structure': [
            {'type': 'intro', 'duration': 5},
            {'type': 'location_shots', 'duration': 15},
            {'type': 'highlights', 'duration': 25},
            {'type': 'outro', 'duration': 5}
        ],
        'style': 'cinematic',
        'music_type': 'upbeat'
    },
    'product_demo': {
        'structure': [
            {'type': 'problem_statement', 'duration': 5},
            {'type': 'product_intro', 'duration': 10},
            {'type': 'features', 'duration': 30},
            {'type': 'call_to_action', 'duration': 5}
        ],
        'style': 'realistic',
        'music_type': 'corporate'
    }
}

# API Settings
API_SETTINGS = {
    'openai': {
        'model': 'gpt-4',
        'temperature': 0.7,
        'max_tokens': 1000
    },
    'dalle': {
        'size': '1024x1024',
        'quality': 'standard',
        'style': 'vivid'
    }
}

# Cache Settings
CACHE_SETTINGS = {
    'enabled': True,
    'directory': 'cache',
    'max_size_gb': 5,
    'expiration_days': 7
}

# Export Settings
EXPORT_SETTINGS = {
    'formats': {
        'tiktok': {
            'width': 1080,
            'height': 1920,
            'fps': 30,
            'max_duration': 180
        },
        'instagram': {
            'width': 1080,
            'height': 1920,
            'fps': 30,
            'max_duration': 60
        },
        'youtube_shorts': {
            'width': 1080,
            'height': 1920,
            'fps': 30,
            'max_duration': 60
        }
    },
    'quality_presets': {
        'high': {
            'video_bitrate': '4000k',
            'audio_bitrate': '192k',
            'codec': 'libx264'
        },
        'medium': {
            'video_bitrate': '2000k',
            'audio_bitrate': '128k',
            'codec': 'libx264'
        },
        'low': {
            'video_bitrate': '1000k',
            'audio_bitrate': '96k',
            'codec': 'libx264'
        }
    }
} 