# AI Video Generator

An advanced tool for generating dynamic video backgrounds synchronized with music and creating TikTok-style portrait videos from text prompts.

## Features

### Music-to-Video Background Generation

- Music upload and selection from integrated platforms
- Visual theme customization (Realistic, Animated, Abstract, Cinematic)
- Beat synchronization with visual effects
- Preview and editing capabilities
- High-quality export options

### TikTok-Style Portrait Video Generation

- Text-to-script conversion using LLMs
- AI-generated visuals
- Voiceover and sound effects
- Portrait-mode editor
- One-click publishing

## Installation

1. Clone the repository:

```bash
git clone [repository-url]
cd ai-video-generator
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   Create a `.env` file in the root directory and add your API keys:

```
OPENAI_API_KEY=your_key_here
TOPMEDIAI_API_KEY=your_key_here
```

## Usage

1. Start the application:

```bash
python src/main.py
```

2. Open your web browser and navigate to `http://localhost:7860`

## Project Structure

```
ai-video-generator/
├── src/
│   ├── audio/
│   │   ├── beat_detection.py
│   │   └── music_processor.py
│   ├── video/
│   │   ├── generator.py
│   │   └── effects.py
│   ├── models/
│   │   ├── text_to_script.py
│   │   └── visual_generator.py
│   └── ui/
│       └── interface.py
├── tests/
├── assets/
│   ├── samples/
│   └── templates/
└── config/
    └── default_settings.py
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
