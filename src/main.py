import logging
from ui.app import VideoApp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the video generation application."""
    try:
        # Create and launch the application
        app = VideoApp()
        ui = app.create_ui()
        ui.launch(
            server_name="0.0.0.0",  # Allow external access
            server_port=7860,       # Default Gradio port
            share=True             # Create a public link
        )
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 