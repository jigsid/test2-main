# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Set Python path and run application
$env:PYTHONPATH = "$PWD"
python src/main.py 