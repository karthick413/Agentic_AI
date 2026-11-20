# Import Path to work with file and folder paths.
from pathlib import Path

# Find the prompts folder relative to the current Python file.
# __file__ represents the location of this prompt_loader.py file.
PROMPT_DIR = Path(__file__).parent / "prompts"


# Define a function to load a prompt using its filename.
def load_prompt(name: str) -> str:

    # Build the complete path to the requested .txt file.
    # Example: summarize_v1 becomes summarize_v1.txt.
    path = PROMPT_DIR / f"{name}.txt"

    # Check whether the requested prompt file exists.
    if not path.exists():
        # Raise an error if the prompt file cannot be found.
        raise FileNotFoundError(f"Prompt not found: {path}")

    # Read the prompt file using UTF-8 encoding.
    # strip() removes unnecessary whitespace at the beginning and end.
    return path.read_text(encoding="utf-8").strip()
