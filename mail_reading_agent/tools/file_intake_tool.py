import os
import hashlib
import json
from datetime import datetime
from typing import Dict
from autogen_core.tools import FunctionTool

# --- Configuration ---
# ATTACHMENTS_FOLDER = "/Users/varunnegi/weave-agent-genui/gmail/gmail/attachments"
ATTACHMENTS_FOLDER = os.path.join(os.getcwd(), "attachments")
STATE_FILE = "processed_files.json"

# --- Helper Functions ---

def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_processed_files() -> Dict:
    """Load processed files from JSON, handling empty or invalid files."""
    if not os.path.exists(STATE_FILE):
        with open(STATE_FILE, "w") as f:
            f.write("{}")
        return {}

    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def save_processed_files(processed_files: Dict) -> None:
    """Save processed files dictionary to JSON."""
    with open(STATE_FILE, 'w') as f:
        json.dump(processed_files, f, indent=2)


def mark_file_as_processed(file_path: str, file_hash: str) -> None:
    """Mark a file as processed."""
    processed_files = load_processed_files()
    processed_files[file_hash] = {
        "file_name": os.path.basename(file_path),
        "file_path": file_path,
        "processed_at": datetime.now().isoformat()
    }
    save_processed_files(processed_files)


# --- File Intake Function ---

def file_intake_tool() -> Dict:
    """
    Check attachments folder for new PDF or Excel files.
    Returns one unprocessed file at a time and marks it as processed.
    """
    if not os.path.exists(ATTACHMENTS_FOLDER):
        return {
            "status": "error",
            "message": f"Attachments folder not found: {ATTACHMENTS_FOLDER}",
            "timestamp": datetime.now().isoformat()
        }

    processed_files = load_processed_files()

    # Get all PDF/Excel files
    files = [
        f for f in os.listdir(ATTACHMENTS_FOLDER)
        if f.lower().endswith(('.pdf', '.xls', '.xlsx')) and os.path.isfile(os.path.join(ATTACHMENTS_FOLDER, f))
    ]

    if not files:
        return {
            "status": "no_new_files",
            "message": "No PDF or Excel files found in attachments folder",
            "timestamp": datetime.now().isoformat()
        }

    # Check for new/unprocessed files
    for file_name in sorted(files):
        file_path = os.path.join(ATTACHMENTS_FOLDER, file_name)
        try:
            file_hash = calculate_file_hash(file_path)
            if file_hash not in processed_files:
                mark_file_as_processed(file_path, file_hash)
                return {
                    "status": "found",
                    "file_path": file_path,
                    "file_name": file_name,
                    "file_hash": file_hash,
                    "timestamp": datetime.now().isoformat(),
                    "message": f"New file detected and marked as processed: {file_name}"
                }
        except Exception as e:
            print(f"Warning: Could not process {file_name}: {str(e)}")
            continue

    return {
        "status": "no_new_files",
        "message": f"All {len(files)} files in folder have been processed",
        "timestamp": datetime.now().isoformat()
    }


# --- Wrap FunctionTool ---

file_intake_tool = FunctionTool(
    name="file_intake_tool",
    description="find the attachments folder for new PDF or Excel files and marks them as processed.",
    func=file_intake_tool
)
