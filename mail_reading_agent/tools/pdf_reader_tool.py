import os
import fitz  # PyMuPDF
import json
from datetime import datetime
from autogen_core.tools import FunctionTool
from typing import Optional

# ---------------- Config ----------------
STATE_FILE = "processed_files.json"
# ATTACHMENTS_FOLDER = "attachments"
ATTACHMENTS_FOLDER = os.path.join(os.getcwd(), "attachments")

# ---------------- Utility Functions ----------------
def load_processed_files():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_processed_files(processed_files):
    with open(STATE_FILE, 'w') as f:
        json.dump(processed_files, f, indent=2)

def mark_file_as_processed(file_path):
    processed_files = load_processed_files()
    processed_files[os.path.basename(file_path)] = {
        "file_path": file_path,
        "processed_at": datetime.now().isoformat()
    }
    save_processed_files(processed_files)

def get_next_pdf_file():
    """Return the next unprocessed PDF file from attachments folder."""
    processed_files = load_processed_files()
    pdf_files = sorted([
        f for f in os.listdir(ATTACHMENTS_FOLDER)
        if f.lower().endswith(".pdf") and f not in processed_files
    ])
    if not pdf_files:
        return None
    return os.path.join(ATTACHMENTS_FOLDER, pdf_files[0])

# ---------------- PDF Reader Function ----------------
async def read_single_pdf(file_path: Optional[str] = None) -> dict:
    """
    Reads a single PDF file and returns its text content.
    If no file_path is provided, picks the next unprocessed PDF in attachments.
    """
    if not file_path:
        file_path = get_next_pdf_file()
        if not file_path:
            return {"status": "no_files", "message": "No unprocessed PDF files found."}

    if not os.path.isfile(file_path):
        return {"status": "failed", "error": f"File not found: {file_path}"}

    try:
        with fitz.open(file_path) as pdf_doc:
            text = ""
            for page_num, page in enumerate(pdf_doc, start=1):
                text += f"\n--- Page {page_num} ---\n{page.get_text('text')}"
            page_count = pdf_doc.page_count  # store before closing

        # Mark as processed
        mark_file_as_processed(file_path)

        return {
            "status": "success",
            "filename": os.path.basename(file_path),
            "pages": page_count,
            "content": text.strip()
        }

    except Exception as e:
        return {"status": "failed", "error": str(e)}

# ---------------- Wrap as FunctionTool ----------------
pdf_reader_tool = FunctionTool(
    func=read_single_pdf,
    name="pdf_reader_tool",
    description="Reads PDF file from attachments folder and marks it as processed."
)
