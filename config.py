from dotenv import load_dotenv
import os

load_dotenv()
CHROMA_DIR = os.getenv("CHROMA_DIR", "database")
SYLLABUS_PATH = os.getenv("SYLLABUS_PDF", "document_load\Syllabus.pdf")
HF_REPO   = os.getenv("HF_REPO_ID", "mistralai/Mistral-7B-Instruct-v0.3")
TIMEOUT   = int(os.getenv("HF_TIMEOUT", "120"))
