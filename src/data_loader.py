# for loading the pdf from client and user ,let say when the user want to upload the docs in our rag pipeline system,then it processs it for rag

import shutil
import uuid
import os
from pathlib import Path

UPLOAD_DIRECETORY=Path("Uploads")
UPLOAD_DIRECETORY.mkdir(exist_ok=True)


ALLOWED_EXTENSION={".pdf",".doc",".docx",".txt",".rtf",".json",".pptx",".csv",".md"}
MAX_FILE_SIZE=50*1024*1024 #50 mb

def upload_file(Source_path)->dict:
    """
    Copies a file into the uploads directory with validation.
    source_path: path to the file you want to upload (e.g. from a file picker or request).
    Returns dict with saved path, filename, and size.
    """
    src_path_name=Path(Source_path)

    if not src_path_name.exists():
        raise ValueError(
            "No path provided"
        )
    ext=src_path_name.suffix.lower()
    if ext not in ALLOWED_EXTENSION:
        raise ValueError(f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSION}")


    file_size=src_path_name.stat().st_size
    if file_size>MAX_FILE_SIZE:
        raise ValueError(
            f"File is too large,maximum allowed {MAX_FILE_SIZE/1000}"
        )
    # generates the unique number filename to avoid coliision/overwrite
    unique_name=f"{uuid.uuid4().hex}{ext}"
    dest_path=UPLOAD_DIRECETORY/unique_name

    shutil.copy2(src_path_name,dest_path)
    return{
        "Original_filename":src_path_name.name,
        "Saved_name":unique_name,
        "saved_path":str(dest_path),
        "size":file_size,
        "extension":ext
    }

