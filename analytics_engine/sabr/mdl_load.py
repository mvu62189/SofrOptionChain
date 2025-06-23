# mdl_load.py
# This module load parquet files from storage

import glob
import os
from typing import Dict, List


def discover_snapshot_files(root_folder: str = "snapshots") -> Dict[str, List[str]]:
    """
    Scan `root_folder` recursively for .parquet files.
    Returns a dict mapping each subfolder (relative to root) to list of file paths.
    """
    local_files = sorted(glob.glob(f"{root_folder}/**/*.parquet", recursive=True))
    file_dict: Dict[str, List[str]] = {}
    for f in local_files:
        folder = os.path.relpath(os.path.dirname(f), root_folder)
        file_dict.setdefault(folder, []).append(f)
    return file_dict


def save_uploaded_files(uploaded_files, upload_dir: str = "uploaded_files") -> List[str]:
    """
    Save Streamlit-uploaded files to disk and return list of saved paths.
    """
    os.makedirs(upload_dir, exist_ok=True)
    uploaded_paths: List[str] = []
    for f in uploaded_files:
        file_path = os.path.join(upload_dir, f.name)
        with open(file_path, "wb") as out_f:
            out_f.write(f.read())
        uploaded_paths.append(file_path)
    return uploaded_paths
