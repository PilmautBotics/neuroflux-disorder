import shutil
import re
from pathlib import Path
import argparse

def get_patient_id(filename):
    """
    Extract patient ID from filename.
    Adjust the regex according to your naming convention.
    """
    match = re.search(r'(\d{3}_S_\d{4})', filename)
    return match.group(1) if match else None

def reorganize(src_dir, dst_dir):
    """
    Reorganize the dataset from class folders to class/patient folders.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")
    
    dst_dir.mkdir(parents=True, exist_ok=True)

    for class_folder in src_dir.iterdir():
        if not class_folder.is_dir():
            continue

        for img_path in class_folder.iterdir():
            if not img_path.is_file() or img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue

            patient_id = get_patient_id(img_path.name)
            if not patient_id:
                continue

            target_folder = dst_dir / class_folder.name / patient_id
            target_folder.mkdir(parents=True, exist_ok=True)

            dst_path = target_folder / img_path.name
            shutil.copy2(img_path, dst_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Reorganize dataset into class/patient folders.")
    parser.add_argument("--src", type=str, required=True, help="Source dataset directory")
    parser.add_argument("--dst", type=str, required=True, help="Destination directory")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    reorganize(args.src, args.dst)
