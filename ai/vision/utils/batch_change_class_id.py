import argparse
from email.base64mime import header_length
from math import e
from pathlib import Path

def change_file(file_path: Path, new_id: int) -> int:
    try:
        text = file_path.read_text(encoding='utf-8').splitlines()
    except UnicodeDecodeError:
        text = file_path.read_text(encoding='utf-8', errors='ignore').splitlines()
        
    out_lines = []
    changed = 0
    
    for line in text:
        parts = line.strip().split()
        if not parts:
            out_lines.append(line)
            continue
        
        try:
            int(parts[0])
            parts[0] = str(new_id)
            changed += 1
            out_lines.append(" ".join(parts))
        except ValueError:
            out_lines.append(line)
            
    file_path.write_text("\n".join(out_lines), encoding="utf-8")
    return changed

def main():
    parser = argparse.ArgumentParser(
        description="Change class IDs in files within a directory."
    )
    parser.add_argument(
        '--labels_dir',
        help="Directory containing files .txt (e.g. labels/train)", 
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Prefix of the files to be changed (e.g. 'cucumber' for files like cucumber_*.txt)",
    )
    parser.add_argument(
        "--new_id",
        type=int,
        help="New class ID to set in the files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search through subdirectories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test run without making any changes.",
    )
    parser.add_argument(
        '--ext',
        default='txt',
        help="File extension to look for (default: txt).",
    )
    
    args = parser.parse_args()
    
    labels_dir = Path(args.labels_dir)
    if not labels_dir.exists():
        raise FileNotFoundError(f"Directory {labels_dir} does not exist.")
    
    
    # create pattern to search for files
    if args.prefix:
        pattern = f"{args.prefix}_*.{args.ext}"
    else:
        pattern = f"*.{args.ext}"
    
    files = labels_dir.rglob(pattern=pattern) if args.recursive else labels_dir.glob(pattern=pattern)
    files = list(files)
    if not files:
        raise FileNotFoundError(f"No files found with pattern {pattern} in {labels_dir}.")
    
    total_files = 0
    total_lines = 0
    for file in files:
        if args.dry_run:
            try:
                lines = file.read_text(encoding='utf-8').splitlines()
            except UnicodeDecodeError:
                lines = file.read_text(encoding='utf-8', errors='ignore').splitlines()
            changables = sum(1 for line in lines if line.strip().split()[:1] and line.strip().split()[0].isdigit())
            total_files += 1
            total_lines += changables
        else:
            changed = change_file(file, args.new_id)
            total_files += 1
            total_lines += changed
            
    mode = "DRY RUN: " if args.dry_run else "DONE"
    print(f"\n[{mode}] SUM: {total_files} file, {total_lines} handled (id → {args.new_id}).")
            
if __name__ == "__main__":
    main()

# python batch_change_class_id.py `
#   --labels_dir "C:/Users/ADMIN/Downloads/YUMMI.v3i.yolov8/train/labels" `
#   --new_id 0 `
#   --prefix "noodle_fresh" `
#   --recursive

# change all class id in files to specified new id (should rename dataset first then use this to change class ids) 