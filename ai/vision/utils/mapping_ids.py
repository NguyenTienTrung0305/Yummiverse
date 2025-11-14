import argparse
import yaml
from pathlib import Path


def load_names(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names")

    # is dict
    if isinstance(names, dict):
        names_list = [name for _, name in sorted(names.items(), key=lambda kv: int(kv[0]))]
    else:
        # is list
        names_list = list(names)

    return names_list

def mapping_ids(master_labels_path, subset_labels_path):
    master_names = load_names(master_labels_path)
    subset_names = load_names(subset_labels_path)
    
    mapping = {}
    
    for subset_idx, name in enumerate(subset_names):
        if name in master_names:
            master_idx = master_names.index(name)
            mapping[subset_idx] = master_idx
        else:
            print(f"[WARN] Class '{name}' không tồn tại trong master labels, bỏ qua.")

    return mapping


def mapping_labels_in_file(file_path: Path, id_mapping: dict):
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = file_path.read_text(encoding='utf-8', errors='ignore').splitlines()
        
    out_lines = []
    
    for line in lines:
        parts = line.strip().split()
        if not parts:
            out_lines.append(line)
            continue
        
        try:
            old_id = int(parts[0])
        except ValueError:
            out_lines.append(line)
            continue

        new_id = id_mapping.get(old_id, old_id)  # nếu không có mapping thì giữ nguyên
        parts[0] = str(new_id)
        out_lines.append(" ".join(parts))
    
    file_path.write_text("\n".join(out_lines), encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(
        description="Map class IDs in files within a directory based on master and subset label files."
    )
    parser.add_argument(
        '--labels_dir',
        required=True,
        help="Directory containing files .txt", 
    )
    parser.add_argument(
        '--master_labels',
        required=True,
        help="File yaml that containing labels name and ids", 
    )
    parser.add_argument(
        '--subset_labels',
        required=True,
        help="File yaml that containing sub labels name", 
    )
    
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search through subdirectories.",
    )
    
    args = parser.parse_args()
    labels_dir = Path(args.labels_dir)
    master_labels_path = Path(args.master_labels)
    subset_labels_path = Path(args.subset_labels)
    
    id_mapping = mapping_ids(master_labels_path, subset_labels_path)
    
    for file_path in labels_dir.rglob("*.txt") if args.recursive else labels_dir.glob("*.txt"):
        mapping_labels_in_file(file_path, id_mapping)

if __name__ == "__main__":
    main()

# python mapping_ids.py `
#   --master_labels "D:/Code/Python/Yummiverse/ai/vision/data_detection.yaml" `
#   --subset_labels "C:/Users/ADMIN/Downloads/YUMMI.v3i.yolov8/data.yaml" `
#   --labels_dir "C:/Users/ADMIN/Downloads/YUMMI.v3i.yolov8/train/labels"


# change class id in the subset dataset to match the master dataset