import os
import shutil

def get_unique_prefix(dst_root, prefix):
    base_prefix = prefix
    counter = 1
    exts = [".jpg", ".jpeg", ".png", ".webp"]

    while True:
        exists = False
        for ext in exts:
            test_file = os.path.join(dst_root, "images", "train", f"{prefix}_000001{ext}")
            if os.path.exists(test_file):
                exists = True
                break
        if not exists:
            return prefix
        counter += 1
        prefix = f"{base_prefix}{counter}"
    




def rename_and_copy_dataset(src_root, dst_root, prefix="0___0"):
    os.makedirs(dst_root, exist_ok=True)
    prefix = get_unique_prefix(dst_root, prefix)

    for split in ["train", "val", "test"]:
        roboflow_split = "valid" if split == "val" else split
        
        img_src = os.path.join(src_root, roboflow_split, "images")
        lbl_src = os.path.join(src_root, roboflow_split, "labels")

        img_dst = os.path.join(dst_root, "images", split)
        lbl_dst = os.path.join(dst_root, "labels", split)
        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(lbl_dst, exist_ok=True)

        images = sorted([f for f in os.listdir(img_src) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))])

        for idx, img_name in enumerate(images, start=1):
            new_name = f"{prefix}_{idx:06d}"
            ext = os.path.splitext(img_name)[1].lower()

            old_img_path = os.path.join(img_src, img_name)
            old_lbl_path = os.path.join(lbl_src, os.path.splitext(img_name)[0] + ".txt")

            new_img_path = os.path.join(img_dst, new_name + ext)
            new_lbl_path = os.path.join(lbl_dst, new_name + ".txt")

            shutil.copy2(old_img_path, new_img_path)

         
            if os.path.exists(old_lbl_path):
                shutil.copy2(old_lbl_path, new_lbl_path)

rename_and_copy_dataset(
    "C:/Users/ADMIN/Downloads/SEGYUM.v3i.yolov8",
    "D:/Code/Python/Yummiverse/ai/datasets/food_detection",
    prefix="dataset_ver_1"
)

# rename dataset and copy to another location that suitable for yolov8 training