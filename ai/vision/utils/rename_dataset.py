import os
import shutil

def get_unique_prefix(dst_root, prefix):
    base_prefix = prefix
    counter = 1
    exts = [".jpg", ".jpeg", ".png"]

    while True:
        exists = False
        for ext in exts:
            test_file = os.path.join(dst_root, "train", "images", f"{prefix}_001{ext}")
            if os.path.exists(test_file):
                exists = True
                break
        if not exists:
            return prefix
        counter += 1
        prefix = f"{base_prefix}{counter}"
    




def rename_and_copy_dataset(src_root, dst_root, prefix="egg"):
    os.makedirs(dst_root, exist_ok=True)
    prefix = get_unique_prefix(dst_root, prefix)

    for split in ["train", "valid", "test"]:
        img_src = os.path.join(src_root, split, "images")
        lbl_src = os.path.join(src_root, split, "labels")

        img_dst = os.path.join(dst_root, split, "images")
        lbl_dst = os.path.join(dst_root, split, "labels")
        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(lbl_dst, exist_ok=True)

        images = sorted([f for f in os.listdir(img_src) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

        for idx, img_name in enumerate(images, start=1):
            new_name = f"{prefix}_{idx:03d}"
            ext = os.path.splitext(img_name)[1].lower()

            old_img_path = os.path.join(img_src, img_name)
            old_lbl_path = os.path.join(lbl_src, os.path.splitext(img_name)[0] + ".txt")

            new_img_path = os.path.join(img_dst, new_name + ext)
            new_lbl_path = os.path.join(lbl_dst, new_name + ".txt")

            shutil.copy2(old_img_path, new_img_path)

         
            if os.path.exists(old_lbl_path):
                shutil.copy2(old_lbl_path, new_lbl_path)

rename_and_copy_dataset("/home/ubuntu/Downloads/egg_detection_images", "/home/ubuntu/Workspaces/Yummiverse/datasets/food_detection")
