import os
import shutil

def rename_and_copy_dataset(src_root, dst_root, prefix="egg"):
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    os.makedirs(dst_root)

    for split in ["train", "val", "test"]:
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

        print(f"[{split}] Done {len(images)} files")

rename_and_copy_dataset("path/to/original_dataset", "path/to/renamed_dataset")
