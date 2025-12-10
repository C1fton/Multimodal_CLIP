import os

# 1. 定义 ImageNette 的 10 个类别映射 (WNID -> 类别名)
IMAGENETTE_CLASSES = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n03180011": "cassette player",
    "n03425413": "chain saw",
    "n03028079": "church",
    "n03445777": "French horn",
    "n03445924": "garbage truck",
    "n03461329": "gas pump",
    "n03544360": "golf ball",
    "n04037443": "parachute"
}

# 2. 设置路径
root_dir = os.path.join("data", "imagenet")
train_dir = os.path.join(root_dir, "images", "train")
target_file = os.path.join(root_dir, "classnames.txt")


def fix():
    if not os.path.exists(train_dir):
        print(f"❌ 错误：找不到训练目录：{train_dir}")
        return

    print(f"正在扫描目录：{train_dir} ...")

    folders = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

    lines = []
    for folder in folders:
        if folder in IMAGENETTE_CLASSES:
            class_name = IMAGENETTE_CLASSES[folder]
            # --- 关键修改：写入 "ID 名字" 的格式 ---
            lines.append(f"{folder} {class_name}")
        else:
            # 如果遇到未知文件夹，也保持格式
            print(f"⚠️ 警告：未知文件夹 {folder}")
            lines.append(f"{folder} {folder}")

    with open(target_file, "w") as f:
        f.write("\n".join(lines))

    print(f"✅ 成功！已生成修正后的文件：{target_file}")
    print("内容预览（前3行）：")
    for i in range(min(3, len(lines))):
        print(f" - {lines[i]}")
    print("格式应为: 'n0xxxxxxx classname'")


if __name__ == "__main__":
    fix()