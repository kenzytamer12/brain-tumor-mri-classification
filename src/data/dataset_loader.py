import os
from PIL import Image


def check_corrupted_images(folder):
    corrupted_images = []

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(root, file)

                try:
                    with Image.open(image_path) as img:
                        img.verify()
                except Exception:
                    corrupted_images.append(image_path)

    print(f"\nCorrupted images found in {folder}: {len(corrupted_images)}")

    if corrupted_images:
        for img in corrupted_images:
            print("Corrupted:", img)

    return corrupted_images


def organize_dataset(source_dir="data/raw/brain_dataset"):
    print("Checking Dataset Structure...\n")

    train_dir = os.path.join(source_dir, "Training")
    test_dir = os.path.join(source_dir, "Testing")

    def check_folder(folder, name):
        if os.path.exists(folder):
            classes = sorted(os.listdir(folder))

            print(f"{name} folder found with {len(classes)} classes:")

            total_images = 0

            for cls in classes:
                class_path = os.path.join(folder, cls)

                if os.path.isdir(class_path):
                    count = len([
                        file for file in os.listdir(class_path)
                        if file.lower().endswith((".jpg", ".jpeg", ".png"))
                    ])

                    total_images += count
                    print(f"   • {cls}: {count} images")

            print(f"Total images in {name}: {total_images}")

            return True
        else:
            print(f"{name} folder NOT found!")
            return False

    train_ok = check_folder(train_dir, "Training")
    test_ok = check_folder(test_dir, "Testing")

    print("\n" + "=" * 60)

    if train_ok and test_ok:
        print("Dataset folders are ready.")

        print("\nChecking corrupted images...")

        train_corrupted = check_corrupted_images(train_dir)
        test_corrupted = check_corrupted_images(test_dir)

        total_corrupted = len(train_corrupted) + len(test_corrupted)

        print("\n" + "=" * 60)

        if total_corrupted == 0:
            print("No corrupted images found.")
            print("Dataset is ready for training!")
        else:
            print(f"Found {total_corrupted} corrupted images.")
            print("Please remove or replace corrupted images before training.")

    else:
        print("Please download the dataset and place it correctly.")

    print("=" * 60)


if __name__ == "__main__":
    organize_dataset()