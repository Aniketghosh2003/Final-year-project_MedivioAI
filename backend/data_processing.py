import os
import numpy as np
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt

# Allowed image extensions
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def is_image_file(filename: str) -> bool:
    """Return True if filename looks like a valid image file we want to load."""
    name = os.path.basename(filename)
    return (
        not name.startswith(".")
        and os.path.splitext(name)[1].lower() in ALLOWED_EXTS
    )


def iter_image_files(root: str):
    """Yield valid image file paths under root (recursively)."""
    for p in Path(root).rglob("*"):
        if p.is_file() and is_image_file(p.name):
            yield str(p)


def safe_open(path: str):
    """Open an image safely; return None if it's not a valid image."""
    try:
        # Quick verification without full decode
        with Image.open(path) as im:
            im.verify()
        # Reopen to actually load after verify
        return Image.open(path).convert("L")  # visualize as grayscale
    except (UnidentifiedImageError, OSError):
        return None

def analyze_dataset(data_dir):
    """Analyze dataset distribution (counts only valid images)."""
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    class_counts = {}

    for cls in classes:
        class_path = os.path.join(data_dir, cls)
        # Count only valid image files and skip hidden/system files (e.g., .DS_Store, Thumbs.db)
        count = sum(
            1
            for name in os.listdir(class_path)
            if os.path.isfile(os.path.join(class_path, name)) and is_image_file(name)
        )
        class_counts[cls] = count

    print(f"\nDataset: {data_dir}")
    for cls, count in class_counts.items():
        print(f"{cls}: {count} images")

    return class_counts

def visualize_samples(data_dir, num_samples=5):
    """Visualize sample images from each class, skipping non-image/system files."""
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    rows, cols = len(classes), max(1, num_samples)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

    # Ensure axes is 2D for consistent indexing
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = axes.reshape(rows, 1)

    for i, cls in enumerate(classes):
        class_path = os.path.join(data_dir, cls)
        valid_paths = list(iter_image_files(class_path))
        shown = 0
        j = 0
        while shown < cols and j < len(valid_paths):
            img = safe_open(valid_paths[j])
            j += 1
            if img is None:
                continue

            axes[i, shown].imshow(img, cmap='gray')
            axes[i, shown].axis('off')
            if shown == 0:
                axes[i, shown].set_title(cls, fontsize=12)
            shown += 1

        # If not enough images to fill the row, hide remaining axes
        for k in range(shown, cols):
            axes[i, k].axis('off')

    plt.tight_layout()
    plt.savefig('sample_images.png')
    print("\nSample images saved as 'sample_images.png'")

if __name__ == "__main__":
    # Analyze all datasets
    train_counts = analyze_dataset('data/train')
    val_counts = analyze_dataset('data/val')
    test_counts = analyze_dataset('data/test')
    
    # Visualize samples
    visualize_samples('data/train')