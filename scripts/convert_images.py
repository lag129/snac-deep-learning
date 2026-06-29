from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "data"
DST_DIR = PROJECT_ROOT / "data_processed"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".avif",
    ".jfif",
    ".gif",
}


def main():
    converted = 0
    skipped = 0

    for src_path in sorted(SRC_DIR.rglob("*")):
        if not src_path.is_file():
            continue
        if src_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            skipped += 1
            print(f"Skipped: {src_path}")
            continue

        relative = src_path.relative_to(SRC_DIR)
        dst_path = DST_DIR / relative.parent / (src_path.stem + ".jpg")
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(src_path) as img:
            img.convert("RGB").save(dst_path, "JPEG", quality=95)

        print(f"Converted: {src_path} -> {dst_path}")
        converted += 1

    print(f"\nDone: {converted} converted, {skipped} skipped")


if __name__ == "__main__":
    main()
