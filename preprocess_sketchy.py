import shutil
from collections import defaultdict
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def preprocess_sketchy(
    root: Path,
    out_images: Path,
    out_sketches: Path,
) -> None:
    out_images.mkdir(parents=True, exist_ok=True)
    out_sketches.mkdir(parents=True, exist_ok=True)

    photo_root = root / "photo"
    sketch_root = root / "sketch"

    # stem -> photo path
    photos: dict[str, Path] = {}

    # stem -> {k: sketch path}
    sketches: dict[str, dict[int, Path]] = defaultdict(dict)

    # ---- collect photos (deduplicate tx_*)
    for p in photo_root.glob("tx_*/**/*"):
        if p.suffix.lower() not in IMG_EXT:
            continue
        photos.setdefault(p.stem, p)

    # ---- collect sketches (deduplicate by stem + index)
    for s in sketch_root.glob("tx_*/**/*"):
        if s.suffix.lower() not in IMG_EXT:
            continue

        name = s.stem
        if "-" not in name:
            continue

        stem, k = name.rsplit("-", 1)
        if not k.isdigit():
            continue

        sketches[stem].setdefault(int(k), s)

    print(f"Found {len(photos)} unique photos")
    print(f"Found {sum(len(v) for v in sketches.values())} unique sketches")
    print(f"Will process {len(set(photos) & set(sketches))} paired instances")

    # ---- write outputs (each file exactly once)
    for stem, photo_path in photos.items():
        if stem not in sketches:
            continue

        out_photo = out_images / f"{stem}{photo_path.suffix.lower()}"
        if not out_photo.exists():
            shutil.copy2(photo_path, out_photo)

        for k, sk in sketches[stem].items():
            out_sk = out_sketches / f"{stem}_{k}{sk.suffix.lower()}"
            if not out_sk.exists():
                shutil.copy2(sk, out_sk)

    print("Processing complete.")
