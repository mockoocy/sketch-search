"""
Used to create a smaller gallery from some dataset
for purposes of testing the application.

Usage:
    python create_gallery.py --src path/to/data --dst path/to/gallery --per-class 5
"""

import argparse
import shutil
from pathlib import Path
from random import sample

parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True)
parser.add_argument("--dst", required=True)
parser.add_argument("--per-class", type=int, default=10)
parser.add_argument("--max-classes", type=int, default=None)
args = parser.parse_args()

src = Path(args.src)
dst = Path(args.dst)
dst.mkdir(parents=True, exist_ok=True)


dirs = sorted(path for path in src.iterdir() if path.is_dir())
if args.max_classes is not None:
    dirs = sample(dirs, args.max_classes)
for cls_dir in dirs:
    out_cls = dst / cls_dir.name
    out_cls.mkdir(parents=True, exist_ok=True)
    images = sorted(f for f in cls_dir.iterdir() if f.is_file())
    for img in images[: args.per_class]:
        shutil.copy2(img, out_cls / img.name)
