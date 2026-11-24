"""Construit data/bus_coco/ : les images COCO 2017 qui contiennent au moins un bus.

Structure produite (format YOLO, lisible par ultralytics et par fpga_inference_yolo.py) :
  data/bus_coco/images/<split>/*.jpg
  data/bus_coco/labels/<split>/*.txt   une ligne par objet : classe cx cy w h (normalisés 0-1)
  data/bus_coco.yaml                   à passer à YOLO.val(data=...)

Toutes les classes sont gardées dans les labels (un bus est souvent avec des voitures,
des personnes...), seule la sélection des images se fait sur la classe bus.

  python src/download_bus_dataset.py                 # val2017 : ~190 images
  python src/download_bus_dataset.py --split train2017 --max 1000
"""
import argparse
import json
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

from parameters import DATA_DIR

BUS = 5  # index COCO-80 (ultralytics)
LABELS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"
IMAGE_URL = "http://images.cocodataset.org/{split}/{name}"
OUT = DATA_DIR / "bus_coco"

parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["val2017", "train2017"], default="val2017")
parser.add_argument("--max", type=int, help="Nombre max d'images (par défaut : toutes).")
args = parser.parse_args()

labels_zip = DATA_DIR / "coco2017labels.zip"
if not labels_zip.is_file():
    print(f"Téléchargement des labels COCO (~46 Mo) -> {labels_zip}")
    urllib.request.urlretrieve(LABELS_URL, labels_zip)

img_dir = OUT / "images" / args.split
lbl_dir = OUT / "labels" / args.split
img_dir.mkdir(parents=True, exist_ok=True)
lbl_dir.mkdir(parents=True, exist_ok=True)

selected = []
with zipfile.ZipFile(labels_zip) as z:
    prefix = f"coco/labels/{args.split}/"
    for entry in sorted(n for n in z.namelist() if n.startswith(prefix) and n.endswith(".txt")):
        text = z.read(entry).decode()
        if any(line.split()[0] == str(BUS) for line in text.splitlines() if line.strip()):
            name = entry.rsplit("/", 1)[1]
            (lbl_dir / name).write_text(text)
            selected.append(name.replace(".txt", ".jpg"))
            if args.max and len(selected) >= args.max:
                break


def fetch(name):
    dst = img_dir / name
    if not dst.is_file():
        urllib.request.urlretrieve(IMAGE_URL.format(split=args.split, name=name), dst)


print(f"{len(selected)} images avec un bus dans {args.split}, téléchargement...")
with ThreadPoolExecutor(max_workers=16) as pool:
    list(pool.map(fetch, selected))

# Noms des 80 classes, dans l'ordre d'ultralytics (le JSON est du YAML valide).
names = json.loads((DATA_DIR / "coco128.yaml").read_text())["names"] if (DATA_DIR / "coco128.yaml").is_file() else None
splits = sorted(p.name for p in (OUT / "images").iterdir() if p.is_dir())
cfg = {"path": str(OUT), "train": f"images/{splits[0]}", "val": f"images/{args.split}"}
if names:
    cfg["names"] = names
(DATA_DIR / "bus_coco.yaml").write_text(json.dumps(cfg))
print(f"Terminé : {img_dir} ({len(list(img_dir.glob('*.jpg')))} images)")
