from ultralytics import YOLO

from parameters import CHECKPOINTS_DIR, ROOT

RESULTS_DIR = ROOT / "results"

# Pré-entraîné COCO ; téléchargé dans checkpoints/ au premier lancement.
model = YOLO(CHECKPOINTS_DIR / "yolov8m.pt")
model.info()

results = model(ROOT / "bus.jpg")
for r in results:
    for box in r.boxes:
        print(f"{r.names[int(box.cls)]:>10s}  conf={float(box.conf):.2f}  xyxy={[round(v) for v in box.xyxy[0].tolist()]}")
    RESULTS_DIR.mkdir(exist_ok=True)
    r.save(filename=str(RESULTS_DIR / "bus_yolov8m_fp32.jpg"))
