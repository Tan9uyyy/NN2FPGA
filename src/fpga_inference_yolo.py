import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

BUS = 5
IMGSZ = 640


def letterbox(path: Path):
    """Redimensionne en gardant le ratio et complète en gris (114), comme ultralytics."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    r = IMGSZ / max(w, h)
    nw, nh = round(w * r), round(h * r)
    pad_x, pad_y = (IMGSZ - nw) // 2, (IMGSZ - nh) // 2
    canvas = Image.new("RGB", (IMGSZ, IMGSZ), (114, 114, 114))
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (pad_x, pad_y))
    x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(x), (w, h), r, pad_x, pad_y


def iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area(box) + area(boxes) - inter + 1e-9)


def postprocess(output, r, pad_x, pad_y, conf_thr, iou_thr):
    """[1, 84, 8400] -> détections (x1, y1, x2, y2, conf, classe) en pixels de l'image d'origine."""
    pred = np.asarray(output)[0].T
    scores = pred[:, 4:]
    cls = scores.argmax(1)
    conf = scores[np.arange(len(cls)), cls]
    keep = conf > conf_thr
    pred, cls, conf = pred[keep], cls[keep], conf[keep]
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)

    # NMS par classe (décalage des boîtes par classe pour une seule passe).
    shifted = boxes + cls[:, None] * 4096.0
    order, kept = conf.argsort()[::-1], []
    while order.size:
        i = order[0]
        kept.append(i)
        order = order[1:][iou(shifted[i], shifted[order[1:]]) < iou_thr]

    boxes = boxes[kept]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / r
    return boxes, conf[kept], cls[kept]


def load_bus_labels(image_path: Path, size):
    """Boîtes de bus de vérité terrain (format YOLO normalisé -> pixels)."""
    label = image_path.parent.parent.parent / "labels" / image_path.parent.name / (image_path.stem + ".txt")
    if not label.is_file():
        return None
    w, h = size
    rows = [l.split() for l in label.read_text().splitlines() if l.strip()]
    gt = np.array([[float(v) for v in row[1:5]] for row in rows if int(row[0]) == BUS]).reshape(-1, 4)
    return np.stack([(gt[:, 0] - gt[:, 2] / 2) * w, (gt[:, 1] - gt[:, 3] / 2) * h,
                     (gt[:, 0] + gt[:, 2] / 2) * w, (gt[:, 1] + gt[:, 3] / 2) * h], 1)


def match_bus(boxes, cls, conf, gt, iou_thr=0.5):
    """Nombre de bus prédits qui correspondent à un bus réel distinct (glouton par confiance)."""
    pred = boxes[cls == BUS][conf[cls == BUS].argsort()[::-1]]
    used, tp = np.zeros(len(gt), bool), 0
    for b in pred:
        if not len(gt):
            break
        ious = np.where(used, 0, iou(b, gt))
        j = ious.argmax()
        if ious[j] >= iou_thr:
            used[j], tp = True, tp + 1
    return tp, len(pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nn2FPGA_yolov8m.onnx", help="Modèle ONNX (défaut : nn2FPGA_yolov8m.onnx).")
    parser.add_argument("--custom-op", default="libnn2fpga_customop.so", help="Bibliothèque du custom op nn2FPGA.")
    parser.add_argument("--cpu", action="store_true", help="Sans custom op : référence CPU (ex. original_model_qcdq.onnx).")
    parser.add_argument("--images", required=True, help="Dossier d'images (.jpg), labels YOLO dans ../../labels/<split>/.")
    parser.add_argument("--samples", type=int, help="Nombre d'images (par défaut : toutes).")
    parser.add_argument("--warmup", type=int, default=3, help="Inférences de chauffe non mesurées (défaut : 3).")
    parser.add_argument("--conf", type=float, default=0.25, help="Seuil de confiance (défaut : 0.25).")
    parser.add_argument("--iou", type=float, default=0.7, help="Seuil IoU de la NMS (défaut : 0.7).")
    parser.add_argument("--output", default="detections.json", help="Détections et métriques (défaut : detections.json).")
    args = parser.parse_args()
    if args.samples is not None and args.samples <= 0:
        parser.error("--samples doit être strictement positif.")

    images = sorted(Path(args.images).glob("*.jpg"))[:args.samples]
    if not images:
        parser.error(f"Aucune image .jpg dans {args.images}")
    if not Path(args.model).is_file():
        parser.error(f"Modèle absent : {args.model}")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    if not args.cpu:
        so.register_custom_ops_library(str(Path(args.custom_op).resolve()))
    t0 = time.perf_counter()
    session = ort.InferenceSession(args.model, sess_options=so, providers=["CPUExecutionProvider"])
    load_time = time.perf_counter() - t0  # inclut la programmation du bitstream en mode FPGA
    input_name = session.get_inputs()[0].name
    print(f"Modèle : {args.model} | Mode : {'CPU' if args.cpu else 'FPGA'} | Chargement : {load_time:.2f} s")

    first = letterbox(images[0])[0]
    for _ in range(args.warmup):
        session.run(None, {input_name: first})

    total_time = pre_time = post_time = 0.0
    latencies, records = [], []
    total_bus_gt = total_bus_pred = total_bus_tp = total_detections = 0

    for path in images:
        t = time.perf_counter()
        x, size, r, pad_x, pad_y = letterbox(path)
        pre_time += time.perf_counter() - t

        start = time.perf_counter()
        outputs = session.run(None, {input_name: x})
        dt = time.perf_counter() - start
        total_time += dt
        latencies.append(dt)

        t = time.perf_counter()
        boxes, conf, cls = postprocess(outputs[0], r, pad_x, pad_y, args.conf, args.iou)
        post_time += time.perf_counter() - t

        gt = load_bus_labels(path, size)
        tp, n_pred = match_bus(boxes, cls, conf, gt) if gt is not None else (0, int((cls == BUS).sum()))
        total_bus_gt += len(gt) if gt is not None else 0
        total_bus_pred += n_pred
        total_bus_tp += tp
        total_detections += len(boxes)
        records.append({
            "image": path.name, "run_ms": dt * 1e3,
            "bus_gt": None if gt is None else len(gt), "bus_pred": n_pred, "bus_tp": tp,
            "detections": [{"cls": int(c), "conf": round(float(s), 4), "xyxy": [round(float(v), 1) for v in b]}
                           for b, s, c in zip(boxes, conf, cls)],
        })

    n = len(images)
    precision = total_bus_tp / total_bus_pred if total_bus_pred else 0.0
    recall = total_bus_tp / total_bus_gt if total_bus_gt else 0.0
    average_image_time = total_time / n
    throughput = n / total_time if total_time > 0 else 0.0
    p50, p90, p95 = np.percentile(np.array(latencies) * 1e3, [50, 90, 95])

    print("\n===== Résultats finaux =====")
    print(f"Échantillons traités :  {n}")
    print(f"Détections (toutes) :   {total_detections}")
    print(f"Bus réels :             {total_bus_gt}")
    print(f"Bus détectés :          {total_bus_pred}")
    print(f"Bus corrects (IoU≥0.5): {total_bus_tp}")
    print(f"Précision bus :         {precision:.4f}")
    print(f"Rappel bus :            {recall:.4f}")
    print(f"Temps total :           {total_time:.6f} s")
    print(f"Temps moyen/image :     {average_image_time:.6f} s")
    print(f"Latence p50/p90/p95 :   {p50:.2f} / {p90:.2f} / {p95:.2f} ms")
    print(f"Débit :                 {throughput:.2f} images/s")
    print(f"Pré-traitement moyen :  {pre_time / n * 1e3:.2f} ms/image")
    print(f"Post-traitement moyen : {post_time / n * 1e3:.2f} ms/image")

    summary = {
        "model": args.model, "mode": "cpu" if args.cpu else "fpga", "load_time_s": load_time,
        "samples": n, "detections": total_detections, "bus_gt": total_bus_gt, "bus_pred": total_bus_pred,
        "bus_tp": total_bus_tp, "precision_bus": precision, "recall_bus": recall,
        "total_time_s": total_time, "avg_image_s": average_image_time, "throughput_img_s": throughput,
        "p50_ms": p50, "p90_ms": p90, "p95_ms": p95,
        "preprocess_ms": pre_time / n * 1e3, "postprocess_ms": post_time / n * 1e3,
    }
    Path(args.output).write_text(json.dumps({"summary": summary, "images": records}, indent=1))
    print(f"Détails : {args.output}")


if __name__ == "__main__":
    main()
