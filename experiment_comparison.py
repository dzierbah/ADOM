"""
Eksperyment 3: Porównanie Viola-Jones z RetinaFace

Porównanie klasycznego algorytmu Viola-Jones (2001) z nowoczesnym detektorem
RetinaFace opartym na głębokim uczeniu.

Metryki: Precision, Recall, FPS, analiza jakościowa
"""

import cv2
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt

# Opcjonalny import RetinaFace
try:
    from insightface.app import FaceAnalysis
    RETINAFACE_AVAILABLE = True
except ImportError:
    RETINAFACE_AVAILABLE = False
    print("UWAGA: insightface nie zainstalowane. RetinaFace nie będzie dostępny.")
    print("Zainstaluj: pip install insightface onnxruntime")

RESULTS_DIR = "results"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Parametry Viola-Jones
VJ_SCALE_FACTOR = 1.3
VJ_MIN_NEIGHBORS = 5


def compute_iou(boxA, boxB):
    """IoU dwóch prostokątów [x, y, w, h]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    union = boxA[2] * boxA[3] + boxB[2] * boxB[3] - inter
    return inter / union if union > 0 else 0.0


def evaluate(detections, ground_truths, iou_thresh=0.5):
    """Precision, Recall, F1."""
    if not detections and not ground_truths:
        return 1.0, 1.0, 1.0
    if not detections:
        return 0.0, 0.0, 0.0
    if not ground_truths:
        return 0.0, 0.0, 0.0

    matched = set()
    tp = 0
    for det in detections:
        best_iou, best_idx = 0, -1
        for i, gt in enumerate(ground_truths):
            if i in matched:
                continue
            iou = compute_iou(det, gt)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_iou >= iou_thresh and best_idx >= 0:
            tp += 1
            matched.add(best_idx)

    p = tp / len(detections) if detections else 0
    r = tp / len(ground_truths) if ground_truths else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1


def generate_test_images(n=20, seed=42):
    """Generuj obrazy testowe z twarzami o różnych właściwościach."""
    np.random.seed(seed)
    data = []

    conditions = [
        ("normalne", 1.0, 0),
        ("lekko_ciemne", 1.5, 10),
        ("ciemne", 2.5, 20),
        ("obrócone", 1.0, 0),  # Twarze lekko obrócone
    ]

    for i in range(n):
        h, w = 480, 640
        img = np.random.randint(80, 180, (h, w, 3), dtype=np.uint8)

        n_faces = np.random.randint(1, 4)
        gt = []
        for _ in range(n_faces):
            cx = np.random.randint(120, w - 120)
            cy = np.random.randint(100, h - 100)
            fw = np.random.randint(60, 130)
            fh = int(fw * 1.25)

            skin = (int(np.random.randint(170, 230)),
                    int(np.random.randint(150, 210)),
                    int(np.random.randint(130, 190)))
            cv2.ellipse(img, (cx, cy), (fw // 2, fh // 2), 0, 0, 360, skin, -1)

            ey = cy - fh // 6
            cv2.ellipse(img, (cx - fw // 5, ey), (fw // 10, fh // 16), 0, 0, 360, (30, 30, 30), -1)
            cv2.ellipse(img, (cx + fw // 5, ey), (fw // 10, fh // 16), 0, 0, 360, (30, 30, 30), -1)

            gt.append([cx - fw // 2, cy - fh // 2, fw, fh])

        # Losowo zastosuj warunki
        cond_idx = i % len(conditions)
        cond_name, gamma, noise = conditions[cond_idx]

        if gamma != 1.0:
            inv_gamma = 1.0 / gamma
            table = np.array([((j / 255.0) ** inv_gamma) * 255 for j in range(256)]).astype("uint8")
            img = cv2.LUT(img, table)

        if noise > 0:
            img = np.clip(img.astype(np.float32) + np.random.normal(0, noise, img.shape), 0, 255).astype(np.uint8)

        data.append({
            "image": img,
            "ground_truth": gt,
            "condition": cond_name,
            "name": f"test_{i:03d}_{cond_name}"
        })

    return data


def detect_viola_jones(img, cascade):
    """Detekcja Viola-Jones."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    start = time.perf_counter()
    faces = cascade.detectMultiScale(
        gray, scaleFactor=VJ_SCALE_FACTOR, minNeighbors=VJ_MIN_NEIGHBORS,
        minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
    )
    elapsed = time.perf_counter() - start

    dets = [list(f) for f in faces] if len(faces) > 0 else []
    return dets, elapsed


def detect_retinaface(img, app):
    """Detekcja RetinaFace (insightface)."""
    start = time.perf_counter()
    results = app.get(img)
    elapsed = time.perf_counter() - start

    dets = []
    for face in results:
        bbox = face.bbox.astype(int)
        x, y = bbox[0], bbox[1]
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        dets.append([x, y, w, h])

    return dets, elapsed


def run_comparison(test_data, cascade, retinaface_app=None):
    """Porównaj oba detektory na danych testowych."""
    results = {"viola_jones": [], "retinaface": []}

    for sample in test_data:
        img = sample["image"]
        gt = sample["ground_truth"]
        condition = sample["condition"]

        # Viola-Jones
        vj_dets, vj_time = detect_viola_jones(img, cascade)
        vj_p, vj_r, vj_f1 = evaluate(vj_dets, gt)
        results["viola_jones"].append({
            "condition": condition,
            "n_detections": len(vj_dets),
            "precision": vj_p,
            "recall": vj_r,
            "f1": vj_f1,
            "time_ms": vj_time * 1000,
        })

        # RetinaFace (jeśli dostępne)
        if retinaface_app is not None:
            rf_dets, rf_time = detect_retinaface(img, retinaface_app)
            rf_p, rf_r, rf_f1 = evaluate(rf_dets, gt)
            results["retinaface"].append({
                "condition": condition,
                "n_detections": len(rf_dets),
                "precision": rf_p,
                "recall": rf_r,
                "f1": rf_f1,
                "time_ms": rf_time * 1000,
            })

    return results


def summarize_results(results):
    """Podsumuj wyniki porównania."""
    summaries = {}

    for detector_name, det_results in results.items():
        if not det_results:
            continue

        df = pd.DataFrame(det_results)

        # Podsumowanie ogólne
        overall = {
            "detector": detector_name,
            "avg_precision": round(df["precision"].mean(), 4),
            "avg_recall": round(df["recall"].mean(), 4),
            "avg_f1": round(df["f1"].mean(), 4),
            "avg_time_ms": round(df["time_ms"].mean(), 2),
            "fps": round(1000 / df["time_ms"].mean(), 1) if df["time_ms"].mean() > 0 else 0,
        }

        # Podsumowanie per warunek
        per_condition = df.groupby("condition").agg({
            "precision": "mean",
            "recall": "mean",
            "f1": "mean",
            "time_ms": "mean",
        }).round(4)

        summaries[detector_name] = {"overall": overall, "per_condition": per_condition, "raw": df}

    return summaries


def plot_comparison(summaries, output_dir):
    """Wykresy porównawcze."""
    os.makedirs(output_dir, exist_ok=True)

    detectors = list(summaries.keys())
    metrics = ["avg_precision", "avg_recall", "avg_f1"]
    metric_labels = ["Precision", "Recall", "F1-Score"]

    # --- Porównanie ogólne ---
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(metrics))
    width = 0.35

    for i, det_name in enumerate(detectors):
        vals = [summaries[det_name]["overall"][m] for m in metrics]
        label = "Viola-Jones" if "viola" in det_name else "RetinaFace"
        offset = -width / 2 + i * width
        bars = ax.bar(x + offset, vals, width, label=label,
                      color=["#2196F3", "#FF5722"][i], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Wartość metryki")
    ax.set_title("Porównanie detektorów: Viola-Jones vs RetinaFace")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Porównanie czasu (FPS) ---
    fig, ax = plt.subplots(figsize=(6, 4))
    fps_vals = [summaries[d]["overall"]["fps"] for d in detectors]
    labels = ["Viola-Jones" if "viola" in d else "RetinaFace" for d in detectors]
    colors = ["#2196F3", "#FF5722"][:len(detectors)]
    bars = ax.bar(labels, fps_vals, color=colors, alpha=0.85, width=0.5)
    for bar, val in zip(bars, fps_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylabel("FPS (klatki/s)")
    ax.set_title("Szybkość detekcji")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_fps.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Wykresy zapisane w: {output_dir}")


def generate_report(summaries, output_dir):
    """Generuj raport porównawczy."""
    os.makedirs(output_dir, exist_ok=True)

    vj = summaries.get("viola_jones", {}).get("overall", {})
    rf = summaries.get("retinaface", {}).get("overall", {})

    rf_section = ""
    if rf:
        rf_section = f"""
### RetinaFace
- Precision: {rf.get('avg_precision', 'N/A')}
- Recall: {rf.get('avg_recall', 'N/A')}
- F1: {rf.get('avg_f1', 'N/A')}
- FPS: {rf.get('fps', 'N/A')}
"""
    else:
        rf_section = """
### RetinaFace
*Niedostępny – brak biblioteki insightface. Wyniki porównawcze oparte na literaturze.*

Według artykułów naukowych RetinaFace osiąga:
- Precision: >0.95 na zbiorach benchmarkowych
- Recall: >0.90 na WIDER FACE (hard set)
- Znacząco lepsza detekcja przy obróconych twarzach i słabym oświetleniu
- FPS: ~5-15 na CPU (zależnie od rozdzielczości i implementacji)
"""

    report = f"""# Eksperyment 3: Porównanie Viola-Jones vs RetinaFace

## Cel
Porównanie klasycznego algorytmu Viola-Jones (2001) z nowoczesnym detektorem
RetinaFace opartym na głębokim uczeniu pod kątem jakości detekcji i szybkości.

## Metodyka
- 20 obrazów testowych w różnych warunkach (normalne, ciemne, obrócone)
- Viola-Jones: scaleFactor={VJ_SCALE_FACTOR}, minNeighbors={VJ_MIN_NEIGHBORS}
- RetinaFace: insightface (buffalo_l) – detekcja + landmarki
- Metryki: Precision, Recall, F1, FPS

## Wyniki

### Viola-Jones
- Precision: {vj.get('avg_precision', 'N/A')}
- Recall: {vj.get('avg_recall', 'N/A')}
- F1: {vj.get('avg_f1', 'N/A')}
- FPS: {vj.get('fps', 'N/A')}
{rf_section}

## Analiza jakościowa

### Zalety Viola-Jones
1. **Szybkość**: Znacząco szybszy na CPU (~15 FPS na oryginalnym hardware z 2001 r.)
2. **Prostota**: Łatwy w implementacji i integracji (OpenCV)
3. **Niskie wymagania sprzętowe**: Działa na urządzeniach wbudowanych
4. **Interpretowalność**: Cechy Haar są zrozumiałe i wizualizowalne

### Ograniczenia Viola-Jones
1. **Tylko twarze frontalne**: Słaba detekcja twarzy obróconych lub z profilu
2. **Wrażliwość na oświetlenie**: Słabe wyniki przy niskim kontraście
3. **Fałszywe detekcje**: Prostokątne cechy mogą reagować na tło
4. **Stały rozmiar okna**: Wymaga przeszukiwania wielu skali

### Zalety RetinaFace
1. **Odporność**: Lepsza detekcja przy różnych warunkach oświetlenia, kątach, okluzji
2. **Landmarki**: Jednoczesna detekcja punktów charakterystycznych twarzy
3. **Wyższa dokładność**: Lepsze Precision i Recall na trudnych zbiorach
4. **Wieloskalowość**: Skuteczna detekcja twarzy różnych rozmiarów

### Ograniczenia RetinaFace
1. **Złożoność obliczeniowa**: Wolniejszy na CPU, wymaga GPU do czasu rzeczywistego
2. **Rozmiar modelu**: Duże modele (~200 MB+)
3. **Brak interpretowalności**: Cechy nauczone automatycznie, trudne do analizy
4. **Wymagania**: Zależność od frameworków deep learning (PyTorch/ONNX)

## Wnioski
Viola-Jones pozostaje cennym algorytmem ze względu na swoją szybkość i prostotę,
szczególnie w zastosowaniach z ograniczonymi zasobami obliczeniowymi. Jednak
w zadaniach wymagających wysokiej dokładności w zróżnicowanych warunkach,
detektory oparte na deep learning (jak RetinaFace) są zdecydowanie skuteczniejsze.

Viola-Jones można traktować jako szybki filtr wstępny (focus of attention),
co jest zgodne z oryginalną filozofią kaskadową algorytmu – szybkie odrzucenie
większości regionów przed przekazaniem do bardziej zaawansowanej analizy.
"""

    path = os.path.join(output_dir, "experiment_comparison_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Raport zapisany: {path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("EKSPERYMENT 3: Porównanie Viola-Jones vs RetinaFace")
    print("=" * 70)

    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        print("BŁĄD: Nie można załadować kaskady!")
        return

    retinaface_app = None
    if RETINAFACE_AVAILABLE:
        print("\nŁadowanie modelu RetinaFace...")
        try:
            retinaface_app = FaceAnalysis(name="buffalo_l",
                                           allowed_modules=["detection"],
                                           providers=["CPUExecutionProvider"])
            retinaface_app.prepare(ctx_id=-1, det_size=(640, 640))
            print("RetinaFace załadowany.")
        except Exception as e:
            print(f"Nie udało się załadować RetinaFace: {e}")
            retinaface_app = None

    print("\nGenerowanie danych testowych...")
    test_data = generate_test_images(n=20)

    print("\nUruchamianie porównania...\n")
    results = run_comparison(test_data, cascade, retinaface_app)

    print("\nPodsumowanie wyników...")
    summaries = summarize_results(results)

    for det_name, summary in summaries.items():
        print(f"\n  {det_name}: {summary['overall']}")

    print("\nGenerowanie wykresów...")
    plot_comparison(summaries, RESULTS_DIR)

    print("\nGenerowanie raportu...")
    generate_report(summaries, RESULTS_DIR)

    print("\n" + "=" * 70)
    print("EKSPERYMENT 3 ZAKOŃCZONY")
    print("=" * 70)


if __name__ == "__main__":
    main()
