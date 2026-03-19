"""
Eksperyment 1: Wpływ parametrów scaleFactor i minNeighbors

Badanie siatki parametrów (min. 20 kombinacji) i ich wpływu na:
- Precision, Recall, F1-score
- FPS (klatki na sekundę)

Wymaga: zbiór danych z anotacjami twarzy (np. WIDER FACE subset lub własne anotacje).
Dla celów demo generuje syntetyczne dane testowe z OpenCV.
"""

import cv2
import numpy as np
import pandas as pd
import time
import os
import json
import matplotlib.pyplot as plt
from itertools import product


# =============================================================================
# Konfiguracja eksperymentu
# =============================================================================

SCALE_FACTORS = [1.01, 1.05, 1.1, 1.2, 1.3, 1.5]
MIN_NEIGHBORS_LIST = [1, 2, 3, 5, 7, 10]

# Siatka: 6 x 6 = 36 kombinacji (> 20 wymaganych)

RESULTS_DIR = "results"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def compute_iou(boxA, boxB):
    """Oblicz Intersection over Union (IoU) dwóch prostokątów [x,y,w,h]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]
    union_area = boxA_area + boxB_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def evaluate_detections(detections, ground_truths, iou_threshold=0.5):
    """
    Oblicz Precision i Recall na podstawie detekcji i anotacji.

    Parametry:
        detections: lista list [x, y, w, h] – detekcje
        ground_truths: lista list [x, y, w, h] – prawdziwe twarze
        iou_threshold: próg IoU do uznania detekcji za poprawną

    Zwraca:
        precision, recall, f1
    """
    if len(detections) == 0 and len(ground_truths) == 0:
        return 1.0, 1.0, 1.0
    if len(detections) == 0:
        return 0.0, 0.0, 0.0
    if len(ground_truths) == 0:
        return 0.0, 0.0, 0.0

    matched_gt = set()
    tp = 0

    for det in detections:
        best_iou = 0
        best_gt_idx = -1
        for i, gt in enumerate(ground_truths):
            if i in matched_gt:
                continue
            iou = compute_iou(det, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)

    fp = len(detections) - tp
    fn = len(ground_truths) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def generate_synthetic_test_data(n_images=20, seed=42):
    """
    Generuj syntetyczne obrazy testowe z twarzami (przy użyciu OpenCV do detekcji
    na wbudowanych przykładach lub generowanie prostych obrazów z elipsami).

    Dla pełnej wersji eksperymentu należy użyć zbioru WIDER FACE lub FDDB.
    """
    np.random.seed(seed)
    test_data = []

    for i in range(n_images):
        h, w = 480, 640
        img = np.random.randint(60, 180, (h, w, 3), dtype=np.uint8)

        # Dodaj syntetyczną "twarz" (jasna elipsa)
        n_faces = np.random.randint(1, 4)
        gt_boxes = []
        for _ in range(n_faces):
            cx = np.random.randint(100, w - 100)
            cy = np.random.randint(80, h - 80)
            fw = np.random.randint(60, 120)
            fh = int(fw * 1.2)

            # Narysuj owalną twarz z podstawowymi cechami
            skin_color = (np.random.randint(160, 220),
                          np.random.randint(140, 200),
                          np.random.randint(120, 180))
            cv2.ellipse(img, (cx, cy), (fw // 2, fh // 2), 0, 0, 360, skin_color, -1)

            # Oczy (ciemniejsze niż policzki – kluczowa cecha Haar)
            eye_y = cy - fh // 6
            cv2.ellipse(img, (cx - fw // 5, eye_y), (fw // 10, fh // 16), 0, 0, 360, (40, 40, 40), -1)
            cv2.ellipse(img, (cx + fw // 5, eye_y), (fw // 10, fh // 16), 0, 0, 360, (40, 40, 40), -1)

            # Nos (jaśniejszy od oczu)
            nose_color = tuple(min(c + 20, 255) for c in skin_color)
            cv2.ellipse(img, (cx, cy + fh // 10), (fw // 12, fh // 10), 0, 0, 360, nose_color, -1)

            gt_boxes.append([cx - fw // 2, cy - fh // 2, fw, fh])

        test_data.append({"image": img, "ground_truth": gt_boxes, "name": f"synthetic_{i:03d}"})

    return test_data


def run_experiment(test_data, cascade):
    """Uruchom siatkę parametrów na danych testowych."""
    results = []

    total_combos = len(SCALE_FACTORS) * len(MIN_NEIGHBORS_LIST)
    print(f"Liczba kombinacji parametrów: {total_combos}")
    print(f"Liczba obrazów testowych: {len(test_data)}")
    print("-" * 70)

    for sf, mn in product(SCALE_FACTORS, MIN_NEIGHBORS_LIST):
        all_precision, all_recall, all_f1 = [], [], []
        total_det_time = 0
        total_frames = 0

        for sample in test_data:
            img = sample["image"]
            gt = sample["ground_truth"]

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            start = time.perf_counter()
            faces = cascade.detectMultiScale(
                gray, scaleFactor=sf, minNeighbors=mn,
                minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
            )
            elapsed = time.perf_counter() - start

            total_det_time += elapsed
            total_frames += 1

            dets = [list(f) for f in faces] if len(faces) > 0 else []
            p, r, f1 = evaluate_detections(dets, gt)
            all_precision.append(p)
            all_recall.append(r)
            all_f1.append(f1)

        avg_p = np.mean(all_precision)
        avg_r = np.mean(all_recall)
        avg_f1 = np.mean(all_f1)
        fps = total_frames / total_det_time if total_det_time > 0 else 0

        results.append({
            "scaleFactor": sf,
            "minNeighbors": mn,
            "precision": round(avg_p, 4),
            "recall": round(avg_r, 4),
            "f1_score": round(avg_f1, 4),
            "fps": round(fps, 1),
            "avg_det_time_ms": round((total_det_time / total_frames) * 1000, 2),
        })

        print(f"  scaleFactor={sf:.2f}, minNeighbors={mn:2d} -> "
              f"P={avg_p:.3f} R={avg_r:.3f} F1={avg_f1:.3f} FPS={fps:.1f}")

    return pd.DataFrame(results)


def plot_results(df, output_dir):
    """Generuj wykresy wyników eksperymentu."""
    os.makedirs(output_dir, exist_ok=True)

    # --- Heatmapa F1-score ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, metric, title in zip(
        axes,
        ["precision", "recall", "f1_score"],
        ["Precision", "Recall", "F1-Score"]
    ):
        pivot = df.pivot(index="minNeighbors", columns="scaleFactor", values=metric)
        im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns], rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("scaleFactor")
        ax.set_ylabel("minNeighbors")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Wartości na heatmapie
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, f"{pivot.values[i, j]:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if pivot.values[i, j] > 0.5 else "black")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- FPS vs scaleFactor ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for mn in df["minNeighbors"].unique():
        subset = df[df["minNeighbors"] == mn]
        ax.plot(subset["scaleFactor"], subset["fps"], marker="o", label=f"minNeighbors={mn}")
    ax.set_xlabel("scaleFactor")
    ax.set_ylabel("FPS")
    ax.set_title("Szybkość detekcji vs scaleFactor")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fps_vs_scalefactor.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- F1 vs FPS tradeoff ---
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(df["fps"], df["f1_score"],
                         c=df["scaleFactor"], cmap="viridis", s=80, edgecolors="black")
    ax.set_xlabel("FPS")
    ax.set_ylabel("F1-Score")
    ax.set_title("Kompromis: Jakość (F1) vs Szybkość (FPS)")
    plt.colorbar(scatter, ax=ax, label="scaleFactor")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f1_vs_fps_tradeoff.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Wykresy zapisane w: {output_dir}")


def generate_report(df, output_dir):
    """Generuj raport tekstowy z wynikami eksperymentu."""
    os.makedirs(output_dir, exist_ok=True)

    best_f1 = df.loc[df["f1_score"].idxmax()]
    best_fps = df.loc[df["fps"].idxmax()]

    report = f"""# Eksperyment 1: Wpływ parametrów scaleFactor i minNeighbors

## Cel
Zbadanie wpływu parametrów `scaleFactor` i `minNeighbors` detektora Viola-Jones
na jakość detekcji twarzy (Precision, Recall, F1) oraz szybkość działania (FPS).

## Metodyka
- Siatka parametrów: {len(SCALE_FACTORS)} wartości scaleFactor × {len(MIN_NEIGHBORS_LIST)} wartości minNeighbors = {len(SCALE_FACTORS) * len(MIN_NEIGHBORS_LIST)} kombinacji
- scaleFactor: {SCALE_FACTORS}
- minNeighbors: {MIN_NEIGHBORS_LIST}
- Klasyfikator: haarcascade_frontalface_default.xml (OpenCV)
- Metryki: Precision, Recall, F1-score, FPS
- Próg IoU: 0.5

## Wyniki

### Najlepsza kombinacja wg F1-score
- scaleFactor = {best_f1['scaleFactor']:.2f}
- minNeighbors = {int(best_f1['minNeighbors'])}
- Precision = {best_f1['precision']:.4f}
- Recall = {best_f1['recall']:.4f}
- F1 = {best_f1['f1_score']:.4f}
- FPS = {best_f1['fps']:.1f}

### Najszybsza kombinacja
- scaleFactor = {best_fps['scaleFactor']:.2f}
- minNeighbors = {int(best_fps['minNeighbors'])}
- F1 = {best_fps['f1_score']:.4f}
- FPS = {best_fps['fps']:.1f}

### Tabela wyników

{df.to_markdown(index=False)}

## Obserwacje

1. **scaleFactor**: Wyższe wartości (np. 1.3-1.5) przyspieszają detekcję (wyższy FPS),
   ale mogą pomijać twarze o rozmiarach pomiędzy skalami, co obniża Recall.
   Niższe wartości (np. 1.01-1.05) dają dokładniejsze przeszukiwanie skali, ale
   znacząco spowalniają detekcję.

2. **minNeighbors**: Niższe wartości (1-2) dają wyższy Recall kosztem Precision
   (więcej fałszywych detekcji). Wyższe wartości (7-10) poprawiają Precision,
   ale mogą obniżyć Recall, odrzucając prawdziwe twarze.

3. **Kompromis jakość-szybkość**: Istnieje wyraźny kompromis pomiędzy dokładnością
   a szybkością. Dla zastosowań czasu rzeczywistego scaleFactor=1.2-1.3 z
   minNeighbors=3-5 oferuje dobry balans.

## Wnioski
Parametry `scaleFactor` i `minNeighbors` istotnie wpływają na zachowanie detektora
Viola-Jones. Dobór parametrów zależy od zastosowania: aplikacje czasu rzeczywistego
preferują wyższy scaleFactor, podczas gdy zadania wymagające wysokiej czułości
powinny stosować niższy scaleFactor i minNeighbors.
"""

    report_path = os.path.join(output_dir, "experiment_params_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    csv_path = os.path.join(output_dir, "experiment_params_results.csv")
    df.to_csv(csv_path, index=False)

    print(f"Raport zapisany: {report_path}")
    print(f"Wyniki CSV: {csv_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("EKSPERYMENT 1: Wpływ scaleFactor i minNeighbors")
    print("=" * 70)

    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        print("BŁĄD: Nie można załadować kaskady!")
        return

    print("\nGenerowanie danych testowych...")
    test_data = generate_synthetic_test_data(n_images=20)

    print("\nUruchamianie siatki parametrów...\n")
    df = run_experiment(test_data, cascade)

    print("\nGenerowanie wykresów...")
    plot_results(df, RESULTS_DIR)

    print("\nGenerowanie raportu...")
    generate_report(df, RESULTS_DIR)

    print("\n" + "=" * 70)
    print("EKSPERYMENT 1 ZAKOŃCZONY")
    print("=" * 70)


if __name__ == "__main__":
    main()
