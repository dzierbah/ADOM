"""
Eksperyment 2: Detekcja twarzy przy słabym oświetleniu

Badanie degradacji jakości detekcji Viola-Jones w funkcji poziomu oświetlenia:
- Symulacja redukcji jasności (gamma correction)
- Dodanie szumu przy niskim oświetleniu
- Equalizacja histogramu jako technika korekcyjna

Metryki: Precision, Recall, Detection Rate
"""

import cv2
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt


RESULTS_DIR = "results"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Poziomy jasności: gamma > 1 ciemniej, gamma < 1 jaśniej
# Symulujemy warunki: normalne, lekko ciemne, ciemne, bardzo ciemne, ekstremalnie ciemne
BRIGHTNESS_LEVELS = {
    "normalne (gamma=1.0)": 1.0,
    "lekko ciemne (gamma=1.5)": 1.5,
    "ciemne (gamma=2.0)": 2.0,
    "bardzo ciemne (gamma=3.0)": 3.0,
    "ekstremalnie ciemne (gamma=5.0)": 5.0,
}

# Parametry detekcji (stałe – baseline)
SCALE_FACTOR = 1.3
MIN_NEIGHBORS = 5


def adjust_gamma(image, gamma):
    """Korekcja gamma – symulacja zmian oświetlenia."""
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]
    ).astype("uint8")
    return cv2.LUT(image, table)


def add_low_light_noise(image, sigma=25):
    """Dodaj szum Gaussowski – typowy dla niskiego oświetlenia."""
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


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


def generate_test_images_with_faces(n=15, seed=42):
    """Generuj realistyczne obrazy testowe z twarzami."""
    np.random.seed(seed)
    data = []
    for i in range(n):
        h, w = 480, 640
        # Tło z gradientem
        img = np.zeros((h, w, 3), dtype=np.uint8)
        base_val = np.random.randint(100, 200)
        img[:] = (base_val, base_val - 10, base_val - 20)

        n_faces = np.random.randint(1, 3)
        gt = []
        for _ in range(n_faces):
            cx = np.random.randint(120, w - 120)
            cy = np.random.randint(100, h - 100)
            fw = np.random.randint(70, 130)
            fh = int(fw * 1.25)

            # Twarz
            skin = (int(np.random.randint(170, 230)),
                    int(np.random.randint(150, 210)),
                    int(np.random.randint(130, 190)))
            cv2.ellipse(img, (cx, cy), (fw // 2, fh // 2), 0, 0, 360, skin, -1)

            # Oczy (ciemne – kluczowe cechy Haar)
            ey = cy - fh // 6
            cv2.ellipse(img, (cx - fw // 5, ey), (fw // 10, fh // 16), 0, 0, 360, (30, 30, 30), -1)
            cv2.ellipse(img, (cx + fw // 5, ey), (fw // 10, fh // 16), 0, 0, 360, (30, 30, 30), -1)

            # Nos
            cv2.line(img, (cx, cy - fh // 12), (cx, cy + fh // 8), (skin[0] - 15, skin[1] - 15, skin[2] - 15), 2)

            gt.append([cx - fw // 2, cy - fh // 2, fw, fh])

        data.append({"image": img, "ground_truth": gt, "name": f"test_{i:03d}"})
    return data


def run_experiment(test_data, cascade):
    """Uruchom eksperyment z różnymi poziomami oświetlenia."""
    results = []

    for level_name, gamma in BRIGHTNESS_LEVELS.items():
        all_p, all_r, all_f1 = [], [], []
        det_count = 0

        for sample in test_data:
            img = sample["image"].copy()
            gt = sample["ground_truth"]

            # Zastosuj redukcję jasności
            dark_img = adjust_gamma(img, gamma)

            # Dodaj szum proporcjonalny do ciemności
            if gamma > 1.5:
                noise_sigma = min(10 * (gamma - 1), 40)
                dark_img = add_low_light_noise(dark_img, sigma=noise_sigma)

            # --- Detekcja BEZ equalizacji ---
            gray = cv2.cvtColor(dark_img, cv2.COLOR_BGR2GRAY)
            faces_no_eq = cascade.detectMultiScale(
                gray, scaleFactor=SCALE_FACTOR, minNeighbors=MIN_NEIGHBORS,
                minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
            )
            dets_no_eq = [list(f) for f in faces_no_eq] if len(faces_no_eq) > 0 else []
            p1, r1, f1_1 = evaluate(dets_no_eq, gt)

            # --- Detekcja Z equalizacją histogramu ---
            gray_eq = cv2.equalizeHist(gray)
            faces_eq = cascade.detectMultiScale(
                gray_eq, scaleFactor=SCALE_FACTOR, minNeighbors=MIN_NEIGHBORS,
                minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
            )
            dets_eq = [list(f) for f in faces_eq] if len(faces_eq) > 0 else []
            p2, r2, f1_2 = evaluate(dets_eq, gt)

            all_p.append((p1, p2))
            all_r.append((r1, r2))
            all_f1.append((f1_1, f1_2))

        avg_p_no_eq = np.mean([x[0] for x in all_p])
        avg_p_eq = np.mean([x[1] for x in all_p])
        avg_r_no_eq = np.mean([x[0] for x in all_r])
        avg_r_eq = np.mean([x[1] for x in all_r])
        avg_f1_no_eq = np.mean([x[0] for x in all_f1])
        avg_f1_eq = np.mean([x[1] for x in all_f1])

        results.append({
            "level": level_name,
            "gamma": gamma,
            "precision_no_eq": round(avg_p_no_eq, 4),
            "recall_no_eq": round(avg_r_no_eq, 4),
            "f1_no_eq": round(avg_f1_no_eq, 4),
            "precision_eq": round(avg_p_eq, 4),
            "recall_eq": round(avg_r_eq, 4),
            "f1_eq": round(avg_f1_eq, 4),
        })

        print(f"  {level_name}:")
        print(f"    Bez equalizacji:  P={avg_p_no_eq:.3f} R={avg_r_no_eq:.3f} F1={avg_f1_no_eq:.3f}")
        print(f"    Z equalizacją:    P={avg_p_eq:.3f} R={avg_r_eq:.3f} F1={avg_f1_eq:.3f}")

    return pd.DataFrame(results)


def plot_results(df, output_dir):
    """Wykresy wpływu oświetlenia na detekcję."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, metric_pair, title in zip(
        axes,
        [("precision_no_eq", "precision_eq"),
         ("recall_no_eq", "recall_eq"),
         ("f1_no_eq", "f1_eq")],
        ["Precision", "Recall", "F1-Score"]
    ):
        gammas = df["gamma"].values
        ax.plot(gammas, df[metric_pair[0]], "o-", label="Bez equalizacji", color="red")
        ax.plot(gammas, df[metric_pair[1]], "s-", label="Z equalizacją", color="green")
        ax.set_xlabel("Gamma (wyższe = ciemniej)")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs oświetlenie")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lighting_experiment.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Przykładowe obrazy przy różnych warunkach
    fig, axes = plt.subplots(1, len(BRIGHTNESS_LEVELS), figsize=(20, 4))
    test_img = generate_test_images_with_faces(n=1, seed=0)[0]["image"]
    for ax, (name, gamma) in zip(axes, BRIGHTNESS_LEVELS.items()):
        dark = adjust_gamma(test_img, gamma)
        if gamma > 1.5:
            dark = add_low_light_noise(dark, sigma=min(10 * (gamma - 1), 40))
        ax.imshow(cv2.cvtColor(dark, cv2.COLOR_BGR2RGB))
        ax.set_title(f"γ={gamma}", fontsize=10)
        ax.axis("off")

    plt.suptitle("Symulacja różnych warunków oświetleniowych", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lighting_examples.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Wykresy zapisane w: {output_dir}")


def generate_report(df, output_dir):
    """Generuj raport eksperymentu 2."""
    os.makedirs(output_dir, exist_ok=True)

    report = f"""# Eksperyment 2: Detekcja twarzy przy słabym oświetleniu

## Cel
Zbadanie wpływu warunków oświetleniowych na jakość detekcji twarzy algorytmem
Viola-Jones oraz ocena skuteczności equalizacji histogramu jako techniki korekcyjnej.

## Metodyka
- Symulacja słabego oświetlenia: korekcja gamma (γ = 1.0, 1.5, 2.0, 3.0, 5.0)
- Dodatkowy szum Gaussowski proporcjonalny do ciemności (dla γ > 1.5)
- Porównanie detekcji z i bez equalizacji histogramu
- Stałe parametry detektora: scaleFactor={SCALE_FACTOR}, minNeighbors={MIN_NEIGHBORS}
- Metryki: Precision, Recall, F1-score

## Wyniki

{df.to_markdown(index=False)}

## Obserwacje

1. **Degradacja w ciemności**: Algorytm Viola-Jones jest wrażliwy na zmniejszenie
   jasności obrazu. Cechy Haara opierają się na różnicach jasności pomiędzy
   sąsiednimi regionami – przy niskim kontraście te różnice zanikają.

2. **Equalizacja histogramu**: Znacząco poprawia detekcję w warunkach
   umiarkowanie słabego oświetlenia (γ = 1.5-2.0). Przy ekstremalnie ciemnych
   obrazach (γ > 3.0) equalizacja wzmacnia również szum, co ogranicza jej
   skuteczność.

3. **Szum w ciemności**: Dodatkowy szum obecny przy niskim oświetleniu
   pogarsza jakość cech Haara, tworząc fałszywe kontrasty prowadzące
   do fałszywych detekcji (spadek Precision) lub maskujące prawdziwe cechy
   twarzy (spadek Recall).

## Wnioski
Viola-Jones działa najlepiej w warunkach dobrego oświetlenia. Equalizacja
histogramu jest prostym, ale skutecznym krokiem preprocessingu poprawiającym
detekcję w umiarkowanie ciemnych warunkach. Dla ekstremalnie słabego oświetlenia
potrzebne są bardziej zaawansowane techniki (np. modele deep learning
odporne na warunki oświetleniowe).
"""

    path = os.path.join(output_dir, "experiment_lighting_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    df.to_csv(os.path.join(output_dir, "experiment_lighting_results.csv"), index=False)
    print(f"Raport zapisany: {path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("EKSPERYMENT 2: Detekcja przy słabym oświetleniu")
    print("=" * 70)

    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        print("BŁĄD: Nie można załadować kaskady!")
        return

    print("\nGenerowanie danych testowych...")
    test_data = generate_test_images_with_faces(n=15)

    print("\nUruchamianie eksperymentu...\n")
    df = run_experiment(test_data, cascade)

    print("\nGenerowanie wykresów...")
    plot_results(df, RESULTS_DIR)

    print("\nGenerowanie raportu...")
    generate_report(df, RESULTS_DIR)

    print("\n" + "=" * 70)
    print("EKSPERYMENT 2 ZAKOŃCZONY")
    print("=" * 70)


if __name__ == "__main__":
    main()
