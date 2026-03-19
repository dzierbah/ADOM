"""
Demo: Detekcja twarzy na pojedynczym obrazie (Viola-Jones)

Użycie:
    python scripts/demo_image.py --image data/test_image.jpg
    python scripts/demo_image.py --image data/test_image.jpg --scale 1.1 --neighbors 3
"""

import cv2
import argparse
import time
import os


def main():
    parser = argparse.ArgumentParser(description="Viola-Jones face detection on image")
    parser.add_argument("--image", type=str, required=True, help="Ścieżka do obrazu")
    parser.add_argument("--cascade", type=str, default=None, help="Ścieżka do kaskady XML")
    parser.add_argument("--scale", type=float, default=1.3, help="scaleFactor")
    parser.add_argument("--neighbors", type=int, default=5, help="minNeighbors")
    parser.add_argument("--output", type=str, default=None, help="Ścieżka do zapisu wyniku")
    args = parser.parse_args()

    # Załaduj obraz
    img = cv2.imread(args.image)
    if img is None:
        print(f"Nie można załadować obrazu: {args.image}")
        return

    # Załaduj kaskadę
    cascade_path = args.cascade or (cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        print(f"Nie można załadować kaskady: {cascade_path}")
        return

    # Konwersja do skali szarości i equalizacja histogramu
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Detekcja
    start = time.perf_counter()
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=args.scale,
        minNeighbors=args.neighbors,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    elapsed = time.perf_counter() - start

    # Rysowanie detekcji
    result = img.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(result, "face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                     0.6, (0, 255, 0), 2)

    # Informacje
    h_img, w_img = img.shape[:2]
    info = (
        f"Obraz: {os.path.basename(args.image)} ({w_img}x{h_img}) | "
        f"scaleFactor={args.scale} | minNeighbors={args.neighbors} | "
        f"Wykryte twarze: {len(faces)} | Czas: {elapsed*1000:.1f} ms"
    )
    print(info)

    # Zapis lub wyświetlenie
    if args.output:
        cv2.imwrite(args.output, result)
        print(f"Zapisano wynik: {args.output}")
    else:
        cv2.imshow("Viola-Jones Detection", result)
        print("Naciśnij dowolny klawisz, aby zamknąć okno.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
