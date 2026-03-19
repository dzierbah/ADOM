"""
Demo: Detekcja twarzy z kamery w czasie rzeczywistym (Viola-Jones)

Sterowanie:
    q       - wyjście
    +/-     - zmiana scaleFactor (krok 0.05)
    [/]     - zmiana minNeighbors (krok 1)
    s       - zapisz bieżącą klatkę
"""

import cv2
import time
import os
import argparse


def load_cascade(cascade_path=None):
    """Załaduj klasyfikator kaskadowy Haar."""
    if cascade_path is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise FileNotFoundError(f"Nie można załadować kaskady: {cascade_path}")
    return cascade


def detect_faces(frame, cascade, scale_factor=1.3, min_neighbors=5, min_size=(30, 30)):
    """
    Wykryj twarze na klatce.

    Parametry:
        frame: obraz BGR
        cascade: klasyfikator kaskadowy
        scale_factor: współczynnik skalowania piramidy obrazów
        min_neighbors: minimalna liczba sąsiednich detekcji
        min_size: minimalny rozmiar twarzy (szer, wys)

    Zwraca:
        faces: lista prostokątów (x, y, w, h)
        elapsed: czas detekcji w sekundach
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    start = time.perf_counter()
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    elapsed = time.perf_counter() - start

    return faces, elapsed


def draw_detections(frame, faces, scale_factor, min_neighbors, fps, det_time_ms):
    """Narysuj ramki detekcji i informacje na klatce."""
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Panel informacyjny
    info_lines = [
        f"scaleFactor: {scale_factor:.2f} (+/-)",
        f"minNeighbors: {min_neighbors} ([/])",
        f"Twarze: {len(faces)}",
        f"Det. czas: {det_time_ms:.1f} ms",
        f"FPS: {fps:.1f}",
    ]

    y_offset = 25
    for line in info_lines:
        cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                     0.6, (0, 255, 0), 2)
        y_offset += 25

    return frame


def main():
    parser = argparse.ArgumentParser(description="Viola-Jones face detection demo (camera)")
    parser.add_argument("--cascade", type=str, default=None, help="Ścieżka do pliku kaskady XML")
    parser.add_argument("--camera", type=int, default=0, help="Indeks kamery")
    parser.add_argument("--width", type=int, default=640, help="Szerokość klatki")
    parser.add_argument("--height", type=int, default=480, help="Wysokość klatki")
    args = parser.parse_args()

    cascade = load_cascade(args.cascade)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print("Nie można otworzyć kamery!")
        return

    scale_factor = 1.30
    min_neighbors = 5
    fps = 0.0
    frame_count = 0
    fps_start = time.perf_counter()

    print("Demo Viola-Jones – detekcja twarzy z kamery")
    print("Klawisze: q=wyjście, +/-=scaleFactor, [/]=minNeighbors, s=zapisz klatkę")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces, det_time = detect_faces(frame, cascade, scale_factor, min_neighbors)
        det_time_ms = det_time * 1000

        # Oblicz FPS
        frame_count += 1
        elapsed = time.perf_counter() - fps_start
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_start = time.perf_counter()

        display = draw_detections(frame.copy(), faces, scale_factor, min_neighbors, fps, det_time_ms)
        cv2.imshow("Viola-Jones Face Detection", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("+") or key == ord("="):
            scale_factor = min(scale_factor + 0.05, 2.0)
        elif key == ord("-"):
            scale_factor = max(scale_factor - 0.05, 1.01)
        elif key == ord("]"):
            min_neighbors = min(min_neighbors + 1, 20)
        elif key == ord("["):
            min_neighbors = max(min_neighbors - 1, 0)
        elif key == ord("s"):
            fname = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(fname, display)
            print(f"Zapisano: {fname}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
