# Temat 2: Detekcja twarzy – Viola–Jones (2001)

## Opis projektu

Projekt realizowany w ramach przedmiotu **Analiza Danych Obrazowych i Multimedialnych (ADOM 26L)** – zima 2025/26, prowadzący: dr hab. inż. Marcin Iwanowski.

Celem projektu jest dogłębna analiza algorytmu **Viola–Jones** do detekcji twarzy w czasie rzeczywistym, opisanego w artykule:

> P. Viola, M. Jones, *"Rapid Object Detection using a Boosted Cascade of Simple Features"*, CVPR 2001.

Algorytm wykorzystuje:
- **Cechy Haara** (Haar-like features) – proste cechy prostokątne opisujące lokalne kontrasty jasności
- **Obraz całkowy** (Integral Image) – reprezentacja umożliwiająca szybkie obliczanie cech
- **AdaBoost** – algorytm boostingu do selekcji najistotniejszych cech i budowy klasyfikatora
- **Kaskadę klasyfikatorów** – strukturę pozwalającą szybko odrzucać regiony niezawierające twarzy

## Instalacja

### Wymagania systemowe
- Python 3.8+
- Kamera internetowa (dla demo w czasie rzeczywistym)

### Instalacja zależności

```bash
pip install -r requirements.txt
```

### Weryfikacja instalacji

```bash
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "from insightface.app import FaceAnalysis; print('RetinaFace: OK')"
```

## Uruchomienie

### Demo – detekcja z kamery

```bash
python scripts/demo_camera.py
```

Klawisze sterujące:
- `q` – wyjście
- `+`/`-` – zmiana scaleFactor
- `[`/`]` – zmiana minNeighbors

### Demo – detekcja na obrazie

```bash
python scripts/demo_image.py --image data/test_image.jpg
```

### Notebook demonstracyjny

```bash
jupyter notebook notebooks/viola_jones_demo.ipynb
```

### Eksperymenty

```bash
# Eksperyment 1: Wpływ scaleFactor i minNeighbors
python scripts/experiment_params.py

# Eksperyment 2: Wpływ słabego oświetlenia
python scripts/experiment_lighting.py

# Eksperyment 3: Porównanie z RetinaFace
python scripts/experiment_comparison.py
```

## Struktura repozytorium

```
viola-jones-project/
├── README.md                          # Ten plik
├── requirements.txt                   # Zależności Python
├── notebooks/
│   └── viola_jones_demo.ipynb         # Notebook demonstracyjny
├── scripts/
│   ├── demo_camera.py                 # Demo z kamery
│   ├── demo_image.py                  # Demo na obrazie
│   ├── experiment_params.py           # Eksp. 1: parametry scaleFactor/minNeighbors
│   ├── experiment_lighting.py         # Eksp. 2: słabe oświetlenie
│   └── experiment_comparison.py       # Eksp. 3: porównanie z RetinaFace
├── results/
│   ├── experiment_params_report.md    # Raport eksperymentu 1
│   ├── experiment_lighting_report.md  # Raport eksperymentu 2
│   └── experiment_comparison_report.md# Raport eksperymentu 3
├── data/                              # Dane testowe
└── models/                            # Modele kaskadowe OpenCV
```

## Eksperymenty

### Eksperyment 1: Wpływ parametrów scaleFactor i minNeighbors
Badanie wpływu parametrów `scaleFactor` (1.01–1.5) i `minNeighbors` (1–10) na jakość detekcji. Siatka minimum 20 kombinacji parametrów. Metryki: Precision, Recall, F1-score, FPS.

### Eksperyment 2: Detekcja przy słabym oświetleniu
Symulacja warunków słabego oświetlenia poprzez redukcję jasności i kontrastu obrazów testowych. Badanie degradacji jakości detekcji w funkcji poziomu oświetlenia. Metryki: Precision, Recall, Detection Rate.

### Eksperyment 3: Porównanie z RetinaFace
Porównanie klasycznego algorytmu Viola–Jones z nowoczesnym detektorem RetinaFace (opartym na głębokim uczeniu). Metryki: Precision, Recall, FPS, analiza jakościowa.

## Zespół

| Rola | Osoba | Odpowiedzialność |
|------|-------|------------------|
| Theory Lead | [Imię Nazwisko] | Analiza artykułu, formalizacja matematyczna, ograniczenia, prezentacja krótka |
| Implementation Lead | [Imię Nazwisko] | Uruchomienie repo, refaktoryzacja, środowisko |
| Experiments Lead | [Imię Nazwisko] | Projekt eksperymentów, metryki |
| Presentation/Demo Lead | [Imię Nazwisko] | Demo, integracja kodu, prezentacja długa |

## Źródła

- Artykuł: [Viola & Jones, CVPR 2001](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf)
- Repozytorium bazowe: [OpenCV Cascade Classifier Tutorial](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)
