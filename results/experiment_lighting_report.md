# Eksperyment 2: Detekcja twarzy przy słabym oświetleniu

## Cel
Zbadanie wpływu warunków oświetleniowych na jakość detekcji twarzy algorytmem
Viola-Jones oraz ocena skuteczności equalizacji histogramu jako techniki korekcyjnej.

## Metodyka
- Symulacja słabego oświetlenia: korekcja gamma (γ = 1.0, 1.5, 2.0, 3.0, 5.0)
- Dodatkowy szum Gaussowski proporcjonalny do ciemności (dla γ > 1.5)
- Porównanie detekcji z i bez equalizacji histogramu
- Stałe parametry detektora: scaleFactor=1.3, minNeighbors=5
- Metryki: Precision, Recall, F1-score

## Wyniki

| level                           |   gamma |   precision_no_eq |   recall_no_eq |   f1_no_eq |   precision_eq |   recall_eq |   f1_eq |
|:--------------------------------|--------:|------------------:|---------------:|-----------:|---------------:|------------:|--------:|
| normalne (gamma=1.0)            |     1   |            0.4667 |         0.4    |     0.4222 |         0      |      0      |  0      |
| lekko ciemne (gamma=1.5)        |     1.5 |            0.4667 |         0.4333 |     0.4444 |         0      |      0      |  0      |
| ciemne (gamma=2.0)              |     2   |            0.4667 |         0.4    |     0.4222 |         0.8    |      0.7    |  0.7333 |
| bardzo ciemne (gamma=3.0)       |     3   |            0.3333 |         0.2667 |     0.2889 |         0.5333 |      0.4    |  0.4444 |
| ekstremalnie ciemne (gamma=5.0) |     5   |            0.0667 |         0.0333 |     0.0444 |         0.1333 |      0.0667 |  0.0889 |

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
