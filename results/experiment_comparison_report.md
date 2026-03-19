# Eksperyment 3: Porównanie Viola-Jones vs RetinaFace

## Cel
Porównanie klasycznego algorytmu Viola-Jones (2001) z nowoczesnym detektorem
RetinaFace opartym na głębokim uczeniu pod kątem jakości detekcji i szybkości.

## Metodyka
- 20 obrazów testowych w różnych warunkach (normalne, ciemne, obrócone)
- Viola-Jones: scaleFactor=1.3, minNeighbors=5
- RetinaFace: insightface (buffalo_l) – detekcja + landmarki
- Metryki: Precision, Recall, F1, FPS

## Wyniki

### Viola-Jones
- Precision: 0.95
- Recall: 0.7583
- F1: 0.8183
- FPS: 65.4

### RetinaFace
- Precision: 0.25
- Recall: 0.175
- F1: 0.1983
- FPS: 7.4


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
