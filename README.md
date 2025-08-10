# EMNIST Handwritten Character Recognition
Projekt implementuje system rozpoznawania odręcznie rysowanych znaków (cyfr i liter) w oparciu o zbiór danych EMNIST Balanced oraz sieci konwolucyjne (CNN) w PyTorch.
Aplikacja udostępnia interfejs graficzny w technologii Gradio, umożliwiający użytkownikowi rysowanie znaku na panelu i uzyskanie predykcji w czasie rzeczywistym.

# Funkcjonalności
Trening modelu CNN na zbiorze EMNIST Balanced z augmentacją danych
Pipeline przetwarzania rysunku użytkownika: wykrycie obszaru znaku, normalizacja, skalowanie do 28×28 pikseli
Interfejs webowy w Gradio z obsługą rysowania w panelu canvas
Prezentacja Top-k predykcji z prawdopodobieństwami
Eksport modelu do formatu ONNX oraz kwantyzacja
Obsługa pustych lub niejednoznacznych rysunków
Konfigurowalne parametry treningu w plikach YAML

# Diagnostyka danych
Szybkie narzędzia podglądu i debugowania:

# Podgląd próbek po transformacjach
python notebooks/inspect_sample.py
python notebooks/inspect_preprocess.py
python scripts/dump_batch.py
data/debug/batch_grid.png
python scripts/label_stats.py


# Wymagania
Python 3.11+
PyTorch z obsługą CUDA 12.1 (opcjonalnie CPU)

# Plan rozwoju
 Implementacja podstawowej CNN
 Pipeline przetwarzania rysunku użytkownika
 Interfejs Gradio
 Eksport i kwantyzacja modelu
 Obsługa wielocyfrowych ciągów znaków
 Wersja webowa w TensorFlow.js
 Raport porównawczy wydajności modeli

# Licencja
Projekt udostępniony na licencji MIT.
