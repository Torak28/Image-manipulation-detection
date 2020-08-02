# Raport 02.08.2020

SVM wykonany na zbiorze danych Casia

## Założone stałe dotyczące zdjęcia

Zgodnie z Pana wytycznymi posłużyłem się bazą danych CASIA w wersji ITDE V2.0, o której więcej można przeczytać 

```
Dong, J., Wang, W., & Tan, T. (2013). CASIA Image Tampering Detection Evaluation Database. 2013 IEEE China Summit and International Conference on Signal and Information Processing.
```

W dużym skrócie baza zawiera ok. 7 500 zdjęć 'oryginalnych' i ok. 5 100 zdjęć 'przerobionych'. Przy czym zdjęcia te sa w różnej rozdzielczości od 320 x 240 do 800 x 600.

![alt](statystyki_zbioru_danych.png?raw=true)

W związku z faktem że w dalszych etapach będę ze zdjęć wyciągał informację, przyjąłem że zmienię ich rozmiar do 128 x 128

> Pytanie: Czy przyjęcie takiego rozmiaru ma sens? Czy zdjęcia nie są w tym wypadku za małe? Albo inaczej - czy dobór wielkości zdjęcia wynika z połączenia intuicji i doświadczenia, czy jest w tym wyborze coś jeszcze?

## Wyciąganie informacji ze zdjęcia

Do wyciągnięcia informacji ze zdjęcia posłużyłem się funkcjami `HuMoments()`, `calcHist()` z biblioteki `cv2`, funkcją `haralick()` z biblioteki `mahotas` i w jednym z przypadków - funkcją `hog` z biblioteki `skimage`.

Szczególnie zwracam uwagę na funkcję `hog`, która dość znacząco zwiększa wielkość ostateczną wektora, który opisuje konkretne zdjęcie(tworzę go poprzez zwykłego `hstack()` z cech zdjęcia).

Wektor zdjęcia z przeliczonym parametrem HOG posiada 856 cech, przy 532, gdy HOGa nie ma. 

> Pytanie: Przeglądałem internet w poszukiwaniu sposobów kategoryzacji zdjęć przy pomocy klasycznych algorytmów uczenia maszynowego. Rozumiem, że nie istnieje uniwersalna metoda wyciągania cech ze zdjęcia. Przyznam jednak, że skorzystałem z wymienionych powyżej funkcji dlatego że miały dla mnie wizualny sens. Czy powinienem użyć czegoś innego?

A! Może jeszcze dodam, że wykonałem `MinMaxScaler` na każdym wektorze.

## Podział danych

Zgodnie z materiałami które mi Pan polecił skorzystałem z `StratifiedKFold` jako że liczba obiektów w każdej z klas jest różna.