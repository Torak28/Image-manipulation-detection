# Raport 03.08.2020

SVM wykonany na zbiorze danych Casia

Przygotowane notebooki:

 - [SVM - Casia.ipynb](https://github.com/Torak28/Image-manipulation-detection/blob/master/PoC%20-%20SVM/Casia/SVM%20-%20Casia.ipynb)
 - [SVM - Casia_256.ipynb](https://github.com/Torak28/Image-manipulation-detection/blob/master/PoC%20-%20SVM/Casia/SVM%20-%20Casia_256.ipynb)
 - [SVM - Casia - skalar first.ipynb](https://github.com/Torak28/Image-manipulation-detection/blob/master/PoC%20-%20SVM/Casia/SVM%20-%20Casia%20-%20skalar%20first.ipynb)


## Założone stałe dotyczące zdjęcia

Zgodnie z Pana wytycznymi posłużyłem się bazą danych CASIA w wersji ITDE V2.0, o której więcej można przeczytać 

```
Dong, J., Wang, W., & Tan, T. (2013). CASIA Image Tampering Detection Evaluation Database. 2013 IEEE China Summit and International Conference on Signal and Information Processing.
```

W dużym skrócie baza zawiera ok. 7 500 zdjęć 'oryginalnych' i ok. 5 100 zdjęć 'przerobionych'. Przy czym zdjęcia te sa w różnej rozdzielczości od 320 x 240 do 800 x 600.

![alt](statystyki_zbioru_danych.png?raw=true)

W związku z faktem że w dalszych etapach będę ze zdjęć wyciągał informację, przyjąłem że zmienię ich rozmiar do 128 x 128

> Pytanie: Czy przyjęcie takiego rozmiaru ma sens? Czy zdjęcia nie są w tym wypadku za małe? Albo inaczej - czy dobór wielkości zdjęcia wynika z połączenia intuicji i doświadczenia, czy jest w tym wyborze coś jeszcze?

Trochę taka odpowiedź z przyszłości, przeliczyłem wersje dla wielkości obrazów 128 x 128 i 256 x 256. Wyniki były wręcz identyczne. Przy czym zadane pytanie nadal wydaje mi sie na miejscu - albo bardziej poszedłbym w stronę:

> Czy lepiej jest wyciągać bardziej szczegółowe informacje z mniejszych zdjęć, czy może lepiej wyciągać mniej informacji ale z większych zdjęć. Rozumiem że to zależy od samej funkcji informacji, bardziej mi chyba chodzi o to, w którym miejscu bardziej dokładne dostrojenie może przynieść większy zysk.

## Wyciąganie informacji ze zdjęcia

Do wyciągnięcia informacji ze zdjęcia posłużyłem się funkcjami `HuMoments()`, `calcHist()` z biblioteki `cv2`, funkcją `haralick()` z biblioteki `mahotas` i w jednym z przypadków - funkcją `hog` z biblioteki `skimage`.

Szczególnie zwracam uwagę na funkcję `hog`, która dość znacząco zwiększa wielkość ostateczną wektora, który opisuje konkretne zdjęcie(tworzę go poprzez zwykłego `hstack()` z cech zdjęcia).

Wektor zdjęcia z przeliczonym parametrem HOG(dla 128 x 128) posiada 856 cech, przy 532, gdy HOGa nie ma. 

> Pytanie: Przeglądałem internet w poszukiwaniu sposobów kategoryzacji zdjęć przy pomocy klasycznych algorytmów uczenia maszynowego. Rozumiem, że nie istnieje uniwersalna metoda wyciągania cech ze zdjęcia. Przyznam jednak, że skorzystałem z wymienionych powyżej funkcji dlatego że miały dla mnie wizualny sens. Czy powinienem użyć czegoś innego?

Podsumowując przeliczyłem dokładnie ten sam model dla 4 zestawów danych:

- wersja z HOGiem
- wersja bez
- wersja z HOGiem i PCA(do 532 elementów, o tym więcej poniżej)
- wersja bez HOGa i PCA(do 532 elementów)

```
Z HOG:
	 Wektor zdjęć: (12614, 3448)
	 Wektor kategorii słownych: (12614,)


BEZ HOG:
	 Wektor zdjęć: (12614, 532)
	 Wektor kategorii słownych: (12614,)

Z HOG + PCA:
	 Wektor zdjęć: (12614, 532)
	 Wektor kategorii słownych: (12614,)


BEZ HOG + PCA:
	 Wektor zdjęć: (12614, 532)
	 Wektor kategorii słownych: (12614,)
```

zdecydowałem się na 532 elementy by móc porównać dane bez HOGa i dane z HOGiem i PCA. Domyślam się że to dość arbitralna liczba ale na ten moment tylko na to wpadłem.

> Pytanie: Czy powinienem na tym etapie, gdzie uzyskane wyniki dla wszystkich zbiorów są wręcz takie same wybrać *zwycięzce* i dalej go rozwijać - dostrajać? 

## Skalowanie danych

Wszystkie dane są oczywiście skalowane do zakresu od 0 do 1. Zdecydowałem się na skorzystanie z `MinMax` skalara z racji że nie jestem pewien że wyciągane cechy mają rozkład normalny.

Mam jednak pewien dylemat. Początkowo posiłkując się różnymi artykułami naukowymi i poprzednią swoją pracą skalowałem dane w osobnym kroku w momencie w którym wszystkie dane miałem już gotowe. Może lepiej to wytłumaczę na przykładzie:

**ver A**

```py
def skaluj_dane(dane):
    ...

def wyciagnij_info_ze_zdjecia(zdjecie):
    ...

dane = []
for pic in zdjecia:
    dane.append(wyciagnij_info_ze_zdjecia(pic))

dane = skaluj_dane(dane)
```

Zadałem sobie jednak pytanie: dlaczego skalowanie danych jest osobnym krokiem i czemu skaluje ogół danych w stosunku do siebie, czy nie powinienem skalować osobno danych z każdego zdjęcia osobno?

**ver B**

```py
def skaluj_zdjecie(zdjecie):
    ...

def wyciagnij_info_ze_zdjecia(zdjecie):
    ...
    skaluj_zdjecie(zdjecie)


dane = []
for pic in zdjecia:
    dane.append(wyciagnij_info_ze_zdjecia(pic))
```

> Pytanie: Czy wersja *A* czy *B* jest lepsza?

## Podział danych

Zgodnie z materiałami które mi Pan polecił skorzystałem z `StratifiedKFold` jako że liczba obiektów w każdej z klas jest różna i ustaliłem `n_splits` na 5.

## Model

Jeśli chodzi o model wszystkie obliczenia wykonywałem na:

```py
svm = SVC(kernel='linear', probability=True, random_state=odp, verbose=True)
```

Mam jednak tutaj kilka pytań

> Czym jest *probability*?

Artykuły które czytałem dotyczące klasyfikacji zdjęć przy pomocy SVMa zawsze ustawiają parametr SVC, `probability` na `True`.

W dokumentacji `scikit-learn` piszę, że

```
Whether to enable probability estimates. This must be enabled prior to calling fit, will slow down that method as it internally uses 5-fold cross-validation, and predict_proba may be inconsistent with predict. Read more in the User Guide.
```

Z czego rozumiem że dla naszych potrzeb i dla naszych obliczeń(już wykonane 5v) jest nie potrzebne. W momencie w którym to piszę właśnie puściłem liczenie dla `probability` na `False`. Jak tylko uzyskam wynik - pojawi się to pewnie na githubie :octocat:.

> Rozumiem że powinienem policzyć wersje dla innych kerneli. Dotychczas w takich wypadkach zdawałem się na GridSearcha. Jednak inne przypadki liczyły się dużo krócej(do 10 min jeden). Czy w tym wypadku mogę po prostu na *ślepo* wybrać inny kernel, czy nie jest to do końca zgodne ze sztuką.

Plus może się jeszcze upewnię: W momencie w którym będę porównywał modele a nie poszczególne zbiory testowe, powinienem ocenić model na podstawie testów statystycznych(patrz: [link od strony kursu](https://metsi.github.io/2020/04/03/kod4.html)).

### Najlepszy wynik

**Wielkość zdjęcia**: 128 x 128
**MinMax jako osobny krok**: Tak
**Dane z Hogiem/bez** : Z Hogiem
**Dane po PCA**: Tak

```
Accuracy: 0.6627552449266904
Precision: 0.6560945187178729
Recall: 0.6627552449266904
F-score: 0.6550985415225934

[[5861 1630]
 [2624 2499]]
```

![alt](cm.png?raw=true)

W późniejszych wersjach do Accuracy, Precision, Recall, F-score dodałem też ich odchylenia standardowe.