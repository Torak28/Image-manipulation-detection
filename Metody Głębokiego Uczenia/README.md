# Opracowanie wyników

Ocena: 4.0 :fire:

![gif](https://i.giphy.com/media/Ub8XEam5vXbMY/giphy.webp)

## Zadanie

W ramach trwania projektu chciałem stworzyć model który byłby w stanie sklasyfikować czy zawartość danego zdjęcia nie została w żaden sposób zmodyfikowana. Projekt realizowałem jako pojedyncza osoba.

## Baza zdjęć(dataset)

Pierwszym punktem przy realizacji tego projektu było znalezienie odpowiedniego zbioru danych. Uznałem, że najlepszy będzie zbiór posiadający najwięcej zdjęć oraz zapewniający ich różnorodność. W pracach naukowych dotykających tej tematyki można często sie spotkać z danymi skupionymi wokół, np. ludzkiej twarzy, czy skupionych na np. wykrywaniu tylko efektu duplikacji części zdjęcia w ramach jego samego.

Zdecydowałem na zbiór PS-Battles Dataset([źródło](https://github.com/dbisUnibas/ps-battles)). Zbiór ten został przygotowany na potrzeby konkursu o tematyce mojej pracy i zawiera sumarycznie 102 028 zdjęć. Niestety, czego nie zauważyłem na tym etapie, a raczej, z czego nie zdałem sobie sprawy, użyty zbiór danych jest znacząco niezbalansowany. Na każde zdjęcie naturalne przypada ok. 7,9 podróbek. Tym samym zdjęcia oryginalne to tylko 10% całego datasetu.

## Początkowy wybór modelu

Moim pierwszym pomysłem i wyborem był ResNet. Sposób w jaki został przedstawiony na wykładzie oraz fakt że zminimalizowano w nim problem zanikającego gradientu poprzez użycie *skip connections* sprawiał że wydawał mi sie dobrym kandydatem.

Jednak pierwsze próby w których, po prostu użyłem nauczonego na imagenetcie ResNetu50, były bardzo słabo obiecujące.

![https://imgur.com/j9FErAx.png](https://imgur.com/j9FErAx.png)

Wyniki oscylujące w okolicach 50-55% celności sprawiały że równie dobrze można by klasyfikować zdjęcia przez rzut monetą, albo przez funkcję `rand()`.

## Dostrajanie hiperparametrów

Uznałem jednak, że moje wcześniejsze niepowodzenia spowodowane były użyciem niedostrojonego narzędzia. Stworzyłem więc eksperyment, celem którego było wygenerować i ocenić szereg modeli na podstawie różnych wartości hiperparametrów.

I tak sprawdzałem:

 * różne metody poolingu, czyli czy lepszy jest *average_pooling* czy *max_pooling*,
 * który optimizer sprawuje się najlepiej(*sgd*, *adam*, *adadelta*),
 * która funkcja straty daje najwięcej(*mse*, *categorical_crossentropy*),
 * oraz wpływ generowania nowych/zmieniania danych poprzez augmentacje(obrót i przybliżenie)

Uzyskane wyniki znacząco się od siebie różniły, co początkowo traktowałem jako dobrą kartę. Okazało się jednak -  po wygenerowaniu macierzy konfuzji, że mój klasyfikator po prostu etykietował wszystkie dane jako *fałszywki*, przez to że jest ich znacznie więcej w porównaniu do zdjęć normalnych. Oczywiście w tym momencie cały eksperyment stał się skażony i jego wyniki nie były już żadną przydatną informacją.

## Zbiór referencyjny

Uznałem więc że skorzystam z podobnego problemu, najlepiej już opisanego. Tak żeby wyznaczyć sobie na prostych danych pewne trendy i potem przenieść je na mój model. Skupiłem się na problemie klasyfikacji Kota i Psa, jako że podobnie jak w moim przypadku mamy do czynienia z dwoma klasami.

Niestety nawet dla takich łatwo klasyfikowanych danych różnice w hiperparametrach niebyły dość znaczące. Udało mi się tylko ustalić że *average_pooling* jest lekko lepszy od *max_poolingu*.

## Próba zmiany danych

Jednak poprzez użycie innych danych dla klasyfikacji(psy i koty) zauważyłem że lepsze ich wyniki mogą być spowodowane zbalansowanymi danymi, w secie uczącym, testującym i validacyjnym. Tym samym postanowiłem zbalansować dane u siebie poprzez undersampling liczniejszego zbioru.

Całość faktycznie poprawiła wyniki klasyfikacji.

![https://imgur.com/SJs4sIw.png](https://imgur.com/SJs4sIw.png)

Jednak nie było to poprawa znacząca. Aktualnie wyniki oscylowały w okolicach 55-60%

## Zmiana ResNetu

Na tym etapie też zauważyłem że biblioteka z której korzystałem - *keras*, posiada więcej niż jedną wersję ResNetu. Jest ResNet50 z którego do tej pory korzystałem, ale jest też ResNet101 i ResNet152 oraz cała rodzina ResNetV2.

Uznałem że może zmiana na któryś z nich polepszy moje wyniki.

```json
{
                     //loss               //accuracy
    'ResNet101_avg': [0.6394971116350687, 0.6305845552017146],
    'ResNet101_max': [1.3586419791387616, 0.6204592939194035],
    'ResNet152_avg': [0.6503496110003776, 0.6037578337547799],
    'ResNet152_max': [1.5897361636161804, 0.5652400892624526],
    'ResNet50_avg':  [0.6776268681616275, 0.5610647237468861],
    'ResNet50_max':  [1.9044997841430855, 0.5165970834644826]
}
```

Jak widać moja wcześniejsza obserwacja o przewadze poolingu typu *avg* nad *max*  jest tutaj ewidentnie widoczna. Udało mi się również uzyskać wynik trochę lepszy niz poprzednio.

![https://imgur.com/PxLoShK.png](https://imgur.com/PxLoShK.png)

Na tym też etapie postanowiłem że trochę pozmienia moją sięć neuronową którą przedstawiam po zakończeniu działania samego ResNetu. Powyższy wynik(który zdradzę żeby najlepszy) został uzyskany na takim wyglądzie sieci:

```
Layer (type)                 Output Shape              Param #   
=================================================================
resnet101 (Model)            (None, 2048)              42658176  
_________________________________________________________________
dense_91 (Dense)             (None, 512)               1049088   
_________________________________________________________________
dense_92 (Dense)             (None, 256)               131328    
_________________________________________________________________
dense_93 (Dense)             (None, 256)               65792     
_________________________________________________________________
dense_94 (Dense)             (None, 128)               32896     
_________________________________________________________________
dense_95 (Dense)             (None, 32)                4128      
_________________________________________________________________
dense_96 (Dense)             (None, 2)                 66        
=================================================================
Total params: 43,941,474
Trainable params: 1,283,298
Non-trainable params: 42,658,176
```

Uznałem więc że może kolejnym krokiem będzie użycie ResNetuV2. Zgodnie z jego pracą naukową, obiecywał lepsze wyniki, poprzez lekki update swojej struktury(poprzez dodanie np. batch normalization).

Niestety, wyniki dla ResNetuV2 się pogorszyły, nawet do momentu <50%

## Uczenie na 'czystym' modelu

W dalszej części pomyślałem że ciekawym pomysłem byłoby nauczenie ResNetu samemu, nie wykorzystując wag jakie posiada z Imagenetu. Nie spodziewałem się po tym zabiegu jakiejkolwiek poprawy - miałem za małą ilość zdjęć by takie nauczanie mogło by być efektywne. Co ciekawe wszystkie wyniki oscylowały w okolicach dokładnie 50%

```json
{
                            //loss               //accuracy
    'VanilaResNet101_avg': [0.8407009588651717, 0.5000000071083828],
    'VanilaResNet101_max': [1.3552867115274974, 0.5000000067972937],
    'VanilaResNet152_avg': [0.696174303026936,  0.4849686909988056],
    'VanilaResNet152_max': [0.8308858323010124, 0.5002087749509697],
    'VanilaResNet50_avg':  [8.05904774738005,   0.5000000065561998],
    'VanilaResNet50_max':  [8.059047742029321,  0.5000000062762198]
 }
```

## Stan wiedzy

W dalszej części projektu postanowiłem zorientować się jak wygląda naukowe podejście do mojego problemu. Już wcześniej, w trakcie pracy nad projektem szukałem i czytałem pracę naukowe w tej tematyce, ale tym razem chciałem spróbować zaimplementować jakieś rozwiązanie. 

Szczególnie zainteresowały mnie dwie pracy:
 * [ManTra-Net: Manipulation Tracing Network For Detection And Localization of Image Forgeries With Anomalous Features](https://zpascal.net/cvpr2019/Wu_ManTra-Net_Manipulation_Tracing_Network_for_Detection_and_Localization_of_Image_CVPR_2019_paper.pdf),
 * [Learning Rich Features for Image Manipulation Detection](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhou_Learning_Rich_Features_CVPR_2018_paper.pdf)

## ManTra-Net

Ideą tego rozwiązania jest zastosowanie m.in. Error level analysis, czyli nałożenia niejako na obraz ziarnistości/szumu który wynika z jego zawartości a następnie analizę tego szumu i odległości w jakiej oba interesujące obszary są od siebie odległe.

Pozwala to nie tylko wykryć 'fałszywkę' ale też zaznaczyć w ramach zdjęcia obszar który wydaje się nam podejrzany. 

![https://imgur.com/NyKCwoR.png](https://imgur.com/NyKCwoR.png)

Jak widać na przykładowym obrazku, wynik jest bardzo imponujący. Kiedy jednak zastosowałem nauczony model udostępniony przez Twórców do własnych zdjęć uzyskałem dużo gorszę wyniki.

![https://imgur.com/Rf4HFTv.png](https://imgur.com/Rf4HFTv.png)

Np. na powyższym obrazku, algorytm ewidentnie wykrywa sztuczny obiekt ale zaznacza tylko jego krawędzie, środek klasyfikując jako niepodejrzany.

![https://imgur.com/HmlQYUO.png](https://imgur.com/HmlQYUO.png)

W drugim przykładzie prawie całe moje zdjęcie zaznaczone jest jako fałszywe. Sami twórcy tej pracy adresują te problemy użytym datasetem.

Nie było w nim wielu jasnych słonecznych zdjęć z duż ilością światła, plus dodatkowo były to dataset skupiony wokół wklejania jednego zdjęcia w drugie.

## Learning Rich Features for IMD

Wydaję się bardziej ogólną pracą z bardzo ciekawa architekturą, inspirowaną Faster-RCNN

![https://imgur.com/9qPiuAL.png](https://imgur.com/9qPiuAL.png)

Górny flow danych, skupia się na przestrzeni barw RGB i jej zależnościach(przejściach, zmianach kontrastu), podczas gdy dolny operuje na zaszumionym zdjęciu. Dalej Region Proposal Network wybiera interesujące elementy z obu potoków i dalej są one zaznaczane jako pudełka i klasyfikowane. 

Niestety ale dla tego przykładu nie znalazłem czasu na implementacje, a sami twórcy takowej nie udostępniają nawet w postaci nauczonych modeli

## Wnioski

Ewidentnie projekt mnie przerósł. Z jednej strony jest to na pewno problem tego że to moja pierwsza styczność z sieciami głębokimi i brakowało mi często wiedzy i intuicji. Z innej, wybrany temat wydaje mi sie trochę zbyt wysoko wysoki ustawioną poprzeczką dla grupy jednoosobowej.