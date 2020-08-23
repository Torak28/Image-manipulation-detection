# Wyniki

Najnowsze wyniki

## SVM Casia

[link_do_gita](https://github.com/Torak28/Image-manipulation-detection/blob/master/PoC%20-%20SVM/Casia/SVM%20-%20Casia%20Final%20v2.ipynb)

```
Kernel       Accuracy      Precision     Recall        Fscore        CM
-----------  ------------  ------------  ------------  ------------  -------------
SVM linear   0.710 (0.01)  0.664 (0.01)  0.581 (0.01)  0.619 (0.01)  [[5983 1508]
                                                                      [2148 2975]]
SVM poly     0.647 (0.01)  0.618 (0.02)  0.343 (0.01)  0.441 (0.01)  [[6405 1086]
                                                                      [3368 1755]]
SVM rbf      0.725 (0.00)  0.669 (0.00)  0.638 (0.01)  0.653 (0.00)  [[5873 1618]
                                                                      [1857 3266]]
SVM sigmoid  0.662 (0.01)  0.593 (0.01)  0.538 (0.01)  0.564 (0.01)  [[5601 1890]
                                                                      [2369 2754]]
```

```
SVM rbf:

    Accuracy: 0.725 (0.00)  
    Precision: 0.669 (0.00)  
    Recall: 0.638 (0.01)  
    Fscore: 0.653 (0.00)

CM:

    [[5983 1508]
    [2148 2975]]
```

## SVM Photos

[link_do_gita](https://github.com/Torak28/Image-manipulation-detection/blob/master/PoC%20-%20SVM/Photos/SVM%20-%20Photos%20Final%20v2.ipynb)

```
Kernel       Accuracy      Precision     Recall        Fscore        CM
-----------  ------------  ------------  ------------  ------------  -------------
SVM linear   0.580 (0.01)  0.590 (0.01)  0.527 (0.01)  0.557 (0.01)  [[6890 3989]
                                                                      [5146 5733]]
SVM poly     0.482 (0.01)  0.468 (0.02)  0.264 (0.01)  0.337 (0.01)  [[7622 3257]
                                                                      [8011 2868]]
SVM rbf      0.529 (0.01)  0.530 (0.01)  0.504 (0.01)  0.517 (0.01)  [[6024 4855]
                                                                      [5398 5481]]
SVM sigmoid  0.541 (0.00)  0.543 (0.00)  0.527 (0.01)  0.535 (0.01)  [[6045 4834]
                                                                      [5144 5735]]
```

```
SVM rbf:

    Accuracy: 0.580 (0.01)
    Precision: 0.590 (0.01)  
    Recall: 0.527 (0.01)  
    Fscore: 0.557 (0.01)

CM:

    [[6890 3989]
    [5146 5733]]
```

## VGG CASIA

[link_do_gita](https://github.com/Torak28/Image-manipulation-detection/blob/master/PoC%20-%20VGG/Casia/VGG%20-%20version%202.ipynb)

> przeliczone zgodnie z plikem [link_do_gita](https://github.com/Torak28/Image-manipulation-detection/blob/master/PoC%20-%20VGG/Casia/results/VGG_Casia_remote.py)

```
Kernel    Accuracy      Precision     Recall        Fscore        CM
--------  ------------  ------------  ------------  ------------  -------------
VGG       0.891 (0.01)  0.945 (0.01)  0.867 (0.01)  0.904 (0.01)  [[4742  381]
                                                                   [ 999 6492]]
```

![Imgur](https://imgur.com/AV5PQcn.png)

![Imgur](https://imgur.com/tlLl6pj.png)

## VGG Photos

[link_do_gita](https://github.com/Torak28/Image-manipulation-detection/blob/master/PoC%20-%20VGG/Photos/VGG%20-%20version%202.ipynb)

> przeliczone zgodnie z plikem [link_do_gita](https://github.com/Torak28/Image-manipulation-detection/blob/master/PoC%20-%20VGG/Photos/results/VGG_Photos_remote.py)

```
Kernel    Accuracy      Precision     Recall        Fscore        CM
--------  ------------  ------------  ------------  ------------  -------------
VGG       0.689 (0.01)  0.696 (0.02)  0.674 (0.02)  0.684 (0.00)  [[7648 3231]
                                                                   [3543 7336]]
```

![Imgur](https://imgur.com/jeSKYse.png)

![Imgur](https://imgur.com/wIL4g7m.png)

## CNN Casia

[link_do_gita](https://github.com/Torak28/Image-manipulation-detection/blob/master/PoC%20-%20CNN/Casia/CNN%20-%20version%202.ipynb)

> przeliczone zgodnie z plikem [link_do_gita](https://github.com/Torak28/Image-manipulation-detection/blob/master/PoC%20-%20CNN/Casia/results/CNN_Casia_remote.py)

```
Kernel    Accuracy      Precision     Recall        Fscore        CM
--------  ------------  ------------  ------------  ------------  -------------
CNN       0.909 (0.01)  0.962 (0.01)  0.882 (0.01)  0.920 (0.01)  [[4865  258]
                                                                   [ 886 6605]]
```

![Imgur](https://imgur.com/pz7rJ9n.png)

![Imgur](https://imgur.com/0CJwBdy.png)

## CNN Photos

[link_do_gita](https://github.com/Torak28/Image-manipulation-detection/blob/master/PoC%20-%20CNN/Photos/CNN%20-%20version%202.ipynb)

> przeliczone zgodnie z plikem [link_do_gita](https://github.com/Torak28/Image-manipulation-detection/blob/master/PoC%20-%20CNN/Photos/results/CNN_Photos_remote.py)

```
Kernel    Accuracy      Precision     Recall        Fscore        CM
--------  ------------  ------------  ------------  ------------  -------------
CNN       0.706 (0.01)  0.711 (0.01)  0.697 (0.01)  0.704 (0.01)  [[7789 3090]
                                                                   [3297 7582]]
```

![Imgur](https://imgur.com/NM1upbr.png)

![Imgur](https://imgur.com/zAJ80W2.png)
