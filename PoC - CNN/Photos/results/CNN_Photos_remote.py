#!/usr/bin/env python
# coding: utf-8

# # Sprawdzanie środowiska

# In[ ]:


from platform import python_version

print(python_version())


# # Przygotowanie odpowiednich danych

# In[ ]:


'''
Dla PoC wykonuje obliczenia dla:
 * '../../data/DogsCats'
Folder docelowy:
 * '../../data/Photos'
Folder Casia:
 * '../../data/Casia'
'''

dir_path = 'D:/Studia/Magisterka/Image-manipulation-detection/data/Photos'
A_folder = 'originals'
B_folder = 'photoshops'


# # Załadowanie danych

# In[ ]:


import numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import math
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import random
import os

# fix random bo tak ( ͡° ͜ʖ ͡°)
odp = 42
numpy.random.seed(odp)


# # Stałe

# In[ ]:


# Wilkości odpowiednie dla ResNetu

IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3


# # Przygotowanie Danych

# In[ ]:


'''
Opis danych:
1 - klasa 1 -> Originals
0 - klasa 2 -> Photoshops
''' 

A_folder_list = os.listdir(dir_path + '/' + A_folder)
B_folder_list = os.listdir(dir_path + '/' + B_folder)

filenames = []
categories = []

for filename in A_folder_list:
    categories.append(1)
    filenames.append(dir_path + '/' + A_folder + '/' + filename)

for filename in B_folder_list:
    categories.append(0)
    filenames.append(dir_path + '/' + B_folder + '/' + filename)


df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})


# In[ ]:


# Mieszamy!
df = df.sample(frac=1).reset_index(drop=True)


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df['category'].value_counts().plot.bar()


# In[ ]:


sample = random.choice(df['filename'])
image = load_img(sample)


# # ELA

# In[ ]:


import cv2
from PIL import Image, ImageChops, ImageEnhance

def ft_ela(image_path):
    im = Image.open(image_path).convert('RGB')
    im.save('tmp.jpg', 'JPEG', quality=95)
    resaved_im = Image.open('tmp.jpg')

    ela_im = ImageChops.difference(im, resaved_im)

    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
    ela_im = ela_im.resize(IMAGE_SIZE)
    ret = numpy.array(ela_im).flatten() / 255

    return ret, ela_im


# In[ ]:


ela = ft_ela(sample)

print(f'Kształt: {ela[0].shape}')
print(f'Max: {numpy.amax(ela[0])}')
print(f'Min: {numpy.amin(ela[0])}')


# # Przeliczenie Cech Zdjęć + Kategorii

# In[ ]:


'''
Podział danych z całego df na X i y:

X - wszystko oprócz category
y - category
'''

X = []
Y = []

for index, row in df.iterrows():
   X.append(numpy.array(ft_ela(row[0])[0]))
   Y.append(row[1])


# In[ ]:


df.head()


# In[ ]:


X = numpy.array(X)
Y = numpy.array(Y)


# In[ ]:


X.shape


# In[ ]:


Y.shape


# # Zapis/Odczyt

# In[ ]:


import h5py

def save(features, labels, dataframe, name):
    h5f_data = h5py.File('data_' + str(name) + '.h5', 'w')
    h5f_data.create_dataset('dataset', data=numpy.array(features))

    h5f_label = h5py.File('labels_' + str(name) + '.h5', 'w')
    h5f_label.create_dataset('dataset', data=numpy.array(labels))

    h5f_data.close()
    h5f_label.close()

    dataframe.to_csv('dataframe_' + str(name) + '.csv')
    
def load(features, labels, dataframe):
    h5f_data  = h5py.File(features, 'r')
    h5f_label = h5py.File(labels, 'r')

    global_features_string = h5f_data['dataset']
    global_labels_string   = h5f_label['dataset']

    global_features = numpy.array(global_features_string)
    global_labels   = numpy.array(global_labels_string)

    h5f_data.close()
    h5f_label.close()
    
    df = pd.read_csv(dataframe, index_col = 0)  
    
    return global_features, global_labels, df
    
save(X, Y, df, name='Casia_CNN')

X, Y, df = load('data_Casia_CNN.h5', 'labels_Casia_CNN.h5', 'dataframe_Casia_CNN.csv')


# In[ ]:


df.head()


# In[ ]:


print('Kształt po wczytaniu:')
print(f'\t X: {X.shape}')
print(f'\t Y: {Y.shape}')


# # Funkcję liczące statystyki

# In[ ]:


from IPython.display import Markdown, display

def printmd(string):
    display(Markdown(string))


# In[ ]:


from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import load_model, clone_model
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import keras.backend as K
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, roc_auc_score
from keras.optimizers import RMSprop


# In[ ]:


def countStats(_y_true, _y_pred):
    accuracy = accuracy_score(_y_true, _y_pred, normalize=True)
    precision = precision_score(_y_true, _y_pred, average='binary')
    recall = recall_score(_y_true, _y_pred, average='binary')
    fscore = f1_score(_y_true, _y_pred, average='binary')
    
    return accuracy, precision, recall, fscore


# In[ ]:


'''
Źrodło:
https://medium.com/@aakashgoel12/how-to-add-user-defined-function-get-f1-score-in-keras-metrics-3013f979ce0d
'''

def get_f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    
    return f1_val


# In[ ]:


from mlxtend.plotting import plot_confusion_matrix

def plot_cm(cm, classes):
    fig = plot_confusion_matrix(conf_mat=cm,
                          colorbar=True, 
                          show_absolute=True,
                          show_normed=True,
                          class_names=classes,
                          figsize=(10, 8))
    
    return fig


# In[ ]:


'''
cb_early_stopper - skończenie uczenia kiedy val_loss nie będzie się poprawiać przez 10 epok
cb_checkpointer - zapis modelu do pliku 'best.h5' modeli o najlepszym(najmniejszym) val_loss
cb_learning_rate_reduction - zmniejszenie LR jeśli val_loss nie będzie się poprawiać przez 5 epok
'''

EARLY_STOP_PATIENCE = 10
LEARNING_RATE_PATIENCE = 5

cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = EARLY_STOP_PATIENCE, verbose=0)
cb_checkpointer = ModelCheckpoint(filepath = 'best.h5', monitor = 'val_loss', save_best_only = True, verbose=0)
cb_learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=LEARNING_RATE_PATIENCE, verbose=0)


# # Dobór parametrów

# In[ ]:


# To Do

epochs = 50
batch_size = 100
activation = 'relu'
loss_type = 'binary_crossentropy'
optimizer = RMSprop(lr=1e-4)
dropout = 0.25


# # Model

# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5, 5), activation=activation, 
                 input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
print("Input: ", model.input_shape)
print("Output: ", model.output_shape)
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation=activation))
print("Input: ", model.input_shape)
print("Output: ", model.output_shape)
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(filters=128, kernel_size=(3, 3), activation=activation))
print("Input: ", model.input_shape)
print("Output: ", model.output_shape)
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation=activation))
print("Input: ", model.input_shape)
print("Output: ", model.output_shape)
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(dropout))
model.add(Dense(512, activation=activation))
model.add(Dense(1, activation='sigmoid'))


# In[ ]:


model.summary()


# In[ ]:


model.compile(loss=loss_type, optimizer=optimizer, metrics=['accuracy', get_f1])


# In[ ]:


model.save("base_model.h5")


# # Fit

# In[ ]:


from tabulate import tabulate

features = X
labels = Y
name = 'CNN'


tcm_list = []
tAccuracy_list = []
tPrecision_list = []
tRecall_list = []
tFScore_list = []

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=odp)
    
scores_a = numpy.zeros((1, 5))
scores_p = numpy.zeros((1, 5))
scores_r = numpy.zeros((1, 5))
scores_f = numpy.zeros((1, 5))

for fold_id, (train_index, test_index) in enumerate(kf.split(features, labels)):
    X_train = features[train_index].reshape(-1, 128, 128, 3)
    Y_train = labels[train_index]
    
    X_test = features[test_index].reshape(-1, 128, 128, 3)
    Y_test = labels[test_index]
    
    EARLY_STOP_PATIENCE = 10
    LEARNING_RATE_PATIENCE = 5

    cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = EARLY_STOP_PATIENCE, verbose=0)
    cb_checkpointer = ModelCheckpoint(filepath = 'best.h5', monitor = 'val_loss', save_best_only = True, verbose=0)
    cb_learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=LEARNING_RATE_PATIENCE, verbose=0)
    
    clf = load_model('base_model.h5', custom_objects={'get_f1': get_f1})
    history = clf.fit(
        x = X_train,
        y = Y_train,
        batch_size = batch_size,
        epochs = epochs,
        validation_data = (X_test, Y_test),
        callbacks = [cb_checkpointer, cb_early_stopper, cb_learning_rate_reduction],
        verbose = 0
    )
    
    clf.load_weights('best.h5')
    y_pred = clf.predict(X_test)
    y_pred = y_pred.reshape(-1)
    y_pred = numpy.where(y_pred > 0.5, 1, 0)

    accuracy, precision, recall, fscore = countStats(Y_test, y_pred)
    cm = confusion_matrix(Y_test, y_pred)
    scores_a[0, fold_id] = accuracy
    scores_p[0, fold_id] = precision
    scores_r[0, fold_id] = recall
    scores_f[0, fold_id] = fscore

    tAccuracy_list.append(accuracy)
    tPrecision_list.append(precision)
    tRecall_list.append(recall)
    tFScore_list.append(fscore)
    tcm_list.append(cm)
    
    print(f'[{fold_id + 1} done!]', end='')

printmd(f'# {name}:')
print(f'\n\nKształt danych:')
print(f'\t X_train: {X_train.shape}')
print(f'\t X_test: {X_test.shape}')
print(f'\t y_train: {Y_train.shape}')
print(f'\t y_test: {Y_test.shape}')
    
accuracy_m = numpy.mean(tAccuracy_list)
precision_m = numpy.mean(tPrecision_list)
recall_m = numpy.mean(tRecall_list)
fscore_m = numpy.mean(tFScore_list)
        
accuracy_std = numpy.std(tAccuracy_list)
precision_std = numpy.std(tPrecision_list)
recall_std = numpy.std(tRecall_list)
fscore_std = numpy.std(tFScore_list)
    
cm = sum(tcm_list)

results = [['CNN', 
            f'{accuracy_m:.3f} ({accuracy_std:.2f})',
            f'{precision_m:.3f} ({precision_std:.2f})',
            f'{recall_m:.3f} ({recall_std:.2f})',
            f'{fscore_m:.3f} ({fscore_std:.2f})',
            f'{cm}']]
                            
printmd(f'### Rezultaty:')
headers = ["Kernel", "Accuracy", "Precision", "Recall", "Fscore", "CM"]
print('\n')
print(tabulate(results, headers=headers))
                            
with open(f'{name}_result.txt', 'w') as f:
    print(tabulate(results, headers=headers), file=f)
                            
numpy.save(f'{name}_results_a', scores_a)
numpy.save(f'{name}_results_p', scores_p)
numpy.save(f'{name}_results_r', scores_r)
numpy.save(f'{name}_results_f', scores_f)


# In[ ]:


# Wczytanie najlepszego
model.load_weights('best.h5')

# Zapis
model.save('the_best.h5')

# Zapis History
numpy.save('history.npy', history.history)
# history = numpy.load('history.npy', allow_pickle='TRUE').item()


# # Statystyki

# In[ ]:


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))

# Wykres loss
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="Validation loss")
ax1.set_ylim([-0.1, 1.1])
ax1.legend(loc='best', shadow=True)
ax1.set_ylabel('loss')
ax1.set_xlabel('epoch')

# Wykres accuracy
ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
ax2.set_ylim([-0.1, 1.1])
ax2.legend(loc='best', shadow=True)
ax2.set_ylabel('accuracy')
ax2.set_xlabel('epoch')

# Wykres F1
ax3.plot(history.history['get_f1'], color='b', label="Training f1")
ax3.plot(history.history['val_get_f1'], color='r',label="Validation f1")
ax3.set_ylim([-0.1, 1.1])
ax3.legend(loc='best', shadow=True)
ax3.set_ylabel('f1')
ax3.set_xlabel('epoch')


# plt.tight_layout()
fig.savefig('history.png')


# In[ ]:


fig = plot_cm(cm, ['Originals', 'Photoshops'])
fig[0].savefig('cm.png')

