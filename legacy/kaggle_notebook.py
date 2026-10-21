# Code cells of the 2023 Kaggle notebook "Audio Emotional Classification"
# (https://www.kaggle.com/code/baraamelhem/audio-emotional-classification), transcribed from the
# published version for reference. Inputs: uldisvalainis/audio-emotions and
# dmitrybabko/speech-emotion-recognition-en. Code is unmodified except that copy-pasted cells are
# abridged where marked "...", with their printed results kept as comments. Not part of the aum2
# package and not linted. See AUDIT.md for why its reported numbers are not valid.

# %% [1]
# %%capture
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %% [2]
import numpy as np
import pandas as pd

import random
import os
from tqdm import tqdm

import matplotlib.pylab as plt
import seaborn as sns

from glob import glob

import librosa
import librosa.display
import librosa.effects as le
import IPython.display as ipd

from tensorflow.image import resize

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.image import resize
from tensorflow.keras.models import load_model
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras import layers

from itertools import cycle

import joblib

sns.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

# %% [3]
# !pip install seaborn

# %% [4]
angry = glob(r'/kaggle/input/audio-emotions/Emotions/Angry/*.wav')
angry = sorted(angry)
Disgusted = glob(r'/kaggle/input/audio-emotions/Emotions/Disgusted/*.wav')
Disgusted = sorted(Disgusted)
Fearful = glob(r'/kaggle/input/audio-emotions/Emotions/Fearful/*.wav')
Fearful = sorted(Fearful)
Happy = glob(r'/kaggle/input/audio-emotions/Emotions/Happy/*.wav')
Happy = sorted(Happy)
Neutral = glob(r'/kaggle/input/audio-emotions/Emotions/Neutral/*.wav')
Neutral = sorted(Neutral)
Sad = glob(r'/kaggle/input/audio-emotions/Emotions/Sad/*.wav')
Sad = sorted(Sad)
Suprised = glob(r'/kaggle/input/audio-emotions/Emotions/Suprised/*.wav')
Suprised = sorted(Suprised)

# %% [5] Data preprocessing
##### angry testing
angry_train=[]
label_angry=[]
target_shape=(128,128)
with tqdm(total=len(angry)) as pbar:
    for i in range (len(angry)):
        y, sr =librosa.load(angry[i], sr=None)
        y_stretched = le.time_stretch(y, rate=1)
        Mel_spectrogram = librosa.feature.melspectrogram(y=y_stretched , sr=sr)
        Mel_spectrogram = resize(np.expand_dims(Mel_spectrogram,axis=-1),target_shape)
        angry_train.append(Mel_spectrogram)
        label_angry.append(0)
        pbar.update(1)
print("heeey")

##### Disgusted testing
Disgusted_train=[]
label_Disgusted=[]
target_shape=(128,128)
with tqdm(total=len(Disgusted)) as pbar:
    for i in range (len(Disgusted)):
        y, sr =librosa.load(Disgusted[i], sr=None)
        y_stretched = le.time_stretch(y, rate=1)
        Mel_spectrogram = librosa.feature.melspectrogram(y=y_stretched , sr=sr)
        Mel_spectrogram = resize(np.expand_dims(Mel_spectrogram,axis=-1),target_shape)
        Disgusted_train.append(Mel_spectrogram)
        label_Disgusted.append(1)
        pbar.update(1)
print("heeeey2")

##### Fearful testing
Fearful_train=[]
label_Fearful=[]
target_shape=(128,128)
with tqdm(total=len(Fearful)) as pbar:
    for i in range (len(Fearful)):
        y, sr =librosa.load(Fearful[i], sr=None)
        y_stretched = le.time_stretch(y, rate=1)
        Mel_spectrogram = librosa.feature.melspectrogram(y=y_stretched , sr=sr)
        Mel_spectrogram = resize(np.expand_dims(Mel_spectrogram,axis=-1),target_shape)
        Fearful_train.append(Mel_spectrogram)
        label_Fearful.append(2)

        pbar.update(1)
print("heeeey3")

##### happy testing
Happy_train=[]
label_happy=[]
target_shape=(128,128)
with tqdm(total=len(Happy)) as pbar:
    for i in range (len(Happy)):
        y, sr =librosa.load(Happy[i], sr=None)
        y_stretched = le.time_stretch(y, rate=1)
        Mel_spectrogram = librosa.feature.melspectrogram(y=y_stretched , sr=sr)
        Mel_spectrogram = resize(np.expand_dims(Mel_spectrogram,axis=-1),target_shape)
        Happy_train.append(Mel_spectrogram)
        label_happy.append(3)
        y_stretched = le.time_stretch(y, rate=1.4)
        Mel_spectrogram = librosa.feature.melspectrogram(y=y_stretched , sr=sr)
        Mel_spectrogram = resize(np.expand_dims(Mel_spectrogram,axis=-1),target_shape)
        Happy_train.append(Mel_spectrogram)
        label_happy.append(3)
        pbar.update(1)
print("heeeey4")

##### Neutral testing
Neutral_train=[]
label_Neutral=[]
target_shape=(128,128)
with tqdm(total=len(Neutral)) as pbar:
    for i in range (len(Neutral)):
        y, sr =librosa.load(Neutral[i], sr=None)
        y_stretched = le.time_stretch(y, rate=1)
        Mel_spectrogram = librosa.feature.melspectrogram(y=y_stretched , sr=sr)
        Mel_spectrogram = resize(np.expand_dims(Mel_spectrogram,axis=-1),target_shape)
        Neutral_train.append(Mel_spectrogram)
        label_Neutral.append(4)
        y_stretched = le.time_stretch(y, rate=1)
        Mel_spectrogram = librosa.feature.melspectrogram(y=y_stretched , sr=sr)
        Mel_spectrogram = resize(np.expand_dims(Mel_spectrogram,axis=-1),target_shape)
        Neutral_train.append(Mel_spectrogram)
        label_Neutral.append(4)
        pbar.update(1)
print("heeeey5")

##### Sad testing
Sad_train=[]
label_Sad=[]
target_shape=(128,128)
with tqdm(total=len(Sad)) as pbar:
    for i in range (len(Sad)):
        y, sr =librosa.load(Sad[i], sr=None)
        y_stretched = le.time_stretch(y, rate=1)
        Mel_spectrogram = librosa.feature.melspectrogram(y=y_stretched , sr=sr)
        Mel_spectrogram = resize(np.expand_dims(Mel_spectrogram,axis=-1),target_shape)
        Sad_train.append(Mel_spectrogram)
        label_Sad.append(5)
        pbar.update(1)
print("heeeey6")

# %% [6]
crema=glob("/kaggle/input/speech-emotion-recognition-en/Crema/*.wav")

# %% [7]
#SAD - sadness;
#ANG - angry;
#DIS - disgust;
#FEA - fear;
#HAP - happy;
#NEU - neutral.

target_emotion1 = "ANG"
target_emotion2 = "SAD"
target_emotion3 = "DIS"
target_emotion4 = "FEA"
target_emotion5 = "HAP"
target_emotion6 = "NEU"

i=0
with tqdm(total=len(crema)) as pbar:
    for filename in crema:
        if target_emotion1 in filename:
            y, sr =librosa.load(crema[i], sr=None)
            y_stretched = le.time_stretch(y, rate=1)
            Mel_spectrogram = librosa.feature.melspectrogram(y=y_stretched , sr=sr)
            Mel_spectrogram = resize(np.expand_dims(Mel_spectrogram,axis=-1),target_shape)
            angry_train.append(Mel_spectrogram)
            label_angry.append(0)

        elif target_emotion2 in filename:
            y, sr =librosa.load(crema[i], sr=None)
            y_stretched = le.time_stretch(y, rate=1)
            Mel_spectrogram = librosa.feature.melspectrogram(y=y_stretched , sr=sr)
            Mel_spectrogram = resize(np.expand_dims(Mel_spectrogram,axis=-1),target_shape)
            Sad_train.append(Mel_spectrogram)
            label_Sad.append(5)


        elif target_emotion3 in filename:
            y, sr =librosa.load(crema[i], sr=None)
            y_stretched = le.time_stretch(y, rate=1)
            Mel_spectrogram = librosa.feature.melspectrogram(y=y_stretched , sr=sr)
            Mel_spectrogram = resize(np.expand_dims(Mel_spectrogram,axis=-1),target_shape)
            Disgusted_train.append(Mel_spectrogram)
            label_Disgusted.append(1)

        elif target_emotion4 in filename:
            y, sr =librosa.load(crema[i], sr=None)
            y_stretched = le.time_stretch(y, rate=1)
            Mel_spectrogram = librosa.feature.melspectrogram(y=y_stretched , sr=sr)
            Mel_spectrogram = resize(np.expand_dims(Mel_spectrogram,axis=-1),target_shape)
            Fearful_train.append(Mel_spectrogram)
            label_Fearful.append(2)

        elif target_emotion5 in filename:
            y, sr =librosa.load(crema[i], sr=None)
            y_stretched = le.time_stretch(y, rate=1)
            Mel_spectrogram = librosa.feature.melspectrogram(y=y_stretched , sr=sr)
            Mel_spectrogram = resize(np.expand_dims(Mel_spectrogram,axis=-1),target_shape)
            Happy_train.append(Mel_spectrogram)
            label_happy.append(3)
            y_stretched = le.time_stretch(y, rate=1.3)
            Mel_spectrogram = librosa.feature.melspectrogram(y=y_stretched , sr=sr)
            Mel_spectrogram = resize(np.expand_dims(Mel_spectrogram,axis=-1),target_shape)
            Happy_train.append(Mel_spectrogram)
            label_happy.append(3)

        elif target_emotion6 in filename:
            y, sr =librosa.load(crema[i], sr=None)
            y_stretched = le.time_stretch(y, rate=1)
            Mel_spectrogram = librosa.feature.melspectrogram(y=y_stretched , sr=sr)
            Mel_spectrogram = resize(np.expand_dims(Mel_spectrogram,axis=-1),target_shape)
            Neutral_train.append(Mel_spectrogram)
            label_Neutral.append(4)
            y_stretched = le.time_stretch(y, rate=1.3)
            Mel_spectrogram = librosa.feature.melspectrogram(y=y_stretched , sr=sr)
            Mel_spectrogram = resize(np.expand_dims(Mel_spectrogram,axis=-1),target_shape)
            Neutral_train.append(Mel_spectrogram)
            label_Neutral.append(4)



        pbar.update(1)

        i=i+1

print("done")

# %% [8]
savee=glob("/kaggle/input/speech-emotion-recognition-en/Savee/*.wav")

# %% [9]
#'a' = 'anger'
#'d' = 'disgust'
#'f' = 'fear'
#'h' = 'happiness'
#'n' = 'neutral'
#'sa' = 'sadness'

target_emotion1 = "a"
target_emotion2 = "sa"
target_emotion3 = "d"
target_emotion4 = "f"
target_emotion5 = "h"
target_emotion6 = "n"

# (Identical body to cell 7 with `savee` in place of `crema`; because `filename` is the full
#  /kaggle/... path, `"a" in filename` is always true and every clip takes the first branch.)
i=0
with tqdm(total=len(savee)) as pbar:
    for filename in savee:
        if target_emotion1 in filename:
            y, sr =librosa.load(savee[i], sr=None)
            y_stretched = le.time_stretch(y, rate=1)
            Mel_spectrogram = librosa.feature.melspectrogram(y=y_stretched , sr=sr)
            Mel_spectrogram = resize(np.expand_dims(Mel_spectrogram,axis=-1),target_shape)
            angry_train.append(Mel_spectrogram)
            label_angry.append(0)
        # ... remaining elif branches as in cell 7 (never reached) ...
        pbar.update(1)
        i=i+1

print("done!")

# %% [10] Model 1 (negative vs positive)
exp_train=Happy_train+angry_train+Disgusted_train+Fearful_train+Neutral_train+Sad_train
exp_label=label_happy+label_angry+label_Disgusted+label_Fearful+label_Neutral+label_Sad

for i in range(len(exp_label)):
    if exp_label[i]==1 or exp_label[i]==2 or exp_label[i]==5 :
        exp_label[i]=0

for i in range(len(exp_label)):
    if exp_label[i]==4 or exp_label[i]==3:
        exp_label[i]=1

np.array(exp_label)
np.array(exp_train)

combined = list(zip(exp_train,exp_label))
random.shuffle(combined)
exp_train, exp_label = zip(*combined)

print("done!")

# %% [11]
labels = to_categorical(exp_label, num_classes=2)

X_train, X_test, y_train, y_test = train_test_split(exp_train, labels, test_size=0.25, random_state=42)

# Neural network model
input_shape = X_train[0].shape
input_layer = Input(shape=input_shape)
x = Conv2D(16, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(32, activation='relu')(x)
output_layer = Dense(2, activation='softmax')(x)
model = keras.Model(inputs=input_layer, outputs=output_layer)

# %% [12]
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
X_train = tf.stack(X_train)
y_train = tf.stack(y_train)
X_test = tf.stack(X_test)
y_test = tf.stack(y_test)

# %% [13]
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
test_accuracy=model.evaluate(X_test,y_test,verbose=0)

print("\n Accuracy= ",test_accuracy[1])   # printed: Accuracy=  0.87507563829422

# %% [14]
joblib.dump(model, 'Emotion_Audio_Model.pkl')

# %% [15-30] Models 2-5
# Same pattern as cells 10-14 on subsets:
#   Model 2: angry/disgust/fear/sad (4 classes), epochs=7,  batch_size=35  -> printed 0.6098
#   Model 3: happy/neutral,                      epochs=10, batch_size=25  -> printed 0.9000
#   Model 4: sad/neutral,                        epochs=10, batch_size=25  -> printed 0.8314
#   Model 5: angry/happy,                        epochs=10, batch_size=25  -> printed 0.8725

# %% [31] Prediction
def predict_model(audio1):
    target_shape=(128,128)
    y, sr =librosa.load(audio1, sr=None)
    y_stretched = le.time_stretch(y, rate=1)
    Mel_spectrogram = librosa.feature.melspectrogram(y=y_stretched , sr=sr)
    Mel_spectrogram = resize(np.expand_dims(Mel_spectrogram,axis=-1),target_shape)
    Mel_spectrogram = tf.reshape(Mel_spectrogram, (1,) + target_shape + (1,))

    return Mel_spectrogram

# %% [32]
# models_system(MSG): the five-model cascade, identical to `models_system` in legacy/AUM2.py
# (without the confidence string).

# %% [33]
test_angry=glob("/kaggle/input/speech-emotion-recognition-en/Tess/YAF_angry/*.wav")
test_nutral=glob("/kaggle/input/speech-emotion-recognition-en/Tess/OAF_neutral/*.wav")
test_fear=glob("/kaggle/input/speech-emotion-recognition-en/Tess/OAF_Fear/*.wav")
test_sad=glob("/kaggle/input/speech-emotion-recognition-en/Tess/OAF_Sad/*.wav")
test_disgust=glob("/kaggle/input/speech-emotion-recognition-en/Tess/YAF_disgust/*.wav")
test_happy=glob("/kaggle/input/speech-emotion-recognition-en/Tess/YAF_happy/*.wav")

# %% [34-36] "Testing the model" on the TESS clips above (all of which are in audio-emotions,
# i.e. in the training data). For each emotion: counter / 200. Printed table:
#   Happy 0.985 | Angry 1.000 | Neutral 1.000 | Sad 0.975 | Fearful 0.970 | Disgusted 0.955

# %% [37-40] Visualisation of one waveform, its STFT and its mel spectrogram.
