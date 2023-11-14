import numpy as np 
import pandas as pd 

import random
import os
from tqdm import tqdm
 
import sounddevice as sd
import soundfile as sf

from glob import glob

import librosa
import librosa.display
import librosa.effects as le

import tensorflow as tf
from tensorflow.image import resize
import pickle


import speech_recognition as sr
import time

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences 

# models
file_path = r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\Audio recognition\Model\audio models .h5\model1.h5"
model = load_model(file_path)

file_path = r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\Audio recognition\Model\audio models .h5\model2.h5"
model2 = load_model(file_path)
    
file_path = r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\Audio recognition\Model\audio models .h5\model3.h5"
model3 = load_model(file_path)  
    
file_path = r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\Audio recognition\Model\audio models .h5\model4.h5"
model4 = load_model(file_path)  

file_path = r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\Audio recognition\Model\audio models .h5\model5.h5"
model5 = load_model(file_path)
    
# recall the model
modelSent = load_model(r'C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\sentiment tweets\Model\MODEL.h5')


file_path = r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\sentiment tweets\Model/tokenizer1.pickle"
with open(file_path, 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)
    
#prerpcessing
def predict_model(audio1):       
    target_shape=(128,128)
    y, sr =librosa.load(audio1, sr=None)
    y_stretched = le.time_stretch(y, rate=1)
    Mel_spectrogram = librosa.feature.melspectrogram(y=y_stretched , sr=sr)
    Mel_spectrogram = resize(np.expand_dims(Mel_spectrogram,axis=-1),target_shape)
    Mel_spectrogram = tf.reshape(Mel_spectrogram, (1,) + target_shape + (1,))

    return Mel_spectrogram



#prediction
def models_system(MSG):
    
    predictions = model.predict(MSG)
    class_probabilities = predictions[0]
    predicted_class_index = np.argmax(class_probabilities)

    if predicted_class_index==0:#negative
        predictions = model2.predict(MSG)
        class_probabilities = predictions[0]
        predicted_class_index = np.argmax(class_probabilities)

        if predicted_class_index==0:#model5
            predictions = model5.predict(MSG)
            class_probabilities = predictions[0]
            predicted_class_index = np.argmax(class_probabilities)
            if predicted_class_index==0:
                Final_pred="Angry"
            elif predicted_class_index==1:
                Final_pred="Happy"   
            
        elif predicted_class_index==1:
            Final_pred="Disgusted"
        elif predicted_class_index==2:
            Final_pred="Fearful"        
        elif predicted_class_index==3:#model4
            predictions = model4.predict(MSG)
            class_probabilities = predictions[0]
            predicted_class_index = np.argmax(class_probabilities)
            if predicted_class_index==0:
                Final_pred="Sad"
            elif predicted_class_index==1:
                Final_pred="Neutral"

    elif predicted_class_index==1:#positive
        predictions = model3.predict(MSG)
        class_probabilities = predictions[0]
        predicted_class_index = np.argmax(class_probabilities)

        if predicted_class_index==0:#model5
            predictions = model5.predict(MSG)
            class_probabilities = predictions[0]
            predicted_class_index = np.argmax(class_probabilities)
            if predicted_class_index==0:
                Final_pred="Angry"
            elif predicted_class_index==1:
                Final_pred="Happy"     
            
            
        elif predicted_class_index==1:#model4
            predictions = model4.predict(MSG)
            class_probabilities = predictions[0]
            predicted_class_index = np.argmax(class_probabilities)
            if predicted_class_index==0:
                Final_pred="Sad"
            elif predicted_class_index==1:
                Final_pred="Neutral"
    largest_number = max(class_probabilities)
    FX=f"{Final_pred }{', with precentage of: '}{largest_number}"         
    return FX


def NLP_on_text(txt):
        
    
    
    import neattext as nt
    import nltk
    mytext=txt
    docx = nt.TextFrame(text=mytext)
    docx.text 
    docx.text=docx.normalize(level='deep')
    docx=docx.remove_emojis()
    docx=docx.fix_contractions()
    txt=docx
    
    
    
    nwtxt=[]
    
    import spacy
    
    nlp = spacy.load('en_core_web_sm')
    newcleantext = []
    
    
    doc1 = nlp(txt)
    postex = []
    
    for token in doc1:
        wordtext = token.text
        poswrd = spacy.explain(token.pos_)
    
        # Map POS to single characters
        if poswrd == "verb":
            poswrd = "v"
        elif poswrd == "noun":
            poswrd = "n"
        elif poswrd == "adjective":
            poswrd = "a"
        elif poswrd == "adverb":
            poswrd = "r"
        elif poswrd == "pronoun":
            poswrd = "n"
        elif poswrd == "determiner":
            poswrd = "dt"
        elif poswrd == "conjunction":
            poswrd = "cc"
        elif poswrd == "preposition":
            poswrd = "prep"
        elif poswrd == "interjection":
            poswrd = "intj"
        elif poswrd == "common noun":
            poswrd = "n"
        elif poswrd == "proper noun":
            poswrd = "n"
        elif poswrd == "mass noun":
            poswrd = "n"
        elif poswrd == "count noun":
            poswrd = "n"
        else:
            poswrd = "n"
    
        postex.append(f"({wordtext})({poswrd})")
    
    newcleantext.append(",".join(postex))
    
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet
    from tqdm import tqdm
    
    lemmatizer = WordNetLemmatizer()
    
    textsve = ""
    text_in_data = txt
    tokens = [pair.strip("()").split("),") for pair in text_in_data.split("),(")]
    for word_pos in tokens:
        if len(word_pos) == 1:
            word, pos = word_pos[0], 'n'  #  'n' as a default part of speech if there's only one value.
        else:
            word, pos = word_pos
                
        if pos == "dt" or pos == "cc" or pos == "prep" or pos == "intj":
            pos = wordnet.NOUN
        
        textsve = lemmatizer.lemmatize(word, pos=pos) + " " + textsve
    
    newcleantext = []
    newcleantext.append(textsve)
    
    
    
    
    
    ###
    
    text=newcleantext[0]
    text_without_parentheses = text.replace("(v", "").replace(")", "").replace("(r","").replace("(n","").replace("(a","").replace("(dt","").replace("(cc","").replace("(prep","").replace("(intj","")
    newcleantext=text_without_parentheses
            
            
    
    # text cleaning2 
    
    mytext=newcleantext
    docx = nt.TextFrame(text=mytext)
    docx.text 
    docx=docx.remove_stopwords()
    docx=docx.fix_contractions()
    newcleantext=docx
       

    # Reverse the order of words
    
    text=newcleantext
    words = text.split()
    reversed_text = ' '.join(words[::-1])
    newcleantext=reversed_text
    
    return newcleantext



r = sr.Recognizer()

recorded_audio_file_path = r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\Audio recognition\recorded audio\Record_to_predict.wav"
your_name=input("enter your name :")
sample_rate = 44100
duration = 20

print("\n program started \n")
print("Trigger word is :'AUM2 how are you' ")
while True:
    audio_data = sd.rec(int(sample_rate * 7), samplerate=sample_rate, channels=2)
    sd.wait()
    sf.write(recorded_audio_file_path , audio_data, sample_rate)
    audio_record=glob(recorded_audio_file_path) 
    with sr.AudioFile(audio_record[0]) as source:
        audio = r.listen(source)
    # Convert audio to text
    try:
        text = r.recognize_google(audio)
        #print("You:" + text+"\n")
        
        # Check if the trigger word is present
        if "home2 how are you" in text or "home 2 how are you" in text or "home to how are you" in text or "aom 2 how are you" in text or "awm 2 how are you" in text or "home two how are you" in text or "aom two how are you" in text or "awm two how are you" in text or "I'm too how are you" in text or "02 how are you" in text or "home too how are you" in text:
            print("AUM2: Hi",your_name,"How do you feel today? ..")
            # Record audio from the microphone
            audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=2)
            sd.wait()
            sf.write(recorded_audio_file_path , audio_data, sample_rate)
            audio_record=glob(recorded_audio_file_path) 
            
            with sr.AudioFile(audio_record[0]) as source:
                try:
                    audio = r.record(source)
                    text2 = r.recognize_google(audio)
                    print("Said:", text2)
                    
                    MSG = predict_model(audio_record[0])            
                    print("\n Your emotion depending on your voice now :",models_system(MSG))##emotion model
                    
                    newcleantext=NLP_on_text(text2)
                    sequences = loaded_tokenizer.texts_to_sequences([newcleantext])
                    max_sequence_length = 15  
                    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
                    predictions = modelSent.predict(padded_sequences)
                    predictedind1= np.argmax(predictions[0])# index of the highest probability
                    print("sentiment analysis of your speech context:\n","Negative precentage =",predictions[0][0],"\n Positive precentage =",predictions[0][1])
                                
                except sr.UnknownValueError:
                    print("could not understand audio")
                except sr.RequestError as e:
                    print(f"Could not request results: {e}")
            

        else:
            print("..")
    except sr.UnknownValueError:
        print(".")
    except sr.RequestError as e:
        print("Could not request results ; {0}".format(e))
        
        
        
    time.sleep(0.1)



# audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=2)
# sd.wait()
# sf.write(recorded_audio_file_path , audio_data, sample_rate)
# audio_record=glob(recorded_audio_file_path) 
# if audio_record:
#     with sr.AudioFile(audio_record[0]) as source:
#         try:
#             audio = r.record(source)
#             text = r.recognize_google(audio)
#             print("Said:", text)
#         except sr.UnknownValueError:
#             print("Google Speech Recognition could not understand audio")
#         except sr.RequestError as e:
#             print(f"Could not request results from Google Speech Recognition: {e}")


