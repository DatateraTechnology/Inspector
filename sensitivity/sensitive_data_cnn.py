from flask import  jsonify, request
from azure.storage.blob import BlobClient
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import random, numpy as np
import urllib.request, json, seaborn, os, uuid

import os
from sensitivity import Sensitive_Data_CNN_Response as sd
from os import path

config = path.relpath("config/app-config.json")
with open(config, "r") as f:
    config = json.load(f)

connect_str = config["blobStorage"]["sensitiveDataConnectionString"]
def sensitive_data_cnn():
    Sensitive_Data = request.args.get('Sensitive_Data')
    Nonsensitive_Data = request.args.get('Nonsensitive_Data')

    sensitive_datafile = Sensitive_Data
    nonsensitive_datafile = Nonsensitive_Data

    vocab_size = 3000
    embedding_dim = 32
    max_length = 60
    truncation_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 20000

    dataList = []
    sentences = []
    labels = []

    stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
                 "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
                 "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
                 "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
                 "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself",
                 "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it",
                 "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on",
                 "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
                 "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that",
                 "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these",
                 "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
                 "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what",
                 "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom",
                 "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
                 "yourself", "yourselves"]

    def loadDataset(filename):
        # with open(filename, 'r') as f:
        f = urllib.request.urlopen(filename)
        datastore = json.load(f)
        for item in datastore:
            sentence = item['data']
            label = item['is_sensitive']
            for word in stopwords:
                token = " " + word + " "
                sentence = sentence.replace(token, " ")
            dataList.append([sentence, label])

    ##
    loadDataset(sensitive_datafile)
    loadDataset(nonsensitive_datafile)

    random.shuffle(dataList)

    print("Dataset Size: ", len(dataList))

    for item in dataList:
        sentences.append(item[0])
        labels.append(item[1])

    training_sentences = sentences[0:training_size]
    validation_sentences = sentences[training_size:]
    training_labels = labels[0:training_size]
    validation_labels = labels[training_size:]

    print("Training Dataset Size: ", len(training_sentences))
    print("Sample Training Data:", training_sentences[1])
    print("Validation Dataset Size: ", len(validation_sentences))
    print("Sample Validation Data:", validation_sentences[1])

    sen = pd.read_json(sensitive_datafile)
    sen.head()

    nonsen = pd.read_json(nonsensitive_datafile)
    nonsen.head()

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

    tokenizer.fit_on_texts(training_sentences)

    word_index = tokenizer.word_index
    print("Size of word index:", len(word_index))

    with open("word_index.json", "w") as outfile:
        json.dump(word_index, outfile)
        print("Saving the word index as JSON")

    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type,
                                    truncating=truncation_type)

    validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
    validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type,
                                      truncating=truncation_type)

    training_padded = np.array(training_padded)
    training_labels = np.array(training_labels)
    validation_padded = np.array(validation_padded)
    validation_labels = np.array(validation_labels)

    DESIRED_ACCURACY = 0.999

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('accuracy') > DESIRED_ACCURACY:
                print("Reached 99.9% accuracy so cancelling training!")
                self.model.stop_training = True

    callbacks = myCallback()

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()
    num_epochs = 6

    history = model.fit(training_padded,
                        training_labels,
                        epochs=num_epochs,
                        validation_data=(validation_padded, validation_labels),
                        verbose=1)

    def plot_graphs(history, string):
        plt.plot(history.history[string])
        plt.plot(history.history['val_ ' + string])
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.legend([string, 'val_ ' + string])

        local_path = os.path.expanduser("~/Result")
        if not os.path.exists(local_path):
            os.makedirs(os.path.expanduser("~/Result"))
        local_file_name = "Result_" + str(uuid.uuid4()) + ".png"
        full_path_to_file = os.path.join(local_path, local_file_name)

        plt.savefig(full_path_to_file)

        blob = BlobClient.from_connection_string \
            (connect_str,
             container_name="beta", blob_name=local_file_name)
        with open(full_path_to_file, "rb") as data:
            blob.upload_blob(data)
        blob_url = config["blobStorage"]["datateraAlphaBlobUrl"]
        url = blob_url + local_file_name
        return url

    url_accuracy = plot_graphs(history, "accuracy")
    url_loss = plot_graphs(history, "loss")

    print('Confusion Matrix')
    y_predicted = model.predict(validation_padded)
    y_predicted_labels = y_predicted > 0.5

    size = np.size(y_predicted_labels)
    y_predicted_labels = y_predicted_labels.reshape(size, )

    for i in range(1, 5):
        total = i * size // 4
        cm = tf.math.confusion_matrix(labels=validation_labels[0:total], predictions=y_predicted_labels[0:total])

        cm_np = cm.numpy()
        conf_acc = (cm_np[0, 0] + cm_np[1, 1]) / np.sum(cm_np) * 100
        print("Accuracy for", str(total), "Test Data = ", conf_acc)

        plt.figure(figsize=(10, 7))
        seaborn.heatmap(cm, annot=True, fmt='d')
        plt.title("Confusion Matrix for " + str(total) + " Test Data")
        plt.xlabel('Predicted')
        plt.ylabel('Expected')

    model_json = model.to_json()
    local_file_name = "Model_" + str(uuid.uuid4()) + ".json"
    blob = BlobClient.from_connection_string \
        (connect_str,
         container_name="beta", blob_name=local_file_name)
    blob.upload_blob(model_json)

    print("Saved the model successfully")
    print("Model converted to JSON successfully")

    sentence = ["His Name is John",
                "Her name is Janet",
                "date of birth:17-09-1972",
                "passport number: 123456789",
                "Phone Number is 555555555",
                "Credit card number 341-547-787",
                "Username:John",
                "Password:1345",
                "DataTera:Global Data Source for AI models",
                "Her Race is White",
                "Race:Cacuasian",
                "Eye Color: Brown",
                "That was an awsome movie",
                "Her glucose level was very high",
                "She has sent her e-mail address for the meeting",
                "Her email:janet@gmail.com",
                "e-mail:john@gmail.com",
                "fax number:12456789",
                "Please do not share your private information",
                "Credit card: 1234-1234-1234-1234"]
    sequences = tokenizer.texts_to_sequences(sentence)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=truncation_type)
    predictions = model.predict(padded)
    for i in range(len(predictions)):
        print(predictions[i][0])
        if predictions[i][0] > 0.4:
            print("This Data is More Likely Sensitive -  " + sentence[i])
        else:
            print("This Data is Less Likely Sensitive -  " + sentence[i])

    sd.Sensitive_Data_CNN_Response(url_accuracy, url_loss)
    return sd
