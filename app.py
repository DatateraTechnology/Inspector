from flask import Flask, jsonify, render_template, request
from flask_swagger import swagger
from azure.storage.blob import BlobClient
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import pandas as pd
import matplotlib.pyplot as plt
from termcolor import colored
import klib
import tensorflow as tf
import json, random, numpy as np
import urllib.request, json, seaborn, os, uuid
from typing import List
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer import AnalyzerEngine
from typing import List
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts
import os
from ydata_quality import DataQuality
from ydata_quality.erroneous_data import ErroneousDataIdentifier
from ydata_quality.duplicates import DuplicateChecker

app = Flask(__name__)

@app.route("/")
def spec():
    swag = swagger(app)
    swag['info']['version'] = "v.1.0"
    swag['info']['title'] = "Welcome to Datatera Beta"
    return jsonify(swag)

@app.route('/api')
def get_api():
    return render_template('swaggerui.html')

#Check Credit Card Number
@app.route("/beta/checkcreditcardno/<string:candidate_value>", methods=["GET"], endpoint='check_credit_card_no')
def check_credit_card_no(candidate_value):

  candidate_value = re.sub('\D', '', str(candidate_value))
  x = re.search("^(?:4[0-9]{12}(?:[0-9]{3})?|[25][1-7][0-9]{14}|6(?:011|5[0-9][0-9])[0-9]{12}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|(?:2131|1800|35\d{3})\d{11})$", candidate_value)

  if x:
   return jsonify(f"Credit Card Number is detected!!")
  else:
   return jsonify(f"No Credit Card Number is detected!!")

@app.route("/beta/sensitivedatacnn", methods=["GET"], endpoint='sensitive_data_cnn')
def sensitive_data_cnn():

  Sensitive_Data = request.args.get('Sensitive_Data')
  Nonsensitive_Data = request.args.get('Nonsensitive_Data')

  sensitive_datafile = Sensitive_Data
  nonsensitive_datafile = Nonsensitive_Data

  vocab_size = 3000
  embedding_dim = 32
  max_length = 60
  truncation_type='post'
  padding_type='post'
  oov_tok = "<OOV>"
  training_size = 20000

  dataList = []
  sentences = []
  labels = []

  stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
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
              "yourself", "yourselves" ]

  def loadDataset(filename):
    #with open(filename, 'r') as f:
    f = urllib.request.urlopen(filename)
    datastore = json.load(f)
    for item in datastore:
      sentence = item['data']
      label = item['is_sensitive']
      for word in stopwords:
        token = " " + word + " "
        sentence = sentence.replace(token, " ")
      dataList.append([sentence, label])

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

  sen=pd.read_json(sensitive_datafile)
  sen.head()

  nonsen=pd.read_json(nonsensitive_datafile)
  nonsen.head()

  tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

  tokenizer.fit_on_texts(training_sentences)

  word_index = tokenizer.word_index
  print("Size of word index:", len(word_index))

  with open("word_index.json", "w") as outfile:  
      json.dump(word_index, outfile)
      print("Saving the word index as JSON")

  training_sequences = tokenizer.texts_to_sequences(training_sentences)
  training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=truncation_type)

  validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
  validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=truncation_type)

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

  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

  model.summary()
  num_epochs = 6

  history = model.fit(training_padded, 
                      training_labels, 
                      epochs=num_epochs, 
                      validation_data=(validation_padded, validation_labels), 
                      verbose=1)

  def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])

    local_path = os.path.expanduser("~/Result")
    if not os.path.exists(local_path):
            os.makedirs(os.path.expanduser("~/Result"))
    local_file_name = "Result_" + str(uuid.uuid4()) + ".png"
    full_path_to_file = os.path.join(local_path, local_file_name)

    plt.savefig(full_path_to_file) 

    blob = BlobClient.from_connection_string("DefaultEndpointsProtocol=https;AccountName=datateraalpha;AccountKey=W890/aL1FprdvAsAV4xXpOof1BZQm5Ujb044t8s2XaHFeA0QBYlffI+KYG72uQCg6Ly8SNkeRki8cOwma4co9A==;EndpointSuffix=core.windows.net", container_name="beta", blob_name=local_file_name)
    with open(full_path_to_file, "rb") as data:
      blob.upload_blob(data)  
    url = "https://datateraalpha.blob.core.windows.net/beta/" + local_file_name
    return url

  url_accuracy = plot_graphs(history, "accuracy")
  url_loss = plot_graphs(history, "loss")  

  print('Confusion Matrix')
  y_predicted = model.predict(validation_padded)
  y_predicted_labels = y_predicted > 0.5

  size = np.size(y_predicted_labels)
  y_predicted_labels = y_predicted_labels.reshape(size, )

  for i in range (1, 5):
    total = i * size // 4
    cm = tf.math.confusion_matrix(labels=validation_labels[0:total],predictions=y_predicted_labels[0:total])

    cm_np = cm.numpy()
    conf_acc = (cm_np[0, 0] + cm_np[1, 1])/ np.sum(cm_np) * 100
    print("Accuracy for", str(total), "Test Data = ", conf_acc)

    plt.figure(figsize = (10,7))
    seaborn.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix for " + str(total) + " Test Data")
    plt.xlabel('Predicted')
    plt.ylabel('Expected')

  model_json = model.to_json()
  local_file_name = "Model_" + str(uuid.uuid4()) + ".json"
  blob = BlobClient.from_connection_string("DefaultEndpointsProtocol=https;AccountName=datateraalpha;AccountKey=W890/aL1FprdvAsAV4xXpOof1BZQm5Ujb044t8s2XaHFeA0QBYlffI+KYG72uQCg6Ly8SNkeRki8cOwma4co9A==;EndpointSuffix=core.windows.net", container_name="beta", blob_name=local_file_name)
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
    if predictions[i][0]>0.4:
      print("This Data is More Likely Sensitive - "+ sentence[i])
    else:
      print("This Data is Less Likely Sensitive - "+ sentence[i])
  
  return jsonify(f"Result Accuracy: {url_accuracy} Result Loss: {url_loss}")

@app.route("/beta/sensitivedatapre", methods=["GET"], endpoint='sensitive_data_pre')
def sensitive_data_pre():
  
    url=request.args.get('Anonymize_Data')
    datax=pd.read_csv(url)

    datax.head()

    def anonymize():
        for i in datax:
          datax[i] = datax[i].apply(lambda x: predict_fn({"inputs": x,"parameters": {"anonymize": True}},AnalyzerEngine())["anonymized"])
        return jsonify(f"Found:{datax}")

    anonymize()

    def not_anonymize(datax):
        for i in datax:
          datax[i] = datax[i].apply(lambda x: predict_fn({"inputs": x,"parameters": {"anonymize": False}},AnalyzerEngine()))
        return jsonify(f"Found:{datax}")
          
    not_anonymize(datax)

    def analyze(self, text: str, entities: List[str] = None, nlp_artifacts: NlpArtifacts = None
        ) -> List[RecognizerResult]:
            """
            Extracts entities using Transformers pipeline
            """
            results = []

            # keep max sequence length in mind
            predicted_entities = self.pipeline(text)
            if len(predicted_entities) > 0:
                for e in predicted_entities:
                    converted_entity = self.label2presidio[e["entity_group"]]
                    if converted_entity in entities or entities is None:
                        results.append(
                            RecognizerResult(
                                entity_type=converted_entity, start=e["start"], end=e["end"], score=e["score"]
                            )
                        )
            return (f"{results}")

    def model_fn(model_dir):
        analyzer = AnalyzerEngine()
        #analyzer.registry.add_recognizer(transformers_recognizer)
        return (f"{analyzer}")

    def predict_fn(data, analyzer):
        sentences = data.pop("inputs", data)
        DEFAULT_ANOYNM_ENTITIES = [
          "CREDIT_CARD",
          "CRYPTO",
          "DATE_TIME",
          "EMAIL_ADDRESS",
          "IBAN_CODE",
          "IP_ADDRESS",
          "NRP",
          "LOCATION",
          "PERSON",
          "PHONE_NUMBER",
          "MEDICAL_LICENSE",
          "URL",
          "US_SSN","US_BANK_NUMBER"]
        if "parameters" in data:
            anonymization_entities = data["parameters"].get("entities", DEFAULT_ANOYNM_ENTITIES)
            anonymize_text = data["parameters"].get("anonymize", False)
        else:
            anonymization_entities = DEFAULT_ANOYNM_ENTITIES
            anonymize_text = False

        results = analyzer.analyze(text=sentences, entities=anonymization_entities, language="en")
      
        engine = AnonymizerEngine()
        if anonymize_text:
            result = engine.anonymize(text=sentences, analyzer_results=results)
            return (f"anonymized:{result.text}")
        return (f"Found:{[entity.to_dict() for entity in results]}")
    
    """Testing Model for a given sentence"""

    sentence="""
    Hello, my name is Zack and I live in Istanbul.
    I work for DataTera Tech. 
    You can call me at (212) 555-1234.
    My credit card number is 4001-9192-5753-7193 and my crypto wallet id is 16Yeky6GMjeNkAiNcBY7ZhrLoMSgg1BoyZ.
    My passport number : 191280342.
    This is a valid International Bank Account Number: IL150120690000003111111.
    My social security number is 078-05-1126.  My driver license number is 1234567A."""

    data = {
      "inputs": sentence,
    }

    predict_fn(data,AnalyzerEngine())

    """Detecting only Credit Card- if we only want to detect credit card we need to mention in the "entities" part"""

    data = {
      "inputs": sentence,
      "parameters": {
        "entities":["CREDIT_CARD"]
      }
    }

    predict_fn(data,AnalyzerEngine())

    """Anonymize (Optional) all entities- If we want to anonymize detected entites we need to make "anonymize":True"""

    data = {
      "inputs": sentence,
      "parameters": {
        "anonymize": True,
      }
    }

    print(predict_fn(data,AnalyzerEngine())[1])

    """Anonymize only PERSON and LOCATION in the text- we can anonymize any entities that we choose, in this example person and location were anonymized"""

    data = {
      "inputs": sentence,
      "parameters": {
        "anonymize": True,
        "entities":["PERSON","LOCATION"]
      }
    }

    print(predict_fn(data,AnalyzerEngine())[1])

    return jsonify(f"Anonymized/Nonanonymized Data:{datax}")

@app.route("/beta/qualitydata", methods=["GET"], endpoint='quality_data')
def quality_data():
      
  plt.rcParams["figure.figsize"] = (10,6)
  pd.set_option('max_colwidth',200)
  pd.set_option('display.max_rows', 1000)
  pd.set_option('display.max_columns', 200)
  pd.set_option('display.float_format', lambda x: '%.3f' % x)

  """# Loading Dataset"""
  url=request.args.get('Quality_Data')
  df=pd.read_csv(url)
  
  print(colored("Shape:", attrs=['bold']), df.shape,'\n', 
        colored('*'*100, 'red', attrs = ['bold']),
        colored("\nInfo:\n", attrs = ['bold']), sep = '')
  print(df.info(), '\n', 
        colored('*'*100, 'red', attrs = ['bold']), sep = '')
  print(colored("Number of Uniques:\n", attrs = ['bold']), df.nunique(),'\n',
        colored('*'*100, 'red', attrs = ['bold']), sep = '')
  print(colored("All Columns:", attrs = ['bold']), list(df.columns),'\n', 
        colored('*'*100, 'red', attrs = ['bold']), sep = '')

  df.columns = df.columns.str.lower().str.replace('&', '_').str.replace(' ', '_')
  print(colored("Columns after rename:", attrs = ['bold']), list(df.columns),'\n',
        colored('*'*100, 'red', attrs = ['bold']), sep = '')  
  print(colored("Descriptive Statistics \n", attrs = ['bold']), df.describe().round(2),'\n',
        colored('*'*100, 'red', attrs = ['bold']), sep = '') # Gives a statstical breakdown of the data.
  print(colored("Descriptive Statistics (Categorical Columns) \n", attrs = ['bold']), df.describe(include = object).T,'\n',
        colored('*'*100, 'red', attrs = ['bold']), sep = '') # Gives a statstical breakdown of the data.

  """# Functions for Missing Values, Multicolinearity and Duplicated Values"""
  def missing_values():
   missing_number = df.isnull().sum().sort_values(ascending = False)
   missing_percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending = False)
   missing_values = pd.concat([missing_number, missing_percent], axis = 1, keys = ['Missing_Number', 'Missing_Percent'])
   missing_values[missing_values['Missing_Number'] > 0]
   return missing_values[missing_values['Missing_Number'] > 0]

  def multicolinearity_control():
    feature = []
    collinear = []
    for col in df.corr().columns:
      for i in df.corr().index:
        if (abs(df.corr()[col][i]) > .9 and abs(df.corr()[col][i]) < 1):
          feature.append(col)
          collinear.append(i)
          print(colored(f"Multicolinearity alert in between:{col} - {i}", 
                                  "red", attrs = ['bold']), df.shape,'\n',
                                  colored('*'*100, 'red', attrs = ['bold']), sep = '')

    if len(collinear)==0:
        print("No Multicoliearity, Correlation between collumns is NOT over %90")

  def duplicate_values():
    print(colored("Duplicate check...", attrs = ['bold']), sep = '')
    print("There are", df.duplicated(subset = None, keep = 'first').sum(), "duplicated observations in the dataset.")
    duplicate_values = df.duplicated(subset = None, keep = 'first').sum()
    #if duplicate_values > 0:
        #df.drop_duplicates(keep = 'first', inplace = True)
        #print(duplicate_values, colored(" Duplicates were dropped!"),'\n',
              #colored('*'*100, 'red', attrs = ['bold']), sep = '')
      #     else:
      #         print(colored("There are no duplicates"),'\n',
      #               colored('*'*100, 'red', attrs = ['bold']), sep = '')     
            
      # def drop_columns(df, drop_columns):
      #     if drop_columns != []:
      #         df.drop(drop_columns, axis = 1, inplace = True)
      #         print(drop_columns, 'were dropped')
      #     else:
      #         print(colored('We will now check the missing values and if necessary, the related columns will be dropped!', attrs = ['bold']),'\n',
      #               colored('*'*100, 'red', attrs = ['bold']), sep = '')

  """# Missing Values, Multicolienaity and Duplicated Values"""

  dc = DuplicateChecker(df=df)

  results = dc.evaluate()
  results.keys()

  warnings = dc.get_warnings()

  exact_duplicates_out = dc.exact_duplicates()

  dc.duplicate_columns()

  edi = ErroneousDataIdentifier(df=df)

  edi.evaluate()

  edi.predefined_erroneous_data()

  df.sample(2)

  klib.missingval_plot(df)
  
  """# DATA QUALITY FUNCTION"""

  def Quality_Check():
   print("****************************************MISSING VALUES**********************************************")
   print(missing_values())
   print(colored("Shape:", attrs=['bold']), df.shape,'\n', 
          colored('*'*100, 'red', attrs = ['bold']),
          colored("\nInfo:\n", attrs = ['bold']), sep = '')
   print("***************************************DUPLICATED VALUES********************************************")
   print(duplicate_values())
   print(colored("Shape:", attrs=['bold']), df.shape,'\n', 
          colored('*'*100, 'red', attrs = ['bold']),
          colored("\nInfo:\n", attrs = ['bold']), sep = '')
   print("*************************************MULTICOLINEARITY CHECK*****************************************")
   multicolinearity_control()

  Quality_Check()

   """# KPI Function"""

  def KPI():
   print("**********************************NUMBER OF COLUMNS AND ROWS****************************************")
   print("There are", df.shape[0], "rows", df.shape[1],"columns and",df.shape[0]*df.shape[1],"entries in this dataset")
   print()
   print("****************************************MISSING VALUES**********************************************")
   print("Overall percentage of missing values is %", missing_values().mean()[1]*100)
   print("")
   print("***************************************DUPLICATED VALUES********************************************")
   print("There are",df.duplicated(subset = None, keep = 'first').sum(),"duplicated values.","Overall percentage is %", (df.duplicated(subset = None, keep = 'first').sum()/len(df))*100)
   print("")
   print("*************************************MULTICOLINEARITY CHECK*****************************************")
   multicolinearity_control()
   print("")
   print("******************************************ERRONEOUS DATA********************************************")
   ErroneousDataIdentifier(df=df).predefined_erroneous_data()
   edi.predefined_erroneous_data()
   print("Overall percentage of Eroneous Data is %",(edi.predefined_erroneous_data().sum()[0]/(df.shape[0]*df.shape[1]))*100)
   print()
   print("***************************************OVERALL DATA QUALITY*****************************************")
 
  KPI()

  """# KPI Assesement
    High Quality Data Criteria

    1.   Overall Missing Value percentage less than %5 and,
    2.   Overall Duplicated Value percentage less than %2 and,
    3.   No Multicolinearity (Correlation between columns NOT higher than %90) and,
    4.   Overall Erroneous Data percentage is less than %2.

  """

  if (missing_values().mean()[1] <.05) and (df.duplicated(subset = None, keep = 'first').sum()/len(df)<.02):
    return jsonify("HIGH QUALITY DATA")
  else:
    return jsonify("LOW QUALITY DATA")
