import pandas as pd 
import matplotlib.pyplot as plt
from termcolor import colored
import klib
from flask import Flask, jsonify, render_template, request
from flask_swagger import swagger

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

plt.rcParams["figure.figsize"] = (10,6)
pd.set_option('max_colwidth',200)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 200)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

@app.route("/beta/qualitydata", methods=["GET"], endpoint='quality_data')
def quality_data():
  """# Loading Dataset"""
  df=request.args.get('Quality_Data')

  print(colored("Shape:", attrs=['bold']), df.shape,'\n', 
      colored('*'*100, 'red', attrs = ['bold']),
      colored("\nInfo:\n", attrs = ['bold']), sep = '')
  print(df.info(), '\n', 
      colored('*'*100, 'red', attrs = ['bold']), sep = '')
  print(colored("Number of Uniques:\n", attrs = ['bold']), df.nunique(),'\n',
      colored('*'*100, 'red', attrs = ['bold']), sep = '')
  print(colored("Missing Values:\n", attrs=['bold']), missing_values(df),'\n', 
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
def missing_values(df):
   missing_number = df.isnull().sum().sort_values(ascending = False)
   missing_percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending = False)
   missing_values = pd.concat([missing_number, missing_percent], axis = 1, keys = ['Missing_Number', 'Missing_Percent'])
   missing_values[missing_values['Missing_Number'] > 0]

def multicolinearity_control(df):
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

def duplicate_values(df):
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

    """# Data Summary"""

    quality_data(df)

    """# Missing Values, Multicolienaity and Duplicated Values"""

    multicolinearity_control(df)

    duplicate_values(df)

    missing_values(df)

    df.sample(2)

    klib.missingval_plot(df)

"""# DATA QUALITY FUNCTION"""

def Quality_Check(df):
   print("*****************************************DATA SUMMARY***********************************************")
   print(quality_data(df))
   print("****************************************MISSING VALUES**********************************************")
   print(missing_values(df))
   print(colored("Shape:", attrs=['bold']), df.shape,'\n', 
          colored('*'*100, 'red', attrs = ['bold']),
          colored("\nInfo:\n", attrs = ['bold']), sep = '')
   print("***************************************DUPLICATED VALUES********************************************")
   print(duplicate_values(df))
   print(colored("Shape:", attrs=['bold']), df.shape,'\n', 
          colored('*'*100, 'red', attrs = ['bold']),
          colored("\nInfo:\n", attrs = ['bold']), sep = '')
   print("*************************************MULTICOLINEARITY CHECK*****************************************")
   multicolinearity_control(df)
   Quality_Check(df)

   return jsonify(f"Result Accuracy:")
