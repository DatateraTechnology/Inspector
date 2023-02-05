from flask import jsonify, request
import pandas as pd
import matplotlib.pyplot as plt
from termcolor import colored
import klib
import  json
#from ydata_quality import DataQuality
#from ydata_quality.erroneous_data import ErroneousDataIdentifier
#from ydata_quality.duplicates import DuplicateChecker
from os import path

config = path.relpath("config/app-config.json")
with open(config, "r") as f:
    config = json.load(f)

def quality_data():
    plt.rcParams["figure.figsize"] = (10, 6)
    pd.set_option('max_colwidth', 200)
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 200)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)

    """# Loading Dataset"""
    url = request.args.get('Quality_Data')
    df = pd.read_csv(url)

    print(colored("Shape:", attrs=['bold']), df.shape, '\n',
          colored('*' * 100, 'red', attrs=['bold']),
          colored("\nInfo:\n", attrs=['bold']), sep='')
    print(df.info(), '\n',
          colored('*' * 100, 'red', attrs=['bold']), sep='')
    print(colored("Number of Uniques:\n", attrs=['bold']), df.nunique(), '\n',
          colored('*' * 100, 'red', attrs=['bold']), sep='')
    print(colored("All Columns:", attrs=['bold']), list(df.columns), '\n',
          colored('*' * 100, 'red', attrs=['bold']), sep='')

    df.columns = df.columns.str.lower().str.replace('&', '_').str.replace(' ', '_')
    print(colored("Columns after rename:", attrs=['bold']), list(df.columns), '\n',
          colored('*' * 100, 'red', attrs=['bold']), sep='')
    print(colored("Descriptive Statistics \n", attrs=['bold']), df.describe().round(2), '\n',
          colored('*' * 100, 'red', attrs=['bold']), sep='')  # Gives a statstical breakdown of the data.
    print(colored("Descriptive Statistics (Categorical Columns) \n", attrs=['bold']), df.describe(include=object).T,
          '\n',
          colored('*' * 100, 'red', attrs=['bold']), sep='')  # Gives a statstical breakdown of the data.

    """# Functions for Missing Values, Multicolinearity and Duplicated Values"""

    def missing_values():
        missing_number = df.isnull().sum().sort_values(ascending=False)
        missing_percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
        missing_values = pd.concat([missing_number, missing_percent], axis=1,
                                   keys=['Missing_Number', 'Missing_Percent'])
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
                                  "red", attrs=['bold']), df.shape, '\n',
                          colored('*' * 100, 'red', attrs=['bold']), sep='')

        if len(collinear) == 0:
            print("No Multicoliearity, Correlation between collumns is NOT over %90")

    def duplicate_values():
        print(colored("Duplicate check...", attrs=['bold']), sep='')
        print("There are", df.duplicated(subset=None, keep='first').sum(), "duplicated observations in the dataset.")
        duplicate_values = df.duplicated(subset=None, keep='first').sum()
        # if duplicate_values > 0:
        # df.drop_duplicates(keep = 'first', inplace = True)
        # print(duplicate_values, colored(" Duplicates were dropped!"),'\n',
        # colored('*'*100, 'red', attrs = ['bold']), sep = '')
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

    multicolinearity_control()

    duplicate_values()

    missing_values()
    dc = ''
    #dc = DuplicateChecker(df=df)

    results = dc.evaluate()
    results.keys()

    warnings = dc.get_warnings()

    exact_duplicates_out = dc.exact_duplicates()

    dc.duplicate_columns()
    edi = ''
    #edi = ErroneousDataIdentifier(df=df)

    edi.evaluate()

    edi.predefined_erroneous_data()

    df.sample(2)

    klib.missingval_plot(df)

    """# DATA QUALITY FUNCTION"""

    def Quality_Check():
        print("****************************************MISSING VALUES**********************************************")
        print(missing_values())
        print(colored("Shape:", attrs=['bold']), df.shape, '\n',
              colored('*' * 100, 'red', attrs=['bold']),
              colored("\nInfo:\n", attrs=['bold']), sep='')
        print("***************************************DUPLICATED VALUES********************************************")
        print(duplicate_values())
        print(colored("Shape:", attrs=['bold']), df.shape, '\n',
              colored('*' * 100, 'red', attrs=['bold']),
              colored("\nInfo:\n", attrs=['bold']), sep='')
        print(dc.duplicate_columns())
        print("*************************************MULTICOLINEARITY CHECK*****************************************")
        multicolinearity_control()
        print("*****************************************ERRONEOUS DATA*********************************************")
       # print(ErroneousDataIdentifier(df=df).predefined_erroneous_data())

    Quality_Check()

    def KPI():
        print("**********************************NUMBER OF COLUMNS AND ROWS****************************************")
        print("There are", df.shape[0], "rows", df.shape[1], "columns and", df.shape[0] * df.shape[1],
              "entries in this dataset")
        print()
        print("****************************************MISSING VALUES**********************************************")
        print("Overall percentage of missing values is %", missing_values().mean()[1] * 100)
        print("")
        print("***************************************DUPLICATED VALUES********************************************")
        print("There are", df.duplicated(subset=None, keep='first').sum(), "duplicated values.",
              "Overall percentage is %", (df.duplicated(subset=None, keep='first').sum() / len(df)) * 100)
        print("")
        print("*************************************MULTICOLINEARITY CHECK*****************************************")
        multicolinearity_control()
        print("")
        print("******************************************ERRONEOUS DATA********************************************")
       # ErroneousDataIdentifier(df=df).predefined_erroneous_data()
        edi.predefined_erroneous_data()
        # print("Overall percentage of Eroneous Data is %",(edi.predefined_erroneous_data().sum()[0]/(df.shape[0]*df.shape[1]))*100)
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

    if (missing_values().mean()[1] < .05) and (df.duplicated(subset=None, keep='first').sum() / len(df) < .02):
        return jsonify("HIGH QUALITY DATA")
    else:
        return jsonify("LOW QUALITY DATA")