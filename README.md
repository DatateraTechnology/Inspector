---------------------------------------------------------------Data Quality Inspector Model------------------------------------------------------------------------
	Model aims to detect the quality of a given dataset by using Python libraries. 
1.	Introduction
This notebook aims to detect the quality of a given text dataset. There are couple of metrics need to considered in terms of data quality. These are;
a)	Completeness-This measures whether all the necessary data is present in a specific dataset. You can think about completeness in one of two ways: at the record level or at the attribute level. Measuring completeness at the attribute level is a little more complex however, as not all fields will be mandatory.
b)	Accuracy-How accurately does your data reflect the real-world object? In the financial sector, data accuracy is usually black or white – it either is or isn’t accurate. That’s because the number of pounds and pennies in an account is a precise number. Data accuracy is critical in large organizations, where the penalties for failure are high.
c)	Consistency-Maintaining synchronicity between different databases is essential. To ensure data remains consistent on a daily basis, software systems are often the answer.
d)	Validity-Validity is a measure of how well data conforms to required value attributes. For example, ensuring dates conform to the same format, i.e., date/month/year or month/date/year.
e)	Timeliness-Timeliness reflects the accuracy of data at a specific point in time. An example of this is when a customer moves to a new house, how timely are they in informing their bank of their new address? Few people do this immediately, so there will be a negative impact on the timeliness of their data.
f)	Integrity-To ensure data integrity, it’s important to maintain all the data quality metrics we’ve mentioned above as your data moves between different systems. Typically, data stored in multiple systems breaks data integrity.
In this notebook, we only looked into missing values, duplicated values, multicolinearity and erroneous values for a given dataset in order to analyze the metrics mentioned above.

2.	Criteria Used for Data Quality
High Quality Data Criteria
1.	Overall Missing Value percentage less than %5 and,
2.	Overall Duplicated Value percentage less than %2 and,
3.	No Multicolinearity (Correlation between columns NOT higher than %90) and,
4.	Overall Erroneous Data percentage is less than %2.
3.	Technologies Used
Python Libraries
a)	Ydata-Quality- ydata_quality is an open-source python library for assessing Data Quality throughout the multiple stages of a dataset.
b)	Klib- klib is a Python library for importing, cleaning, analyzing and preprocessing data.

4.	Output
The model evaluates the high quality data criteria and gives whether the given data set is high quality or low quality.

----------------------------------------------------------Sensitive Data Indicator Model-------------------------------------------------------------------------------
	Model aims to detect sensitive data for a given dataset. Model is created by using Convolutional Neural Networks (CNN) and Name Entity Recognition (NER).
1.	Introduction
This notebook aims to capture Sensitive Data (Name, email address, password, phone number, date of birth etc) for a given dataset. Two different models were used and merged in one notebook. 
First Model uses Convolutional Neural Networks (CNN), this model is trained with Sensitive and Non Sensitive Datasets. Output gives the probability of sensitivity for a given sentence. This model is best for a text (sentence like) input data.
Second Model uses Presidio Analyzer and Presidio Anonymizer. Anonymizing data is optional. Name Entity Recogntion (NER) models are pretrained models and they don't need to be trained. Entities such as CREDIT_CARD, IBAN_CODE, EMAIL_ADDRESS, BANK_NUMBER can be detected and anonymized(optional) with this model. Output gives detection of sensitive data for chosen entities. This model can be used for text (sentence like) input data or Data Frames as input.
2.	Technologies Used
Python Libraries
a)	Tensor Flow-Tensor Flow is a foundation library that can be used to create Deep Learning models directly or by using wrapper libraries that simplify the process built on top of TensorFlow.
b)	Keras-Keras is a high-level neural network library that runs on top of TensorFlow.
c)	Presidio Analyzer-The Presidio analyzer is a Python based service for detecting PII entities in text.
d)	Presidio Anonymizer-The Presidio anonymizer is a Python based module for anonymizing detected PII text entities with desired values.
e)	spaCy is a free open-source library for Natural Language Processing in Python. It features NER, POS tagging, dependency parsing, word vectors and more.
Hugging Face 
Hugging Face is a community and data science platform that provides: Tools that enable users to build, train and deploy ML models based on open source (OS) code and technologies.
3.	Output
First model gives the probability of sensitive data for a given sentence-like input.  
Second Model detects sensitive data for a given test dataset.
 



