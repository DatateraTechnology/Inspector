from flask import jsonify
import pandas as pd
import json
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer import AnalyzerEngine
from typing import List
import os
from os import path

config = path.relpath("config/app-config.json")

with open(config, "r") as f:
    config = json.load(f)
    
def sensitive_data_pre(url):
    os.system("spacy download en_core_web_lg")

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
        # analyzer.registry.add_recognizer(transformers_recognizer)
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
            "US_SSN", "US_BANK_NUMBER"]
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

    sentence = """
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

    predict_fn(data, AnalyzerEngine())

    """Detecting only Credit Card- if we only want to detect credit card we need to mention in the "entities" part"""

    data = {
        "inputs": sentence,
        "parameters": {
            "entities": ["CREDIT_CARD"]
        }
    }

    predict_fn(data, AnalyzerEngine())

    """Anonymize (Optional) all entities- If we want to anonymize detected entites we need to make "anonymize":True"""

    data = {
        "inputs": sentence,
        "parameters": {
            "anonymize": True,
        }
    }

    print(predict_fn(data, AnalyzerEngine())[1])

    """Anonymize only PERSON and LOCATION in the text- we can anonymize any entities that we choose, in this example person and location were anonymized"""

    data = {
        "inputs": sentence,
        "parameters": {
            "anonymize": True,
            "entities": ["PERSON", "LOCATION"]
        }
    }

    print(predict_fn(data, AnalyzerEngine())[1])

    #url = request.args.get('Anonymize_Data')
    data = pd.read_csv(url)

    data.head()

    def anonymize(data):
        for i in data:
            data[i] = data[i].apply(
                lambda x: predict_fn({"inputs": x, "parameters": {"anonymize": True}}, AnalyzerEngine())["anonymized"])
        return jsonify(f"Found:{data}")

    anonymize(data)

    data = pd.read_csv(url)

    data.head()

    def not_anonymize(data):
        for i in data:
            data[i] = data[i].apply(
                lambda x: predict_fn({"inputs": x, "parameters": {"anonymize": False}}, AnalyzerEngine()))
        return jsonify(f"Found:{data}")

    not_anonymize(data)