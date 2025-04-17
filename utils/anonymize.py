import spacy
import re
from collections import defaultdict



#load spaCy NLP model once can be downloaed in venv 
nlp = spacy.load("en_core_web_sm")
# Entities we want to anonymize
NER_TYPES_TO_REPLACE = {"PERSON", "ORG", "GPE", "DATE"}

#function to anonymize sensitive data using spacy and repace it with placeholders 
def anonymize_text(text):
    doc = nlp(text)
    anonymized_tokens = []
    entity_counters = defaultdict(int)  #make a dict to hold various names in series like person_x1, person_x2 etc
    replacement_map = {}

    for token in doc:
        ent_type = token.ent_type_
        if ent_type in NER_TYPES_TO_REPLACE:
            key = f"{ent_type}::{token.text}"  #combine type and value for uniqueness
            if key not in replacement_map:
                entity_counters[ent_type] += 1
                replacement_map[key] = f"{ent_type}_X{entity_counters[ent_type]}"
            anonymized_tokens.append(replacement_map[key])
        else:
            anonymized_tokens.append(token.text)

    anonymized_text = " ".join(anonymized_tokens)

    #find names in article of aurther like -Author Name then anonymize it
    def dash_name_replacer(match):
        name = match.group(1)
        key = f"PERSON::{name}"
        if key not in replacement_map:
            entity_counters["PERSON"] += 1
            replacement_map[key] = f"PERSON_X{entity_counters['PERSON']}"
        return replacement_map[key]

    anonymized_text_1 = re.sub(r"[-–]\s?([A-Z][a-z]+\s[A-Z][a-z]+)", dash_name_replacer, anonymized_text)

    #find names in paragraph that have names like Full Name (2 capitals words one after another) eg Shubhendam Shrotriya
    def full_name_replacer(match):
        name = match.group(0)
        key = f"PERSON::{name}"
        if key not in replacement_map:
            entity_counters["PERSON"] += 1
            replacement_map[key] = f"PERSON_X{entity_counters['PERSON']}"
        return replacement_map[key]

    anonymized_text = re.sub(r"\b([A-Z][a-z]+)\s([A-Z][a-z]+)\b", full_name_replacer, anonymized_text_1)

    return anonymized_text
