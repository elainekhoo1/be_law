import json
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import words  # To import all english words
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import json
from sklearn.base import BaseEstimator, TransformerMixin
import spacy
import os
os.chdir(r'D:\00 Self-Learnings\0 Learning and Wellbeing (LAW)\be_law')

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 50)
pd.set_option('display.min_rows', 20)

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')
nltk.download('stopwords')

nlp = spacy.load('en_core_web_sm')

stopwords = set(stopwords.words('english'))
en_words = set(words.words())
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


class TextPreProcessor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def normalize_contractions(self, text): #
        english_contractions= json.loads(open("data/english_contractions.json", "r").read())
        normalized_text = []
        for t in text.split(' '):
            if t.lower() in english_contractions:
                t = english_contractions[t.lower()]
            normalized_text.append(t)
        return ' '.join(normalized_text)

    def text_normalizer(self, text):
        normalized_text = text.lower()
        # normalized_text = re.sub("[a-zA-Z]+[ ]bin[ ][a-zA-Z]+", "", normalized_text)
        # normalized_text = re.sub("[a-zA-Z]+[ ]binti[ ][a-zA-Z]+", "", normalized_text)
        normalized_text =normalized_text.replace("not assigned","")
        normalized_text = re.sub(r"_-\'",'', normalized_text)
        normalized_text = re.sub(r"\n,",' ', normalized_text)
        normalized_text = re.sub(r"[^a-z ]",' ', normalized_text)
        normalized_text = re.sub(r"[ ]+",' ', normalized_text)
        return normalized_text

    def tokenize(self, text): return word_tokenize(text)

    def stem_lemmatize_stopwordremoval(self, tokens):

        #Tag dictionary
        tag_dict = {"J": wn.ADJ,
                    "N": wn.NOUN,
                    "V": wn.VERB,
                    "R": wn.ADV}

        newTokens = []

        tag_words = nltk.pos_tag(tokens)  # apply pos tagging of all tokens at once
        tags = [tag[1][0] for tag in tag_words]
        tags = [tag_dict.get(tag, None) for tag in tags]

        for i in range(len(tokens)):

            token = tokens[i]
            tag = tags[i]
            if token not in stopwords: pass
            lemma = token
            if tag != None: lemma = lemmatizer.lemmatize(token, tag)
            if lemma == token:
                stem = stemmer.stem(token)
                lemma = stem if stem in en_words else lemma
            if len(lemma) > 2:  newTokens.append(lemma)
        return newTokens

    def preProcessIt(self, text):
        normalized_text = self.text_normalizer(text)
        preprocessed_text = self.stem_lemmatize_stopwordremoval(self.tokenize(normalized_text))
        preprocessed_text = " ".join(preprocessed_text)
        return preprocessed_text

    def fit(self, X, y=None): return self

    #Method that describes what we need this transformer to do
    def transform(self, X, y=None): return [self.preProcessIt(x) for x in X]


'''While there are a lot of person names, it would be better if we drop them off from the texts
we can use spaCy to do so using named entitiy recognition feature
Later on, we will learn about named entitiy recognition as part of knowledge extraction '''
def drop_person_names(text):
    doc = nlp(text)

    for x in doc.ents:
        if x.label_ == 'PERSON':
            text = text.replace(x.text, 'x')
    return text


if __name__ == "__main__":


    data = pd.read_csv("./law2021_MVP/data/data_scientist_jobstreet.csv")
    data['cleaned_job_description'] = data['job_details'].apply(drop_person_names)
    preProcessor = TextPreProcessor()

    data['processed_job_description'] = data['cleaned_job_description'].apply(preProcessor.preProcessIt)
    # data['normalize_job_description'] = data['cleaned_job_description'].apply(preProcessor.text_normalizer)
    # data['tokenized_job_description'] = data['normalize_job_description'].apply(preProcessor.tokenize)
    # data['lemmatized_job_description'] = data['tokenized_job_description'].apply(preProcessor.stem_lemmatize_stopwordremoval)

