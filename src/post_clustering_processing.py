import pandas as pd
import numpy as np
import json
import re
import matplotlib.pyplot as plt

import os
os.chdir(r'D:\00 Self-Learnings\0 Learning and Wellbeing (LAW)\be_law')

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 50)
pd.set_option('display.min_rows', 20)


def bow_binary(data, vocabulary):
    bow = []
    for d in data:
        d_vec = [1 if term in d.split() else 0 for term in vocabulary ]
        bow.append(d_vec)
    return bow


def get_most_occuring_words(data, top_n=30, plot=True):
    def text_normalizer(text):
        normalized_text = text.lower()
        normalized_text = normalized_text.replace("not assigned", "")
        normalized_text = re.sub(r"_-\'", '', normalized_text)
        normalized_text = re.sub(r"\n,", ' ', normalized_text)
        normalized_text = re.sub(r"[^a-z ]", ' ', normalized_text)
        normalized_text = re.sub(r"[ ]+", ' ', normalized_text)
        return normalized_text

    data['normalized_job_position'] = data['job_position'].apply(text_normalizer)
    example_data = data['normalized_job_position'].values

    from sklearn.feature_extraction.text import CountVectorizer
    # Convert a collection of text documents to a matrix of token counts
    vect = CountVectorizer(binary=True)
    bow_bin = vect.fit_transform(example_data)
    vect_vocabulary_dictionary = vect.vocabulary_
    # len(vect.get_feature_names())
    vect_vocabulary_dictionary = dict(
        sorted(vect_vocabulary_dictionary.items(), key=lambda item: item[1], reverse=True))
    # bow_bin.toarray()
    # pd.DataFrame(columns = vect.get_feature_names(), data=bow_bin.toarray())
    topN_dict = dict(list(vect_vocabulary_dictionary.items())[1:top_n])
    if plot:
        plt.subplots(figsize=(16, 16))
        plt.bar(topN_dict.keys(), topN_dict.values())
        plt.xticks(rotation=45)
        plt.title(f"CountVectorizer of job position in cluster {data['nltk_clusters'].unique()[0]}")
        plt.savefig(f"./docs/CountVectorizer cluster {data['nltk_clusters'].unique()[0]}.")


if __name__ == "__main__":

    data = pd.read_csv("./data/data_clustered_v2.csv")
    data_subset = data[
        ['processed_job_description', 'nltk_clusters', 'manual_renamed_job_position_by_desc', 'job_position']]
    data_subset['nltk_clusters'].unique()
    # test_id2 = data_subset.query("nltk_clusters==2")
    # example_data = test_id2['normalized_job_position'].values
    # vocabulary = set(" ".join(example_data).split())
    # bow_binary(data, vocabulary)
    # pd.DataFrame(columns=vocabulary, data=bow_binary(example_data, vocabulary))

    data_subset.groupby(['nltk_clusters']).apply(get_most_occuring_words)
