# For python >3.8 -> pip install tensorflow==2.2.0rc4
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from nltk.cluster import KMeansClusterer
import nltk
from pathlib import Path

from pandas.tests.io.json.conftest import orient

os.chdir(r'D:\00 Self-Learnings\0 Learning and Wellbeing (LAW)\be_law')

# module_url = r"D:\00 Self-Learnings\0 Learning and Wellbeing (LAW)\be_law\law2021_MVP\universal-sentence-encoder-large_5"
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
# logging.set_verbosity(logging.ERROR)

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 50)
pd.set_option('display.min_rows', 20)


def embed(input):
    # sentence embedded vector
    return model(input)


def semantic_clustering(data, col_name, dbscan_eps=0.5, metric='cosine'):
    import numpy as np
    from sklearn.cluster import dbscan
    from sklearn.metrics.pairwise import cosine_similarity
    
    def cosine(A, B):
        score = cosine_similarity([A], [B])[0][0]
        return 1 - score

    def get_sim(i, j):
        x, y = int(i[0]), int(j[0])
        x, y = data['embed'][x], data['embed'][y]
        score = round(tf.reduce_sum(tf.multiply(x, y), axis=1).numpy()[0] * 100, 2)
        return (100 - score)/100

    X = np.arange(len(data)).reshape(-1, 1)
    
    if metric == 'cosine':
        db_metric = cosine
    else:
        db_metric = get_sim
        data["embed"] = data[col_name].apply(lambda x: tf.nn.l2_normalize(embed(tf.constant([x])), axis=1))

    clusters = dbscan(X, metric=db_metric, eps=dbscan_eps, min_samples=10, algorithm='auto')[1] #finetune eps
    data[f"db_cluster_{metric}_sim"] = clusters
    return data


def nltk_kmeans_clustering(data, col_name, n_clusters=5):

    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=500)
    X = vec.fit_transform(data[col_name])

    # create clusterer object based on cosine similairy
    kMeans = KMeansClusterer(n_clusters, distance=nltk.cluster.util.cosine_distance, repeats=100)

    assigned_clusters = kMeans.cluster(X.toarray(), assign_clusters=True)
    data['nltk_clusters'] = assigned_clusters
    return data


def sklearn_kmeans_clustering(data, col_name, n_clusters=5):
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(max_features=500)
    X = vec.fit_transform(data[col_name])

    # create clusterer object based on cosine similairy
    kMeans = KMeans(n_clusters, random_state=0, n_jobs=-1)
    kMeans.fit(X)
    data["sklearn.cluster"] = kMeans.labels_
    return data


def get_k_kmeans(data, col_name, range_=range(1, 10), plot_elbow=False):

    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(max_features=500)
    X = vec.fit_transform(data[col_name])
    kMeans_inertia=[]

    for k in range_:
        kMeans=KMeans(n_clusters=k)
        kMeans.fit(X)
        kMeans_inertia.append(kMeans.inertia_)

    dict_kMeans = dict(zip([*range_], kMeans_inertia))
    if plot_elbow:
        plt.subplots()
        # for k, inertia in dict_kMeans.items():
        plt.plot(dict_kMeans.keys(), dict_kMeans.values(), 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.title('Elbow Method For Optimal k')
        plt.show()
    return dict_kMeans


def train_w2v_model(clean_text, save_model_name="ds_jobs_w2v_model"):

    from gensim.models import Word2Vec
    # train model
    model = Word2Vec(sentences=clean_text,
                     size=100,
                     window=5,
                     min_count=1,
                     workers=4,
                     sg=1)

    if save_model_name is not None:
        # save model
        model.save(f'./src/models/{save_model_name}.bin')


def w2v_clustering(data, col_name, saved_model="./src/models/ds_jobs_w2v_model.bin"):

    from gensim.models import Word2Vec
    # # using gensim to upload a pretrained model
    # # Download google news word2vec using gensim downloader
    # import gensim.downloader as api
    # wv = api.load("word2vec-google-news-300")
    # # top most similar word of a given word
    # wv.most_similar("king")
    # # similar word of King + woman - man ?
    # wv.most_similar(positive=['king', 'woman'], negative=['man'])
    # # n_similarity is used to compute cosine similarity between two sets of words.
    # wv.n_similarity(sent1_, sent2_)


    # word2vec is a neural network model
    # The values of the hidden layer will be used as the vector to represent the current word
    # Word2vec online demo: https://projector.tensorflow.org/
    # It seems like word2vec works very well for capturing the semantic similarities between words, but not for sentences.
    # gensim provides another model doc2vec for document to vectors representation,

    # If you have a large enough data from a specific domain and you want to build a word2vec model using that data.
    w2v_model = Word2Vec.load(f'{saved_model}').wv
    # define a function to embed a given text to one vector, as the average of all words' vectors in the text

    def embed(text):
        words = text.split()
        words = [word for word in words if word in w2v_model.vocab]
        return np.average(w2v_model[words], axis=0)
    # embed all texts in both training and test sets
    w2v_x = data[col_name].apply(lambda x: embed(x)).to_numpy()

    # convert them into 2D numpy array
    w2v_x = np.stack(w2v_x, axis=0)
    from sklearn.cluster import dbscan
    from sklearn.metrics.pairwise import cosine_similarity

    def cosine(A, B):
        score = cosine_similarity([A], [B])[0][0]
        return 1 - score

    clusters = dbscan(w2v_x, metric=cosine, eps=0.01, min_samples=2)[1]
    data[f"db_w2v_cluster_sim"] = clusters

    return data



if __name__ == "__main__":

    data = pd.read_csv("./data/preprocessed_data_scientist_jobstreet.csv")
    data_subset = data[['job_position', 'company', 'location', 'career_level', 'qualification', 'years_of_exp', 'job_type', 'job_spec', 'salary', 'normalize_job_description', 'processed_job_description']]
    data_clustered = semantic_clustering(data_subset, 'processed_job_description', metric='cosine')
    data_clustered = semantic_clustering(data_subset, 'processed_job_description', metric='get_sim')
    data_clustered = nltk_kmeans_clustering(data_subset, 'processed_job_description')
    data_clustered = sklearn_kmeans_clustering(data_subset, 'processed_job_description')
    dict_k_inertia = get_k_kmeans(data_subset, 'processed_job_description', plot_elbow=True)

    # EDA on dbscan_eps semantic_clustering
    for try_dbscan_eps in [0.4, 0.5, 0.6, 0.7, 0.8]:
        temp_eda_data_subset = data_subset.copy()
        temp_eda_data_subset = semantic_clustering(temp_eda_data_subset, 'processed_job_description', metric='cosine')
        print(f"========== DBSCAN with Cosine Similarity and eps of {try_dbscan_eps} ==========")
        print(f"{temp_eda_data_subset['db_cluster_cosine_sim'].value_counts()}")

    # EDA on dbscan_eps on semantic_clustering
    for try_dbscan_eps in [0.4, 0.5, 0.6, 0.7, 0.8]:
        temp_eda_data_subset = data_subset.copy()
        temp_eda_data_subset = semantic_clustering(temp_eda_data_subset, 'processed_job_description', metric='get_sim')
        print(f"========== DBSCAN with Embedded Similarity and eps of {try_dbscan_eps} ==========")
        print(f"{temp_eda_data_subset['db_cluster_get_sim_sim'].value_counts()}")


    def simplify_job_position(original_job_position):

        if ('Analyst' in original_job_position) or ('Anal' in original_job_position):
            return "Data Analyst"
        elif ('Scientist' in original_job_position) or ('Scien' in original_job_position):
            return "Data Scientist"
        elif ('Developer' in original_job_position) or ('Engineer' in original_job_position):
            return "Developer/Engineer"
        else:
            return "Others"

    data_clustered['manual_renamed_job_position'] = data_clustered['job_position'].apply(lambda x: simplify_job_position(x))

    def additional_layer_simplify_job_position(cleaned_job_description):

        if ('machine learning' in cleaned_job_description) or ('build models' in cleaned_job_description):
            return "Data Scientist"
        else:
            return np.nan

    data_clustered['manual_renamed_job_position_by_desc'] = data_clustered['normalize_job_description'].apply(lambda x: additional_layer_simplify_job_position(x))
    data_clustered.loc[data_clustered['manual_renamed_job_position_by_desc'].isna(), 'manual_renamed_job_position_by_desc'] = \
        data_clustered.loc[data_clustered['manual_renamed_job_position_by_desc'].isna(), 'manual_renamed_job_position']

    data_clustered.groupby(['manual_renamed_job_position_by_desc', 'db_cluster_cosine_sim'])['db_cluster_cosine_sim'].value_counts().unstack(0).plot.barh()
    data_clustered.groupby(['manual_renamed_job_position_by_desc', 'db_cluster_get_sim_sim'])['db_cluster_get_sim_sim'].value_counts().unstack(0).plot.barh()
    data_clustered.groupby(['manual_renamed_job_position_by_desc', 'nltk_clusters'])['nltk_clusters'].value_counts().unstack(0).plot.barh()
    data_clustered.groupby(['manual_renamed_job_position_by_desc', 'sklearn.cluster'])['sklearn.cluster'].value_counts().unstack(0).plot.barh()

    data_clustered = data_clustered.loc[:, ~data_clustered.columns.duplicated()]
    data_clustered.to_csv("./data/data_clustered_v2.csv")


    data_clustered['job_position']