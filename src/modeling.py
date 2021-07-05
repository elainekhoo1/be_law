from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import FeatureUnion
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import os
import pandas as pd

os.chdir(r'D:\00 Self-Learnings\0 Learning and Wellbeing (LAW)\be_law')

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 50)
pd.set_option('display.min_rows', 20)


if __name__ == "__main__":

    data = pd.read_csv("./data/data_clustered.csv")
    data_subset = data.query("manual_renamed_job_position_by_desc == 'Data Scientist' or "
                             "manual_renamed_job_position_by_desc == 'Data Analyst' or "
                             "manual_renamed_job_position_by_desc == 'Developer/Engineer'")
##
    # spliting the data into train and test
    train_x, test_x, train_y, test_y = train_test_split(data_subset["processed_job_description"], data_subset['manual_renamed_job_position_by_desc'], test_size=0.20, random_state=16)

    pipe = Pipeline([('tfidf_word', TfidfVectorizer(analyzer='word', ngram_range=(1, 3))),
                     ('selector', SelectKBest(chi2, k=20000)),
                     ('classifier', LogisticRegression(penalty='none', max_iter=5000))])
    pipe.fit(train_x, train_y)

    outputs=[]
    preds = pipe.predict(test_x)
    from sklearn.metrics import f1_score
    f1_score(test_y, preds, average='macro')
    # 0.7602551679586563
##
    train_x, test_x, train_y, test_y = train_test_split(data_subset["processed_job_description"], data_subset['manual_renamed_job_position_by_desc'], test_size=0.20, random_state=16)

    pipe = Pipeline([('tfidf_word', TfidfVectorizer(analyzer='word', ngram_range=(1, 3))),
                     ('selector', SelectKBest(chi2, k=20000)),
                     ('classifier', LGBMClassifier())])
    pipe.fit(train_x, train_y)

    outputs=[]
    preds = pipe.predict(test_x)
    from sklearn.metrics import f1_score
    f1_score(test_y, preds, average='macro')
    # 0.8844868322480263
##
    train_x, test_x, train_y, test_y = train_test_split(data_subset["processed_job_description"], data_subset['manual_renamed_job_position_by_desc'], test_size=0.20, random_state=16)

    pipe = Pipeline([('tfidf_word', TfidfVectorizer(analyzer='word', ngram_range=(1, 3))),
                     ('selector', SelectKBest(chi2, k=20000)),
                     ('classifier', LinearSVC())])
    pipe.fit(train_x, train_y)

    outputs=[]
    preds = pipe.predict(test_x)
    from sklearn.metrics import f1_score
    f1_score(test_y, preds, average='macro')
    # 0.7262032085561497
##
# We will save this pipeline object using the dump function in the joblib library
import joblib
from joblib import dump

# dump the pipeline model
dump(pipe, filename="./src/models/job_description_classification_lgbm.joblib")