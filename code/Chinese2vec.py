import pandas as pd
import numpy as np
import jieba
import scipy
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# path to store the dataset files
#only need to change the path
path = '/Users/.../.../'

TRAIN_DATA_FILE = path + 'Chinanews_train.csv'
TEST_DATA_FILE = path + 'Chinanews_test.csv'
train_df = pd.read_csv(TRAIN_DATA_FILE)
test_df = pd.read_csv(TEST_DATA_FILE)

from gensim.models.fasttext import FastText

#stopwords; to remove the words that frequently appear in Chinese text but don't contribute much to the meaning.
import stopwordsiso
stopwords = stopwordsiso.stopwords("zh")
# pandas pre-processing the train dataset and test dataset
train_df.rename(columns={train_df.columns.values[0]: 'label',train_df.columns.values[2]:'text'}, inplace=True)
test_df.rename(columns={test_df.columns.values[0]: 'label',test_df.columns.values[2]:'text'}, inplace=True)
# categories_Ifeng = ['mainland China politics', 'International news', 'Taiwan - Hong Kong- Macau politics', 'military news', 'society news']
#7 categories for Chinanews dataset
categories_Chinanews = ['mainland China politics', 'Taiwan - Hong Kong- Macau politics', 'International news','financial news','culture','entertainment','sports']
# determining the training size and test size
#for each category, select the first Training_size records with text length >15 as the training data.
# for each category, select the first test_size records with text length >15 as the test data 
Training_size =8000
test_size =2000
Category_size=7
train_content= []
train_label = []
test_content=[]
test_label =[]
# a=train_df.loc[train_df['1'] =="1"]
for i in range(Category_size):
    value_i=train_df.loc[(train_df["label"] == i+1)& (train_df["text"].str.len() >15)]
    #
    train_content =train_content+list(value_i.iloc[0:Training_size,2].fillna("NAN_TEXT").values)
    # test_content = test_content + list(value_i.iloc[5000:7000, 2].fillna("NAN_TEXT").values)
    train_label = train_label + list(value_i.iloc[0:Training_size, 0].fillna("NAN").values)
    # test_label = test_label + list(value_i.iloc[5000:7000, 0].fillna("NAN").values)
for i in range(Category_size):
    value_i=test_df.loc[(test_df["label"] == i+1)& (test_df["text"].str.len() >15)]
    #

    test_content = test_content + list(value_i.iloc[0:test_size, 2].fillna("NAN_TEXT").values)

    test_label = test_label + list(value_i.iloc[0:test_size, 0].fillna("NAN").values)
#function to segment sentences
def process_news(list_sentences):
    # news = []
    news =[list(jieba.cut(i)) for i in list_sentences]
    cleaned_news = [el for el in news if el not in list(stopwords)]
    # for text in list_sentences:
    #     txt = text_to_wordlist(text)
    #     news.append(txt)
    return cleaned_news

train_news= process_news(train_content)
test_news= process_news(test_content)
news = train_news+test_news
import gensim
# word2vec, size is set to 300
w2v_model = FastText(vector_size=300, window=2, min_count=1)


# w2v_model = gensim.models.Word2Vec( vector_size=300, min_count=5,window=2)
w2v_model.build_vocab(corpus_iterable = news)
w2v_model.train(
    corpus_iterable=news, epochs=10,
    total_examples=len(news))


w2v_model.save("Chinanews_word2vec.model")

##next time if you want to use the word2vec model, load it
# from gensim.models import Word2Vec
# model = Word2Vec.load("Chinanews_word2vec.model")














