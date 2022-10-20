import pandas as pd
import numpy as np
import jieba
import scipy
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# path to store the dataset files
path = '/Users/runze/Downloads/'

#load dataset,
# go to https://github.com/zhangxiangxiao/glyph, and download Chinanews dataset
TRAIN_DATA_FILE = path + 'Chinanews_train.csv'
TEST_DATA_FILE = path + 'Chinanews_test.csv'
train_df = pd.read_csv(TRAIN_DATA_FILE)
test_df = pd.read_csv(TEST_DATA_FILE)
# print(train_df.head(10))

train_df.rename(columns={train_df.columns.values[0]: 'label',train_df.columns.values[2]:'text'}, inplace=True)
test_df.rename(columns={test_df.columns.values[0]: 'label',test_df.columns.values[2]:'text'}, inplace=True)
categories_Ifeng = ['mainland China politics', 'International news', 'Taiwan - Hong Kong- Macau politics', 'military news', 'society news']
#
categories_Chinanews = ['mainland China politics', 'Taiwan - Hong Kong- Macau politics', 'International news','financial news','culture','entertainment','sports']
###change training size for each category. The total training size would be Training_size*7
Training_size =8000
test_size =2000
Category_size=7
train_content= []
train_label = []
test_content=[]
test_label =[]
for i in range(Category_size):
    value_i=train_df.loc[(train_df["label"] == i+1)& (train_df["text"].str.len() >15)]
    train_content =train_content+list(value_i.iloc[0:Training_size,2].fillna("NAN_TEXT").values)
    # test_content = test_content + list(value_i.iloc[5000:7000, 2].fillna("NAN_TEXT").values)
    train_label = train_label + list(value_i.iloc[0:Training_size, 0].fillna("NAN").values)
    # test_label = test_label + list(value_i.iloc[5000:7000, 0].fillna("NAN").values)
for i in range(Category_size):
    value_i=test_df.loc[(test_df["label"] == i+1)& (test_df["text"].str.len() >15)]
    test_content = test_content + list(value_i.iloc[0:test_size, 2].fillna("NAN_TEXT").values)
    test_label = test_label + list(value_i.iloc[0:test_size, 0].fillna("NAN").values)

def process_news(list_sentences):
    news =[list(jieba.cut(i)) for i in list_sentences]
    return news

train_news= process_news(train_content)
test_news= process_news(test_content)

## download  https://fasttext.cc/docs/en/crawl-vectors.html
##Chinese vector, select .bin format, "cc.zh.300.bin'
# this is Facebook's pretrained Chinese word vectors
# import fasttext
# model=fasttext.load_model('cc.zh.300.bin')
#
# #assue a sentence has N tokens. The sentence vector is average of the N word vectors.
# sen0 =[]
# for sen in train_news:
#     temp =[]
#     for word in sen:
#         temp.append(model.get_word_vector(word))
#     sen0.append(temp)
# # sen0 = [model.get_word_vector[sen] for sen in train_news]
# sentence_vec_train = [  np.sum(np.array(sen),axis=0)/np.array(sen).shape[0] for sen in sen0]
# sen1 =[]
# for sen in test_news:
#     temp =[]
#     for word in sen:
#         temp.append(model.get_word_vector(word))
#     sen1.append(temp)
# # sen0= [model.get_word_vector[sen] for sen in test_news]
# sentence_vec_test = [np.sum(np.array(sen),axis=0)/np.array(sen).shape[0] for sen in sen1]
from gensim.models import Word2Vec
model = Word2Vec.load("Chinanews_word2vec.model")

#sentence embedding, using average
sen_vec_train = [model.wv[word] for word in train_news]

sentence_vec_train = [  np.sum(np.array(sen),axis=0)/np.array(sen).shape[0] for sen in sen_vec_train]


sen_vec_test = [model.wv[word] for word in test_news]
sentence_vec_test = [  np.sum(np.array(sen),axis=0)/np.array(sen).shape[0] for sen in sen_vec_test]


# classification
#Default hyperparameter means C=1.0, kernel=rbf and gamma=auto among other parameters.

# from sklearn import svm
# # X_train, X_test, y_train, y_test = train_test_split(   X, y, test_size=0.33, random_state=42)
# X_train = sentence_vec_train
# y_train = train_label
# X_test = sentence_vec_test
# y_test = test_label
# clf = svm.SVC()
# clf.fit(X_train, y_train)
# y_pred=clf.predict(X_test)
# from sklearn import metrics
# from sklearn.metrics import classification_report, confusion_matrix
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# # print(classification_report(y_test, y_pred))
# import pickle
# filename = 'sentence_model.sav'
# pickle.dump(clf, open(filename, 'wb'))


##grid search

from sklearn.metrics import accuracy_score
X_train = sentence_vec_train
y_train = train_label
X_test = sentence_vec_test
y_test = test_label
from sklearn.naive_bayes import GaussianNB

# params_NB = {
#     'var_smoothing': np.logspace(0,-9, num=10)
# }
# gridCV = GridSearchCV(GaussianNB(), param_grid=params_NB, verbose=1, n_jobs=-1)
# gridCV.fit(X_train, y_train)
#
# best_var_smoothing = gridCV.best_params_['var_smoothing']
#
#
# print("The best performing best_var_smoothing is: {:5.2f}".format(best_var_smoothing))

##best prediction model
# nbClassifier = GaussianNB(var_smoothing= best_var_smoothing)
nbClassifier = GaussianNB( )
nbClassifier.fit(X_train, y_train)
nb_predictions = nbClassifier.predict(X_test)
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
print(metrics.classification_report(y_test, nb_predictions))
print("Overall Accuracy:", round(metrics.accuracy_score(y_test, nb_predictions), 4))


import pickle
filename = 'nb_word2vec_segmented_model.sav'
pickle.dump(nbClassifier, open(filename, 'wb'))

report = metrics.classification_report(y_test, nb_predictions,output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv('classification_report_nb_word2vec_segmented.csv', index = True)

# cm = plot_confusion_matrix(rbfSVM, X_test, y_test)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, nb_predictions, labels=nbClassifier.classes_)
disp= ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nbClassifier.classes_)
disp.plot()
disp.figure_.savefig('confusion_matrix_nb_word2vec_segmented.png')




