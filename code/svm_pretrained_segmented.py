import pandas as pd
import numpy as np
import jieba
import scipy
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# path to store the dataset files

path = '/Users/runze/Downloads/'
# go to https://github.com/zhangxiangxiao/glyph, and download Chinanews dataset
TRAIN_DATA_FILE = path + 'Chinanews_train.csv'
TEST_DATA_FILE = path + 'Chinanews_test.csv'
train_df = pd.read_csv(TRAIN_DATA_FILE)
test_df = pd.read_csv(TEST_DATA_FILE)

train_df.rename(columns={train_df.columns.values[0]: 'label',train_df.columns.values[2]:'text'}, inplace=True)
test_df.rename(columns={test_df.columns.values[0]: 'label',test_df.columns.values[2]:'text'}, inplace=True)
categories_Ifeng = ['mainland China politics', 'International news', 'Taiwan - Hong Kong- Macau politics', 'military news', 'society news']
#
categories_Chinanews = ['mainland China politics', 'Taiwan - Hong Kong- Macau politics', 'International news','financial news','culture','entertainment','sports']

#change training size for each category. The total training size would be Training_size*7
#ideally set 8000 for Training_size, 2000 for test_size, or 12000 for Training_size, 3000 for test_size
Training_size = 8000
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



def process_news(list_sentences):
    # news = []
    news =[list(jieba.cut(i)) for i in list_sentences]
    # for text in list_sentences:
    #     txt = text_to_wordlist(text)
    #     news.append(txt)
    return news

train_news= process_news(train_content)
test_news= process_news(test_content)

## download  https://fasttext.cc/docs/en/crawl-vectors.html
##Chinese vector, select .bin format, "cc.zh.300.bin'
# this is Facebook's pretrained Chinese word vectors
import fasttext
model=fasttext.load_model('cc.zh.300.bin')

sen0 =[]
for sen in train_news:
    temp =[]
    for word in sen:
        temp.append(model.get_word_vector(word))
    sen0.append(temp)
# sen0 = [model.get_word_vector[sen] for sen in train_news]
sentence_vec_train = [  np.sum(np.array(sen),axis=0)/np.array(sen).shape[0] for sen in sen0]
sen1 =[]
for sen in test_news:
    temp =[]
    for word in sen:
        temp.append(model.get_word_vector(word))
    sen1.append(temp)

sentence_vec_test = [np.sum(np.array(sen),axis=0)/np.array(sen).shape[0] for sen in sen1]


##next time if you want to use the word2vec model, load it
# from gensim.models import Word2Vec
# model = Word2Vec.load("word2vec_Chinese.model")

#sentence embedding, using average
# sen_vec_train = [model.get_word_vector[word] for word in train_news]
#
# sentence_vec_train = [  np.sum(np.array(sen),axis=0)/np.array(sen).shape[0] for sen in sen_vec_train]


# sen_vec_test = [model.get_word_vector[word] for word in test_news]
# sentence_vec_test = [  np.sum(np.array(sen),axis=0)/np.array(sen).shape[0] for sen in sen_vec_test]

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

###grid search
# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC
# # defining parameter range
# param_grid = {'C': [0.1, 1, 10, 100, 1000],
#               'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
#               'kernel': ['rbf']}
# from sklearn.metrics import classification_report, confusion_matrix
#
# grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
#
# # fitting the model for grid search
# grid
# .fit(X_train, y_train)
#
# print(grid.best_params_)
# print(grid.best_estimator_)

# grid_predictions = grid.predict(X_test)
#
# # print classification report
# print(classification_report(y_test, grid_predictions))

# cm = confusion_matrix(y_test, y_pred_test)
#
# print('Confusion matrix\n\n', cm)
#
# print('\nTrue Positives(TP) = ', cm[0,0])
#
# print('\nTrue Negatives(TN) = ', cm[1,1])
#
# print('\nFalse Positives(FP) = ', cm[0,1])
#
# print('\nFalse Negatives(FN) = ', cm[1,0])
# visualize confusion matrix with seaborn heatmap

# cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
#                                  index=['Predict Positive:1', 'Predict Negative:0'])
#
# sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
# TP = cm[0,0]
# TN = cm[1,1]
# FP = cm[0,1]
# FN = cm[1,0]
# # print classification accuracy
# classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
#
# print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
# classification_error = (FP + FN) / float(TP + TN + FP + FN)
# # print classification error
#
# print('Classification error : {0:0.4f}'.format(classification_error))
# precision = TP / float(TP + FP)
#
# # print precision score
# print('Precision : {0:0.4f}'.format(precision))
# #print Recall
# recall = TP / float(TP + FN)
#
# print('Recall or Sensitivity : {0:0.4f}'.format(recall))

##random search
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
X_train = sentence_vec_train
y_train = train_label
X_test = sentence_vec_test
y_test = test_label
np.random.seed(123)
g_range = np.random.uniform(0.0, 0.3, 5).astype(float)
C_range = np.random.normal(1, 0.1, 5).astype(float)

# Check that gamma>0 and C>0
C_range[C_range < 0] = 0.0001

hyperparameters = {'gamma': list(g_range),
                   'C': list(C_range)}
# change n_iter here n_iter=10
randomCV = RandomizedSearchCV(SVC(kernel='rbf', ), param_distributions=hyperparameters, n_iter=10)
randomCV.fit(X_train, y_train)

best_gamma = randomCV.best_params_['gamma']
best_C = randomCV.best_params_['C']

print("The best performing gamma value is: {:5.2f}".format(best_gamma))
print("The best performing C value is: {:5.2f}".format(best_C))
##best prediction model
rbfSVM = SVC(kernel='rbf', C=best_C, gamma=best_gamma)
rbfSVM.fit(X_train, y_train)
svm_predictions = rbfSVM.predict(X_test)
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
print(metrics.classification_report(y_test, svm_predictions))
print("Overall Accuracy:", round(metrics.accuracy_score(y_test, svm_predictions), 4))
# def classification_report_csv(report):
#     report_data = []
#     lines = report.split('\n')
#     for line in lines[2:-3]:
#         row = {}
#         row_data = line.split('      ')
#         row['class'] = row_data[0]
#         row['precision'] = float(row_data[1])
#         row['recall'] = float(row_data[2])
#         row['f1_score'] = float(row_data[3])
#         row['support'] = float(row_data[4])
#         report_data.append(row)
#     dataframe = pd.DataFrame.from_dict(report_data)
#     dataframe.to_csv('classification_report_svm.csv', index = False)

import pickle
filename = 'svm_pretrained_segmented_model.sav'
pickle.dump(rbfSVM, open(filename, 'wb'))
from sklearn.metrics import plot_confusion_matrix
report = metrics.classification_report(y_test, svm_predictions,output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv('classification_report_svm_pretrained_segmented.csv', index = False)

# cm = plot_confusion_matrix(rbfSVM, X_test, y_test)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, svm_predictions, labels=rbfSVM.classes_)
disp= ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rbfSVM.classes_)
disp.plot()
# cm.figure_.savefig('confusion_matrix_svm.png')
disp.figure_.savefig('confusion_matrix_svm_pretrained_segmented.png')


