
from tkinter import *
import tkinter as tk
from tkinter import scrolledtext
# from googletrans import Translator # for understanding the meaning of Chinese
import pickle
import fasttext
import jieba
import numpy as np
from googletrans import Translator
from gensim.models import Word2Vec
window = Tk()  #create a window
window.maxsize(900, 600) # define size
window.title('A Chinese classification application')
window.config(bg="skyblue")
# ChineseInputLabel = Label(text="Please Input a Chinese sentence for classification")

left_frame = Frame(window, width=200, height=400)
left_frame.grid(row=0, column=0, padx=10, pady=5)
right_frame = Frame(window, width=650, height=400)
right_frame.grid(row=0, column=1, padx=10, pady=5)
# left_frame.pack()

# Label(master=left_frame, text="""Choose a  classfier:""", justify = LEFT,padx = 20).pack()
setting_bar = Frame(left_frame, width=180, height=185)
setting_bar.grid(row=2, column=0, padx=5, pady=5)
Label(setting_bar, text="""Choose a classfier:""",relief=RAISED).grid(row=0, column=0, padx=5, pady=3, ipadx=10)
v = IntVar()
# v = 1  # set default value =1
# Radiobutton(
# setting_bar,
#             text="SVM",
#             padx = 20,
#             variable=v,
#             value=1).pack(anchor=W)
Radiobutton(setting_bar,
            text="SVM",  variable=v,
            value=1).grid(row=1, column=0, padx=5, pady=5)
Radiobutton(setting_bar,
            text="LR",  variable=v,
            value=2).grid(row=2, column=0, padx=5, pady=5)
# Radiobutton(setting_bar,
#             text="NB",  variable=v,
#             value=3).grid(row=3, column=0, padx=5, pady=5)
v.set(1)
# Radiobutton(setting_bar,
#             text="LR",
#             padx = 20,
#             variable=v,
#             value=2).pack(anchor=W)
#
# Label(setting_bar, text="""Choose an embedding:""",relief=RAISED).grid(row=4, column=0, padx=5, pady=3, ipadx=10)
# v1= IntVar()
# # v1 1 # set default value to v1
# Radiobutton(setting_bar,
#             text="Fasttext pre-trained word vectors",
#             variable=v1,
#             value=1).grid(row=5, column=0, padx=5, pady=5)
# Radiobutton(setting_bar,text="Word vectors trained on Chinanews dataset",variable=v1,
#             value=2).grid(row=6, column=0, padx=5, pady=5)
# v1.set(1)
# Label(left_frame, text="""Choose an embedding:""", justify = LEFT,padx = 20).pack()
# # Label(left_frame, text="""Choose an embedding:""").grid(row=1, column=0, padx=5, pady=5)
# v1= IntVar()
# Radiobutton(left_frame,
#             text="word2vec_average_embedding",
#             padx = 20,
#             variable=v1,
#             value=1).pack(anchor=W)
# Radiobutton(left_frame,
#             text="sentence_embedding",
#             padx = 20,
#             variable=v1,
#             value=2).pack(anchor=W)


svm_pretrained_model = pickle.load(open('svm_pretrained_segmented_model.sav', 'rb'))
lr_pretrained_model = pickle.load(open('LR_pretrained_segmented_model.sav', 'rb'))
# nb_pretrained_model = pickle.load(open('nb_pretrained_segmented_model.sav', 'rb'))
# svm_word2vec_model = pickle.load(open('svm_word2vec_segmented_model.sav', 'rb'))
# lr_word2vec_model = pickle.load(open('LR_word2vec_segmented_model.sav', 'rb'))
# nb_word2vec_model = pickle.load(open('nb_word2vec_segmented_model.sav', 'rb'))
# categories = ['mainland China politics', 'International news', 'Taiwan - Hong Kong- Macau politics', 'military news', 'society news']
# #
categories = ['mainland China politics', 'Taiwan - Hong Kong- Macau politics', 'International news','financial news','culture','entertainment','sports']

#inputbox
input_var=StringVar()
# ChineseText_entry = Entry(right_frame,textvariable = input_var, font=('calibre',10,'normal')).pack()
group1 = LabelFrame(right_frame, text="Input Chinese(news) here", padx=5, pady=5)
# group1.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky=E+W+N+S)
group1.pack()
window.columnconfigure(0, weight=1)
window.rowconfigure(1, weight=1)
group1.rowconfigure(0, weight=1)
group1.columnconfigure(0, weight=1)
# input_var.trace("w", lambda name, index, mode, sv=input_var: input_var('textbox_text', textbox_text))
txtbox = scrolledtext.ScrolledText(group1, width=40, height=10)
txtbox.grid(row=0, column=0, sticky=E+W+N+S)
txtbox.pack()
##google translate API

input_text =""
pretrained_w2vmodel = fasttext.load_model('cc.zh.300.bin')

# Chinanews_w2vmodel = Word2Vec.load("Chinanews_word2vec.model")


var_result = StringVar()
var_result.set('')

def submit():
    # input_text = input_var.get()
    input_text = txtbox.get('1.0', tk.END)
    # input_var.set("")

    # sen_vec0 = model.get_sentence_vector(input_text)  # sen_vec

    segmented = list(jieba.cut(input_text))


    temp = []
    for word in segmented:
        temp.append(pretrained_w2vmodel.get_word_vector(word))


    # sen_vec0 = np.sum(temp, axis=0) / temp.shape[0]
    sen_vec0 =  np.mean(temp,axis=0)
    sen_vec=sen_vec0.reshape(1, -1)
    print(sen_vec0)
    result=['']

    result = lr_pretrained_model.predict(sen_vec)
    if v.get()==1 :
        result = svm_pretrained_model.predict(sen_vec)
    if v.get()==2 :
        result = lr_pretrained_model.predict(sen_vec)


    # print(sen_vec)
    # result = loaded_model.predict(sen_vec)
    print(result[0])
    var_result.set("category: "+str(categories[result[0] - 1]))

# English_text = StringVar()
# English_text.set('')
English_text =''
def translate():
    # input_text = input_var.get()
    input_text =  txtbox.get('1.0', tk.END)
    # print(str(v1.get()))
    # print(str(v.get()))

    # English_text.set( translator.translate(input_text).text)
    English_text =  translator.translate(input_text).text
    txtbox1.configure(state="normal")
    txtbox1.delete('1.0', "end")
    txtbox1.insert(tk.INSERT, English_text)
    txtbox1.configure(state="disabled")


# By default, the translate() method returns the English translation of the text
translator = Translator()

# to fix the error happened for googletrans 4.0.0rc1 translator API
# solution referring to https://github.com/ssut/py-googletrans/issues/257
translator.raise_Exception = True

trans_btn=Button(right_frame,text = 'Translate to English', height = 1, width = 20, command = translate).pack()
# trans_btn = Button(right_frame, text='Translate to English', command=translate)
# trans_btn.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky=E+W+N+S)


# input_var.trace("w", lambda name, index, mode, sv=input_var: input_var('textbox_text', textbox_text))
txtbox1 = scrolledtext.ScrolledText(right_frame, width=40, height=10)
# txtbox1 = Text(right_frame, width=40, height=10)
txtbox1.pack()
txtbox1.configure(state="disabled")
# l_english= Label(right_frame, textvariable=English_text, justify=LEFT,font= ('Aerial', 12), padx=10)
#
# # l_english.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky=E+W+N+S)
# l_english.pack()
sub_btn=Button(right_frame,text = 'Predict Category', height = 1, width = 20,command = submit).pack()
# sub_btn=Button(right_frame,text = 'Submit', command = submit)
# sub_btn.grid(row=4, column=0, columnspan=3, padx=10, pady=10, sticky=E+W+N+S)
# label_cat = Label(right_frame, text="""Category:""", justify=LEFT, padx=20)
# label_cat.grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky=E+W+N+S)
# label_cat .pack()

l= Label(right_frame, textvariable=var_result, justify=LEFT,  fg='#f00',padx=20)
# l.grid(row=6, column=0, columnspan=3, padx=10, pady=10, sticky=E+W+N+S)
l.pack()






exit_button = Button(
    window,
    text='Exit',
    command=lambda: window.quit()
)



# greeting.pack()
window.mainloop()


# from googletrans import Translator
# translator = Translator(service_urls=[
#       'translate.google.com',])
# trans=translator.translate('Hello World', src='en', dest='zh-cn')


