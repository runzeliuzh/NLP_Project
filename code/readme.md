This repository contains codes for training Chinese news classification models and the User interface which allows users to input Chinese texts(news) and predict a label.  

# 1.Training classification models

## Install the required packages:  
First, create an environment and install packages using environment.yml with [anaconda](https://www.anaconda.com/).

Download environment.yml, and create a new environment.      
In the command line, type:  
*conda env create --name your_envname  -f environment.yml*    
For example,  
*conda env create --name nlptask  -f /Users/downloads/environment.yml*   
Then, *conda activate your_envname* (in the example, *conda activate nlptask*)

## Dataset
Please go to [this dataset repository](https://github.com/zhangxiangxiao/glyph) and download the **Chinanews dataset**. 

Both training dataset and testing dataset are required. Rename the training dataset file to 'Chinanews_train.csv',  
and the test dataset file to 'Chinanews_test.csv'.  

## Fasttext pre-trained word vectors
Download Fasttext's Pre-trianed Chinese vector model at [fasttext](https://fasttext.cc/docs/en/crawl-vectors.html).

In the **Models** section, find 'Chinese' select 'bin' format file.  
Download the file to a folder and unzip it.   
Now you can find a 'cc.zh.300.bin' file in the folder. 

## Training
**(1)Training LR, SVM and NB classifier With Fasttext's pre-trained word vectors**  
The model will be saved as output.
| code | saved model |
|-----:|-----------|
|LR_pretrained_segmented.py| LR_pretrained_segmented_model.sav|
|svm_pretrained_segmented.py| svm_pretrained_segmented_model.sav    |
|NB_pretrained_segmented.py| nb_pretrained_segmented_model.sav   |


**(2)Train a word vector model using the Chinanews Dataset**  
Run *Chinese2vec.py*, this step will generate a model called **'Chinanews_word2vec.model'**.  

**(3)Training LR, SVM and NB classifier uisng the 'Chinanews_word2vec.model'**  
| code | saved model |
|-----:|-----------|
|LR_word2vec_segmented.py| LR_word2vec_segmented_model.sav|
|svm_word2vec_segmented.py| svm_word2vec_segmented_model.sav    |
|NB_word2vec_segmented.py| nb_word2vec_segmented_model.sav   |


# 2.UI for predicting a Chinese(news) text to a category.
Check out the video showing how the UI works [here](https://drive.google.com/file/d/14isrZSmOdutfKgSB-BmmdA2H7qmdCbyf/view?usp=sharing)!  
Before using the '**GUI.py**' code,  
you can either run the *LR_pretrained_segmented.py*, *svm_pretrained_segmented.py* and save your classification model,      
Or you can go to [our pre-trained classification model](https://drive.google.com/drive/folders/1X8cW0JZR-7vLWWlCPaIKpXjeHgLtDyJ9?usp=sharing), download  'LR_pretrained_segmented_model.sav' and 'svm_pretrained_segmented_model.sav'.  
The pre-trained classification models are trained on 56000 pieces of Chinese news, each category 8000 piece.






