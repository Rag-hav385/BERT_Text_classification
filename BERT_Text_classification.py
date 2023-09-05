#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


#all imports
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Model


# In[ ]:


tf.test.gpu_device_name()


# <font size=4>Grader function 1 </font>

# In[ ]:


def grader_tf_version():
    assert((tf.__version__)>'2')
    return True
grader_tf_version()


# <pre><font size=6>Part-1: Preprocessing</font></pre>

# In[ ]:


get_ipython().system('pwd')


# In[ ]:


#Read the dataset - Amazon fine food reviews
reviews = pd.read_csv(r"drive/MyDrive/NLP-Transfer-Learning/Reviews.csv")
#check the info of the dataset
reviews.info()


# In[ ]:


#get only 2 columns - Text, Score
#drop the NAN values
print("Getting Texts and scores..")
reviews = reviews[["Text" , "Score"]]
print("Dropping Nan values..")
reviews = reviews.dropna()
print("Done.")
print(reviews.info())


# In[ ]:


#if score> 3, set score = 1
#if score<=2, set score = 0
#if score == 3, remove the rows. 
from tqdm import tqdm

conditions = [
              (reviews.Score > 3),
              (reviews.Score <= 2),
              (reviews.Score == 3)
]

values = [1 , 0 , 2]
reviews["Score"] = np.select(conditions , values)
reviews = reviews[reviews.Score != 2]
reviews.info()


# <font size=4>Grader function 2 </font>

# In[ ]:


def grader_reviews():
    temp_shape = (reviews.shape == (525814, 2)) and (reviews.Score.value_counts()[1]==443777)
    assert(temp_shape == True)
    return True
grader_reviews()


# In[ ]:


def get_wordlen(x):
    return len(x.split())
reviews['len'] = reviews.Text.apply(get_wordlen)
reviews = reviews[reviews.len<50]
reviews = reviews.sample(n=100000, random_state=30)


# In[ ]:


reviews.Text[357766]


# In[ ]:


#remove HTML from the Text column and save in the Text column only
#Reference : https://stackoverflow.com/questions/9662346/python-code-to-remove-html-tags-from-a-string
import re
def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

print("Before ..")
print(reviews.Text[357766])

reviews.Text = reviews.Text.apply(cleanhtml)

print("After ..")
print(reviews.Text[357766])
reviews.head()


# In[ ]:


#print head 5
reviews.head(5)


# In[ ]:


#split the data into train and test data(20%) with Stratify sampling, random state 33,
from sklearn.model_selection import train_test_split
train , test = train_test_split(reviews , test_size = 0.2 , random_state = 33)
print("Reviews Shape",reviews.shape)
print("Train Shape" , train.shape)
print("Test shape" , test.shape)


# In[ ]:


X_train = train.Text
y_train = train.Score

X_test = test.Text
y_test = test.Score

print("Training")
print(X_train.shape , y_train.shape)
print("Testing")
print(X_test.shape , y_test.shape)


# In[ ]:


#plot bar graphs of y_train and y_test
import matplotlib.pyplot as plt
import seaborn as sns

train_counts = y_train.value_counts()
test_counts = y_test.value_counts()

sns.barplot(train_counts.index , train_counts.values)

plt.xlabel("Labels")
plt.ylabel("Counts")
plt.title("Train")
plt.show()

sns.barplot(test_counts.index , test_counts.values)
plt.xlabel("Labels")
plt.ylabel("Counts")
plt.title("Test")
plt.show()


# In[ ]:


#saving to disk. if we need, we can load preprocessed data directly. 
reviews.to_csv('preprocessed.csv', index=False)


# In[ ]:


## Loading the Pretrained Model from tensorflow HUB
tf.keras.backend.clear_session()

# maximum length of a seq in the data we have, for now i am making it as 55. You can change this
max_seq_length = 55

#BERT takes 3 inputs

#this is input words. Sequence of words represented as integers
input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")

#mask vector if you are padding anything
input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")

#segment vectors. If you are giving only one sentence for the classification, total seg vector is 0. 
#If you are giving two sentenced with [sep] token separated, first seq segment vectors are zeros and 
#second seq segment vector are 1's
segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="segment_ids")

#bert layer 
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=False)
pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

#Bert model
#We are using only pooled output not sequence out. 
#If you want to know about those, please read https://www.kaggle.com/questions-and-answers/86510
bert_model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=pooled_output)


# In[ ]:


bert_model.summary()


# In[ ]:


bert_model.output


# <pre><font size=6>Part-3: Tokenization</font></pre>

# In[ ]:


#getting Vocab file
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()


# In[ ]:


pip install sentencepiece


# In[ ]:


import tokenization #- We have given tokenization.py file


# In[ ]:


# Create tokenizer " Instantiate FullTokenizer" 
# name must be "tokenizer"
# the FullTokenizer takes two parameters 1. vocab_file and 2. do_lower_case 
# we have created these in the above cell ex: FullTokenizer(vocab_file, do_lower_case )
# please check the "tokenization.py" file the complete implementation

tokenizer = tokenization.FullTokenizer(vocab_file = vocab_file , do_lower_case=do_lower_case)


# <font size=4>Grader function 3 </font>

# In[ ]:


#it has to give no error 
def grader_tokenize(tokenizer):
    out = False
    try:
        out=('[CLS]' in tokenizer.vocab) and ('[SEP]' in tokenizer.vocab)
    except:
        out = False
    assert(out==True)
    return out
grader_tokenize(tokenizer)


# In[ ]:


[1] * 4 + [0]*4


# In[ ]:


# Create train and test tokens (X_train_tokens, X_test_tokens) from (X_train, X_test) using Tokenizer and 

# add '[CLS]' at start of the Tokens and '[SEP]' at the end of the tokens. 

# maximum number of tokens is 55(We already given this to BERT layer above) so shape is (None, 55)

# if it is less than 55, add '[PAD]' token else truncate the tokens length.(similar to padding)

# Based on padding, create the mask for Train and Test ( 1 for real token, 0 for '[PAD]'), 
# it will also same shape as input tokens (None, 55) save those in X_train_mask, X_test_mask

# Create a segment input for train and test. We are using only one sentence so all zeros. This shape will also (None, 55)

# type of all the above arrays should be numpy arrays

# after execution of this cell, you have to get 
# X_train_tokens, X_train_mask, X_train_segment
# X_test_tokens, X_test_mask, X_test_segment
def split(row):
    return row.split()

def return_token(row):
    
    tokens = tokenizer.tokenize(row)

    if len(tokens) < (max_seq_length - 2):
        mul = (max_seq_length - 2) - len(tokens)
        tokens = tokens + ["[PAD]"]*mul
        if(len(tokens) != max_seq_length - 2):
            print("Length of tokens is not max_seq_length - 2 after padding it is {}".format(len(tokens)))
    
    if len(tokens) > (max_seq_length - 2):
        tokens = tokens[0:(max_seq_length - 2)]

        if(len(tokens) != max_seq_length - 2):
            print("Length of tokens is not max_seq_length - 2 after padding it is {}".format(len(tokens)))
    
    tokens = ["[CLS]",*tokens,"[SEP]"]

    return tokenizer.convert_tokens_to_ids(tokens)

def return_mask(token):
    no_1 = len(token) - token.count(0)
    no_0 = token.count(0)
    return [1]*no_1 + [0]*no_0

def return_segment(token):
    return [0]*max_seq_length


X_train_tokens = pd.DataFrame(X_train.values , columns = ["val"])["val"].apply(return_token)
X_train_mask = pd.DataFrame(X_train_tokens , columns = ["val"])["val"].apply(return_mask)
X_train_segment = pd.DataFrame(X_train_tokens , columns = ["val"])["val"].apply(return_segment)


X_test_tokens = pd.DataFrame(X_test.values , columns = ["val"])["val"].apply(return_token)
X_test_mask = pd.DataFrame(X_test_tokens , columns = ["val"])["val"].apply(return_mask)
X_test_segment = pd.DataFrame(X_test_tokens , columns = ["val"])["val"].apply(return_segment)


# In[ ]:


import pickle


# In[ ]:


##save all your results to disk so that, no need to run all again. 
pickle.dump((X_train, X_train_tokens, X_train_mask, X_train_segment, y_train),open('train_data.pkl','wb'))
pickle.dump((X_test, X_test_tokens, X_test_mask, X_test_segment, y_test),open('test_data.pkl','wb'))


# In[ ]:


#you can load from disk
X_train, X_train_tokens, X_train_mask, X_train_segment, y_train = pickle.load(open("train_data.pkl", 'rb')) 
X_test, X_test_tokens, X_test_mask, X_test_segment, y_test = pickle.load(open("test_data.pkl", 'rb')) 


# In[ ]:


#Converting from serires to numpy
X_train_tokens = X_train_tokens.to_numpy()
X_train_mask = X_train_mask.to_numpy()
X_train_segment = X_train_segment.to_numpy()

X_test_tokens = X_test_tokens.to_numpy()
X_test_mask = X_test_mask.to_numpy()
X_test_segment = X_test_segment.to_numpy()


# In[ ]:


#Expanding the shape into desired ones
def expand_object(array):
    df = pd.DataFrame()
    for i in tqdm(range(max_seq_length)):
        df[i] = [k[i] for k in array]
    
    return df.to_numpy()

X_train_tokens = expand_object(X_train_tokens)
X_train_mask = expand_object(X_train_mask)
X_train_segment = expand_object(X_train_segment)

X_test_tokens = expand_object(X_test_tokens)
X_test_mask = expand_object(X_test_mask)
X_test_segment = expand_object(X_test_segment)


# <font size=4>Grader function 4 </font>

# In[ ]:


def grader_alltokens_train():
    out = False
    
    if type(X_train_tokens) == np.ndarray:
        #print("I m here")
        temp_shapes = (X_train_tokens.shape[1]==max_seq_length) and (X_train_mask.shape[1]==max_seq_length) and \
        (X_train_segment.shape[1]==max_seq_length)
        #print(temp_shapes)
        segment_temp = not np.any(X_train_segment)
        #print(segment_temp)
        mask_temp = np.sum(X_train_mask==0) == np.sum(X_train_tokens==0)
        #print(mask_temp)
        no_cls = np.sum(X_train_tokens==tokenizer.vocab['[CLS]'])==X_train_tokens.shape[0]
        #print(no_cls)
        no_sep = np.sum(X_train_tokens==tokenizer.vocab['[SEP]'])==X_train_tokens.shape[0]
        #print(no_sep)
        out = temp_shapes and segment_temp and mask_temp and no_cls and no_sep
      
    else:
        print('Type of all above token arrays should be numpy array not list')
        out = False
    assert(out==True)
    return out

grader_alltokens_train()


# <font size=4>Grader function 5 </font>

# In[ ]:


def grader_alltokens_test():
    out = False
    if type(X_test_tokens) == np.ndarray:
        
        temp_shapes = (X_test_tokens.shape[1]==max_seq_length) and (X_test_mask.shape[1]==max_seq_length) and \
        (X_test_segment.shape[1]==max_seq_length)
        
        segment_temp = not np.any(X_test_segment)
        
        mask_temp = np.sum(X_test_mask==0) == np.sum(X_test_tokens==0)
        
        no_cls = np.sum(X_test_tokens==tokenizer.vocab['[CLS]'])==X_test_tokens.shape[0]
        
        no_sep = np.sum(X_test_tokens==tokenizer.vocab['[SEP]'])==X_test_tokens.shape[0]
        
        out = temp_shapes and segment_temp and mask_temp and no_cls and no_sep
      
    else:
        print('Type of all above token arrays should be numpy array not list')
        out = False
    assert(out==True)
    return out
grader_alltokens_test()


# In[ ]:


bert_model.input


# In[ ]:


bert_model.output


# In[ ]:


# get the train output, BERT model will give one output so save in
# X_train_pooled_output
X_train_pooled_output=bert_model.predict([X_train_tokens,X_train_mask,X_train_segment])


# In[ ]:


# get the test output, BERT model will give one output so save in
# X_test_pooled_output
X_test_pooled_output=bert_model.predict([X_test_tokens,X_test_mask,X_test_segment])


# In[ ]:


##save all your results to disk so that, no need to run all again. 
pickle.dump((X_train_pooled_output, X_test_pooled_output),open('final_output.pkl','wb'))


# In[ ]:


import pickle
X_train_pooled_output, X_test_pooled_output= pickle.load(open('drive/MyDrive/NLP-Transfer-Learning/final_output.pkl', 'rb'))


# <font size=4>Grader function 6 </font>

# In[ ]:


#now we have X_train_pooled_output, y_train
#X_test_pooled_ouput, y_test

#please use this grader to evaluate
def greader_output():
    assert(X_train_pooled_output.shape[1]==768)
    assert(len(y_train)==len(X_train_pooled_output))
    assert(X_test_pooled_output.shape[1]==768)
    assert(len(y_test)==len(X_test_pooled_output))
    assert(len(y_train.shape)==1)
    assert(len(X_train_pooled_output.shape)==2)
    assert(len(y_test.shape)==1)
    assert(len(X_test_pooled_output.shape)==2)
    return True
greader_output()


# In[ ]:


##imports
from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.models import Model


# In[ ]:


##create an NN and 

input_ = Input(shape = (768,))

x= Dense(768,activation="relu", kernel_initializer="he_normal")(input_)



x= Dense(600,activation="relu", kernel_initializer="he_normal")(x)


x= Dense(300,activation="relu", kernel_initializer="he_normal")(x)

#x= Dropout(0.2)(x)

x= Dense(200,activation="relu", kernel_initializer="he_normal")(x)

#x= Dropout(0.2)(x)

x= Dense(100,activation='relu', kernel_initializer="he_normal")(x)

x= Dense(25,activation='relu', kernel_initializer="he_normal")(x)

x= Dense(10,activation='relu', kernel_initializer="he_normal")(x)

preds = Dense(2, activation='softmax')(x)

model = tf.keras.Model(input_, preds)
model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(lr=0.001),metrics=tf.keras.metrics.AUC())

model.summary()


# In[ ]:


import datetime
train = X_train_pooled_output
test = X_test_pooled_output


# In[ ]:


y_train = y_train.reset_index()
y_train = y_train.Score
y_train


# In[ ]:


y_test = y_test.reset_index()
y_test = y_test.Score
y_test


# In[ ]:


y_train = tf.keras.utils.to_categorical(y_train , num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test , num_classes=2)


# In[ ]:


AUC_Score = return_AUC_score(test , y_test)
log_dir="logs1\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1, write_graph=True)

model.fit(train, y_train, validation_data=(test, y_test),
          epochs=20, batch_size=128 ,callbacks=[tensorboard_callback])


# In[ ]:


tf.keras.utils.plot_model(model, show_shapes=True)


# In[ ]:


get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[ ]:


get_ipython().run_line_magic('tensorboard', '--logdir  .')


# Creating a Data pipeline for BERT Model</font> 
# 
# 1. Download data from <a href="https://drive.google.com/file/d/1QwjqTsqTX2vdy7fTmeXjxP3dq8IAVLpo/view?usp=sharing">here</a>
# 

# In[ ]:


test = pd.read_csv("drive/MyDrive/NLP-Transfer-Learning/test.csv")
test.info()


# In[ ]:


#Removing HTML Tags
test["Text"] = test.Text.apply(cleanhtml)



# In[ ]:


def return_token(row):
    
    tokens = tokenizer.tokenize(row)

    if len(tokens) < (max_seq_length - 2):
        mul = (max_seq_length - 2) - len(tokens)
        tokens = tokens + ["[PAD]"]*mul
        if(len(tokens) != max_seq_length - 2):
            print("Length of tokens is not max_seq_length - 2 after padding it is {}".format(len(tokens)))
    
    if len(tokens) > (max_seq_length - 2):
        tokens = tokens[0:(max_seq_length - 2)]

        if(len(tokens) != max_seq_length - 2):
            print("Length of tokens is not max_seq_length - 2 after padding it is {}".format(len(tokens)))
    
    tokens = ["[CLS]",*tokens,"[SEP]"]

    return tokenizer.convert_tokens_to_ids(tokens)

def return_mask(token):
    no_1 = len(token) - token.count(0)
    no_0 = token.count(0)
    return [1]*no_1 + [0]*no_0

def return_segment(token):
    return [0]*max_seq_length


# In[ ]:


test_tokens = test.Text.apply(return_token)
test_mask = pd.DataFrame(test_tokens)["Text"].apply(return_mask)
test_segment = pd.DataFrame(test_tokens)["Text"].apply(return_segment)


# In[ ]:


X_test_tokens = test_tokens.to_numpy()
X_test_mask = test_mask.to_numpy()
X_test_segment = test_segment.to_numpy()


# In[ ]:


def expand_object(array):
    df = pd.DataFrame()
    for i in tqdm(range(max_seq_length)):
        df[i] = [k[i] for k in array]
    
    return df.to_numpy()


X_test_tokens = expand_object(X_test_tokens)
X_test_mask = expand_object(X_test_mask)
X_test_segment = expand_object(X_test_segment)


# In[ ]:


print(X_test_tokens.shape)
print(X_test_mask.shape)
print(X_test_segment.shape)


# In[ ]:


test=bert_model.predict([X_test_tokens,X_test_mask,X_test_segment])


# In[ ]:


print(test.shape)


# In[ ]:


def return_classlabel(arr):
    if arr[0] > arr[1]:
        return 0
    else:
        return 1


# In[ ]:


pred = model.predict(test)
pred_array = []
for p in pred:
    pred_array.append(return_classlabel(p))

print("Predictions for Test..")
print(np.asarray(pred_array))


# In[2]:


#I m not training the whole thing,as I ll get the same output.

predictions = """0 1 1 1 1 1 0 1 1 1 1 1 0 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1
 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 0 1 0 0 0 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 0 1 1 1 1 1 0 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 1 1 1 1 1 0 1 1 1 0 1 1 1 1 1 1
 1 1 1 1 1 1 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1
 1 0 0 1 0 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
 1 1 1 0 0 1 1 1 1 1 0 1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 0 1 1 1 1 0 1 0 0
 1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"""


# In[ ]:


predictions = list(map(int , predictions.split()))
predictions


# In[5]:


print("1 : ",predictions.count(1))
print("0 : ",predictions.count(0))


# <Pre><font size=6>Observation</font> 
# 
# Part-1 (Training the model):
#     1. First read the reviews dataset and made the neccesary changes that were :
#         i.  Extracting Columns - Text , Score
#         ii. Converting the probelem into 2 class classification by binning the score column ( > 3 == 1 ,<=2 == 0 , ==3 == 0 )
#         iii. Removing the HTML Components from Text Column.
#     2. Now we make a training dataset and test dataset(test = 20%)
#     3. Now we tokenize the data set using tokenization.py and we get tokens , mask and segments.
#     4. Now we create a bert_model which is a 12 layers encoder model
#         i. input (tokens , mask and segments) 
#         ii.(None , 768) shape bert_embeddings of the tokens.
# 
#     5. Using these Bert Embeddings,we create a Forward NN for 2 class-classification with 'auc' as a metric.
#     6. We Train the model using train_bert_embeddings and validate using test_bert_embeddings with 
#     categorizing the y_train and y_test.
#     7. We get the auc = 0.97
# 
# Part - 2(Creating a Pipeline and Predict):
#     1. We Read the test data.
#     2. Remove all the HTML components.
#     3. Tokenization.py to get the tokens and then get the mask and segment.
#     4. Using test_tokens , test_mask , test_segment , get the bert_embeddings using bert_model.
#     5. Use the NN model and get the predictions for the bert_embeddings.
#     6. Prediction:
#         1 :  305
#         0 :  47
# 
# 
# </pre>
