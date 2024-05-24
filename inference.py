
import pandas as pd
import numpy as np
# import tensorflow as tf
import torch
from torch.nn import BCEWithLogitsLoss, BCELoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score
import pickle
from transformers import *
from tqdm import tqdm, trange
from ast import literal_eval


import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
# print(torch.cuda.get_device_name(0))
catagories = ["anger",	"anticipation",	"disgust",	"fear",	"joy",	"love",	"optimism"	,"pessimism"	,"sadness"	,"surprise"	,"trust"]
label_cols = catagories
batch_size = 512
learning_rate = 0.00005
num_of_former = 0
max_length = 30
epochs = 30

# script_path = '.'

# script_dir = os.path.dirname(script_path)

# os.chdir("./")
# current_directory = os.getcwd()

# print("working directory:", current_directory)
text = input("input sentence:")
# put text into "test" dataframe



train_data = pd.read_csv('train.csv', sep='\t')
test_data = pd.DataFrame({"Tweet":text})
print(train_data.shape)
val_data = pd.read_csv("valid.csv", sep='\t')
train_data = pd.concat([train_data,val_data])
train_data = pd.concat([train_data,test_data])

# from sklearn.model_selection import train_test_split

# train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=42)

# train_data, val_data = train_test_split(train_data, test_size=1/8, random_state=42)

# train_data.head()
test_data.head()

# print(train_data.shape)
# print(test_data.head())
# print(train_data["classes"])
# d = []
# print(len(train_data["classes"]))
# for i in train_data["classes"]:
#     # if i not in d:
#     test = i.split(",")
#     # print(i)
#     # if len(test)>1:
#     for e in test:
#         if e not in d:
#             d.append(i)

# print(d)
# catagories = d


# classes = d
# for c in classes:
#   train_data[c] = 0


# print(train_data['labels'][0])
# for index in range(train_data.shape[0]):
#     data = train_data.iloc[index]
#     # print(data['labels'])
#     for c in classes:
#       if c in data['classes']:

#         train_data[c][index] = 1

train_data.head()

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import re

def removeBlank(sentence):
    cleaned_sentence = re.sub(r'\s+', ' ', sentence).strip()
    return cleaned_sentence
# Remove extra spaces

def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext
def cleanEmoji(sentence):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    return  emoji_pattern.sub(r'', sentence) # no emoji

def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned


def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent
def remove_mentions(sentence):
    # 定義匹配 @username 的正則表達式
    pattern = r'@[\w.]+'
    # 使用 sub 方法將匹配到的部分替換為空字串
    cleaned_sentence = re.sub(pattern, '', sentence)
    return cleaned_sentence
# data = train_data
# data['Tweet'] = data['Tweet'].str.lower()
# data['Tweet'] = data['Tweet'].apply(cleanHtml)
# data['Tweet'] = data['Tweet'].apply(cleanPunc)
# data['Tweet'] = data['Tweet'].apply(keepAlpha)
# data['Tweet'] = data['Tweet'].apply(remove_mentions)

test_data['Tweet'] = test_data['Tweet'].str.lower()
test_data['Tweet'] = test_data['Tweet'].apply(cleanHtml)
test_data['Tweet'] = test_data['Tweet'].apply(cleanPunc)
test_data['Tweet'] = test_data['Tweet'].apply(keepAlpha)
test_data['Tweet'] = test_data['Tweet'].apply(cleanEmoji)
# data['Tweet'] = data['Tweet'].apply(cleanEmoji)
test_data['Tweet'] = test_data['Tweet'].apply(remove_mentions)
test_data.head()
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('wordnet')
stemmer = SnowballStemmer("english")
def stemming(text):
    lemmatizer = WordNetLemmatizer()

    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    lemmatized_text = ' '.join(lemmatized_words)

    return lemmatized_text


# data['Tweet'] = data['Tweet'].apply(stemming)
test_data['Tweet'] = test_data['Tweet'].apply(stemming)
# data.head()

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)

#data['Tweet'] = data['Tweet'].apply(removeStopWords)
#test_data['Tweet'] = test_data['Tweet'].apply(removeStopWords)

# data.head()

test_data['Tweet'] = test_data['Tweet'].apply(removeBlank)
# data['Tweet'] = data['Tweet'].apply(removeBlank)

# df = train_data #jigsaw-toxic-comment-classification-challenge
# df.head()

test_data.head()
df = test_data

# print('average sentence length: ', df.Tweet.str.split().str.len().mean())
# print('stdev sentence length: ', df.Tweet.str.split().str.len().std())

# max_length = int(df.Tggweet.str.split().str.len().mean() + 2 * df.Tweet.str.split().str.len().std())

# cols = train_data.columns
# label_cols = list(cols[2:])
# num_labels = len(label_cols)
# print('Label columns: ', label_cols)

# print('Count of 1 per label: \n', df[label_cols].sum(), '\n') # Label counts, may need to downsample or upsample
# print('Count of 0 per label: \n', df[label_cols].eq(0).sum())
# s = sum(df[label_cols].sum())

# weighted_loss= s/(df[label_cols].sum()*num_labels)
# print(type(weighted_loss))
# weighted_loss = weighted_loss.tolist()
#weighted_loss = [w*(3/2) if w >1 else w*(2/3)for w in weighted_loss]
# print(weighted_loss)

df = df.sample(frac=1).reset_index(drop=True) #shuffle rows

df['one_hot_labels'] = list(df[label_cols].values)
df.head()

labels = list(df.one_hot_labels.values)
comments = list(df.Tweet.values)



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True) # tokenizer
encodings = tokenizer.batch_encode_plus(comments,max_length=max_length,truncation = True,pad_to_max_length=True) # tokenizer's encoding method
print('tokenizer outputs: ', encodings.keys())

input_ids = encodings['input_ids'] # tokenized and encoded sentences
token_type_ids = encodings['token_type_ids'] # token type ids
attention_masks = encodings['attention_mask'] # attention masks
print(tokenizer.decode(input_ids[0]))

# Identifying indices of 'one_hot_labels' entries that only occur once - this will allow us to stratify split our training data later
label_counts = df.one_hot_labels.astype(str).value_counts()
one_freq = label_counts[label_counts==1].keys()
one_freq_idxs = sorted(list(df[df.one_hot_labels.astype(str).isin(one_freq)].index), reverse=True)
print('df label indices with only one instance: ', one_freq_idxs)

# Gathering single instance inputs to force into the training set after stratified split
one_freq_input_ids = [input_ids.pop(i) for i in one_freq_idxs]
one_freq_token_types = [token_type_ids.pop(i) for i in one_freq_idxs]
one_freq_attention_masks = [attention_masks.pop(i) for i in one_freq_idxs]
one_freq_labels = [labels.pop(i) for i in one_freq_idxs]

"""Be sure to handle all classes during validation using "stratify" during train/validation split:"""

# Use train_test_split to split our data into train and validation sets

# train_inputs, validation_inputs, train_labels, validation_labels, train_token_types, validation_token_types, train_masks, validation_masks = train_test_split(input_ids, labels, token_type_ids,attention_masks,
#                                                             random_state=None, test_size=0.0001)

# train_inputs, test_input_ids, train_labels, test_labels, train_token_types, test_token_type_ids, train_masks, test_attention_masks = train_test_split(train_inputs, train_labels, train_token_types,train_masks,
#                                                             random_state=None, test_size=0.0001)

# print(len(train_inputs))
# print(len(test_input_ids))


model  = torch.load('./model.pt')
model.eval()


test_inputs = torch.tensor(input_ids)
test_labels = torch.tensor(labels)
test_masks = torch.tensor(attention_masks)
test_token_types = torch.tensor(token_type_ids)
# Create test dataloader
test_ = TensorDataset(test_inputs, test_masks, test_labels, test_token_types)
test_sampler = SequentialSampler(test_)
test_dataloader = DataLoader(test_, sampler=test_sampler, batch_size=batch_size)


#track variables
logit_preds,true_labels,pred_labels,tokenized_texts = [],[],[],[]

# Predict
for i, batch in enumerate(test_dataloader):
  batch = tuple(t.to(device) for t in batch)
  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask, b_labels, b_token_types = batch
  with torch.no_grad():
    # Forward pass
    outs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    b_logit_pred = outs[0]
    pred_label = torch.sigmoid(b_logit_pred)

    b_logit_pred = b_logit_pred.detach().cpu().numpy()
    pred_label = pred_label.to('cpu').numpy()
    b_labels = b_labels.to('cpu').numpy()

  tokenized_texts.append(b_input_ids)
  logit_preds.append(b_logit_pred)
  true_labels.append(b_labels)
  pred_labels.append(pred_label)

# Flatten outputs
tokenized_texts = [item for sublist in tokenized_texts for item in sublist]
pred_labels = [item for sublist in pred_labels for item in sublist]
true_labels = [item for sublist in true_labels for item in sublist]
# Converting flattened binary values to boolean values
true_bools = [tl==1 for tl in true_labels]

pred_bools = [pl>0.50 for pl in pred_labels] #boolean output after thresholding

# Print and save classification report
print('Test F1 Accuracy: ', f1_score(true_bools, pred_bools,average='macro'))
print('Test Flat Accuracy: ', accuracy_score(true_bools, pred_bools),'\n')
clf_report = classification_report(true_bools,pred_bools,target_names=catagories)
pickle.dump(clf_report, open('classification_report.txt','wb')) #save report
print(clf_report)

"""## Load and Preprocess Test Data"""

tb = true_labels.copy()
pb = pred_labels.copy()
#print(tb)
#print(pb)

test_df = test_data
# test_labels_df = pd.read_csv('test_labels.csv')
# test_df = test_df.merge(test_labels_df, on='id', how='left
# test_label_cols = list(test_df.columns[2:])
# print('Null values: ', test_df.isnull().values.any()) #should not be any null sentences or labels
# print('Same columns between train and test: ', label_cols == test_label_cols) #columns should be the same
test_df.head()

# test_df = test_df[~test_df[test_label_cols].eq(-1).any(axis=1)] #remove irrelevant rows/comments with -1 values
# test_df['one_hot_labels'] = list(test_df[test_label_cols].values)
# test_df.head()

# Gathering input data
# test_labels = list(test_df.one_hot_labels.values)
test_comments = list(test_df.Tweet.values)

# Encoding input data
input = tokenizer(test_comments,truncation  = True,max_length=max_length,pad_to_max_length=True)
test_encodings = tokenizer.batch_encode_plus(test_comments,truncation  = True,max_length=max_length,pad_to_max_length=True)
test_input_ids = test_encodings['input_ids']
test_token_type_ids = test_encodings['token_type_ids']
test_attention_masks = test_encodings['attention_mask']

# Make tensors out of data
test_inputs = torch.tensor(test_input_ids)
# test_labels = torch.tensor(test_labels)
test_masks = torch.tensor(test_attention_masks)
test_token_types = torch.tensor(test_token_type_ids)
# Create test dataloader
test_data = TensorDataset(test_inputs, test_masks, test_token_types)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
# Save test dataloader
torch.save(test_dataloader,'test_data_loader')

"""## Prediction and Metics"""

# Test

# Put model in evaluation mode to evaluate loss on the validation set
model.eval()

#track variables
logit_preds,test,pred_labels,tokenized_texts = [],[],[],[]

# Predict
for i, batch in enumerate(test_dataloader):
  batch = tuple(t.to(device) for t in batch)
  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask, b_token_types = batch
  with torch.no_grad():
    # Forward pass
    outs = model(b_input_ids, token_type_ids=b_token_types, attention_mask=b_input_mask)
    b_logit_pred = outs[0]
    pred_label = torch.sigmoid(b_logit_pred)

    b_logit_pred = b_logit_pred.detach().cpu().numpy()
    pred_label = pred_label.to('cpu').numpy()
    # b_labels = b_labels.to('cpu').numpy()

  tokenized_texts.append(b_input_ids)
  logit_preds.append(b_logit_pred)
  # test.append(b_labels)
  pred_labels.append(pred_label)

# Flatten outputs
tokenized_texts = [item for sublist in tokenized_texts for item in sublist]
pred_labels = [item for sublist in pred_labels for item in sublist]
test = [item for sublist in test for item in sublist]
# Converting flattened binary values to boolean values
true_bools = [tl==1 for tl in test]

"""We need to threshold our sigmoid function outputs which range from [0, 1]. Below I use 0.50 as a threshold."""

res = pred_labels

# result = pd.DataFrame(tt,columns=catagories)
# result["id"] = test_df["id"]
# new_order = ["id","AM",       "MS",   "OTHER" ,"PH",  "SF"    ,"SR"]

# result = result[new_order].astype(int)
# print(result)
# result.to_csv("sub.csv", index=False)

"""# Mine

## Output Dataframe
"""

idx2label = dict(zip(range(6),label_cols))
print(idx2label)

# Getting indices of where boolean one hot vector true_bools is True so we can use idx2label to gather label names
true_label_idxs, pred_label_idxs=[],[]
for vals in true_bools:
  true_label_idxs.append(np.where(vals)[0].flatten().tolist())
for vals in pred_bools:
  pred_label_idxs.append(np.where(vals)[0].flatten().tolist())

# Gathering vectors of label names using idx2label
true_label_texts, pred_label_texts = [], []
for vals in true_label_idxs:
  if vals:
    true_label_texts.append([idx2label[val] for val in vals])
  else:
    true_label_texts.append(vals)

for vals in pred_label_idxs:
  if vals:
    pred_label_texts.append([idx2label[val] for val in vals])
  else:
    pred_label_texts.append(vals)

# Decoding input ids to comment text
comment_texts = [tokenizer.decode(text,skip_special_tokens=True,clean_up_tokenization_spaces=False) for text in tokenized_texts]

# Converting lists to df
# comparisons_df = pd.DataFrame({'comment_text': comment_texts, 'true_labels': true_label_texts, 'pred_labels':pred_label_texts})
# comparisons_df.to_csv('comparisons.csv')
# comparisons_df.head()

"""## Bonus - Optimizing threshold value for micro F1 score

Doing this may result in a trade offs between precision, flat accuracy and micro F1 accuracy. You may tune the threshold however you want.
"""

print(len(tb))
print(len(pb))

# Calculate Accuracy - maximize F1 accuracy by tuning threshold values. First with 'macro_thresholds' on the order of e^-1 then with 'micro_thresholds' on the order of e^-2

macro_thresholds = np.array(range(1,10))/10
true_label = tb
pred_labels=pb
print(len(true_label))
print(len(pred_labels))
f1_results, flat_acc_results = [], []
for th in macro_thresholds:
  pred_bools = [pl>th for pl in pred_labels]
  test_f1_accuracy = f1_score(true_label,pred_bools,average='macro')
  test_flat_accuracy = accuracy_score(true_label, pred_bools)
  f1_results.append(test_f1_accuracy)
  flat_acc_results.append(test_flat_accuracy)

best_macro_th = macro_thresholds[np.argmax(f1_results)] #best macro threshold value

micro_thresholds = (np.array(range(10))/100)+best_macro_th #calculating micro threshold values

f1_results, flat_acc_results = [], []
for th in micro_thresholds:
  pred_bools = [pl>th for pl in pred_labels]
  test_f1_accuracy = f1_score(true_label,pred_bools,average='macro')
  test_flat_accuracy = accuracy_score(true_label, pred_bools)
  f1_results.append(test_f1_accuracy)
  flat_acc_results.append(test_flat_accuracy)

best_f1_idx = np.argmax(f1_results) #best threshold value

# Printing and saving classification report
print('Best Threshold: ', micro_thresholds[best_f1_idx],best_macro_th)
print('Test F1 Accuracy: ', f1_results[best_f1_idx])
print('Test Flat Accuracy: ', flat_acc_results[best_f1_idx], '\n')
bt = micro_thresholds[best_f1_idx]
best_pred_bools = [pl>micro_thresholds[best_f1_idx] for pl in pred_labels]
clf_report_optimized = classification_report(true_label,best_pred_bools, target_names=label_cols)
pickle.dump(clf_report_optimized, open('classification_report_optimized.txt','wb'))
print(clf_report_optimized)

tt = [pl > 0.5 for pl in res] #boolean output after thresholding
result = pd.DataFrame(tt,columns=catagories)
result["ID"] = test_df["ID"]
new_order = ["id","AM", "MS",   "OTHER" ,"PH",  "SF"    ,"SR"]

result = result[catagories].astype(int)
print(result)
result.to_csv("sub.csv", index=False)
print("saved")

