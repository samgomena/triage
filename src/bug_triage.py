# Required dependences
# 1. NLTK
# 2. Gensim for word2vec
# 3. Keras with tensorflow/theano backend

import json, re, nltk, string
from os import path, sep

import numpy as np
np.random.seed(1337)
from nltk.corpus import wordnet
from gensim.models import Word2Vec
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge
from keras.optimizers import RMSprop
from keras.utils import np_utils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics.pairwise import cosine_similarity

print("Finished importing dependencies")

cwd = path.dirname(__file__)

open_bugs_json = path.join(path.abspath('./'), 'data/deep_data.json')
closed_bugs_json = path.join(path.abspath('./'), 'data/train_test_json/classifier_data_10.json')

#========================================================================================
# Initializeing Hyper parameter
#========================================================================================
#1. Word2vec parameters
min_word_frequency_word2vec = 5
embed_size_word2vec = 100
context_window_word2vec = 5

#2. Classifier hyperparameters
numCV = 10  # Cross validators
max_sentence_len = 50
min_sentence_length = 15
rankK = 10
batch_size = 32

CUTOFF = -1  # Used to truncate data to reduce processing power needed

#========================================================================================
# Preprocess the open bugs, extract the vocabulary and learn the word2vec representation
#========================================================================================
with open(open_bugs_json) as data_file:
    data = json.load(data_file, strict=False)

all_data = []
CUTOFF = len(data) // 4

# regex for removing hex values
hex_sub = re.compile(r'(\w+)0x\w+')

for item in data[:CUTOFF]:
    #1. Remove \r 
    current_title = item['issue_title'].replace('\r', ' ')
    current_desc = item['description'].replace('\r', ' ')    
    #2. Remove URLs
    current_desc = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', current_desc)    
    #3. Remove Stack Trace
    start_loc = current_desc.find("Stack trace:")
    current_desc = current_desc[:start_loc]    
    #4. Remove hex code
    current_desc = re.sub(hex_sub, '', current_desc)
    current_title = re.sub(hex_sub, '', current_title)    
    #5. Change to lower case
    current_desc = current_desc.lower()
    current_title = current_title.lower()    
    #6. Tokenize
    current_desc_tokens = nltk.word_tokenize(current_desc)
    current_title_tokens = nltk.word_tokenize(current_title)
    #7. Strip trailing punctuation marks    
    current_desc_filter = [word.strip(string.punctuation) for word in current_desc_tokens]
    current_title_filter = [word.strip(string.punctuation) for word in current_title_tokens]      
    #8. Join the lists
    current_data = current_title_filter + current_desc_filter
    current_data = filter(None, current_data)
    all_data.append(current_data)  

print("Finished processing open bug data")


# Learn the word2vec model and extract vocabulary
print("Creating vector space model from open bug data")
wordvec_model = Word2Vec(all_data, min_count=min_word_frequency_word2vec, size=embed_size_word2vec, window=context_window_word2vec, workers=4)
vocabulary = wordvec_model.wv.vocab
vocab_size = len(vocabulary)

print("Vocab size of open bug data: {0}".format(vocab_size))

# wordvec_model.save(ath.join(path.curdir + sep, "word2vec_model"))
# print("Saved Word2Vec model to the current directory")


#========================================================================================
# Preprocess the closed bugs, using the extracted the vocabulary
#========================================================================================
with open(closed_bugs_json) as data_file:
    data = json.load(data_file, strict=False)

all_data = []
all_owner = []    


for item in data[:CUTOFF]:
    #1. Remove \r 
    current_title = item['issue_title'].replace('\r', ' ')
    current_desc = item['description'].replace('\r', ' ')
    #2. Remove URLs
    current_desc = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', current_desc)
    #3. Remove Stack Trace
    start_loc = current_desc.find("Stack trace:")
    current_desc = current_desc[:start_loc]
    #4. Remove hex code
    current_desc = re.sub(hex_sub, '', current_desc)
    current_title= re.sub(hex_sub, '', current_title)
    #5. Change to lower case
    current_desc = current_desc.lower()
    current_title = current_title.lower()
    #6. Tokenize
    current_desc_tokens = nltk.word_tokenize(current_desc)
    current_title_tokens = nltk.word_tokenize(current_title)
    #7. Strip punctuation marks
    current_desc_filter = [word.strip(string.punctuation) for word in current_desc_tokens]
    current_title_filter = [word.strip(string.punctuation) for word in current_title_tokens]       
    #8. Join the lists
    current_data = current_title_filter + current_desc_filter
    current_data = filter(None, current_data)
    all_data.append(current_data)
    all_owner.append(item['owner'])

print("Completed closed bug data")

#========================================================================================
# Split cross validation sets and perform deep learning + softamx based classification
#========================================================================================
totalLength = len(all_data)
splitLength = int(totalLength / (numCV + 1))

for i in range(1, numCV + 1):
    # Split cross validation set
    print("Starting work on cross validation set {0}".format(i))
    train_data = all_data[:i*splitLength-1]
    test_data = all_data[i*splitLength:(i+1)*splitLength-1]
    train_owner = all_owner[:i*splitLength-1]
    test_owner = all_owner[i*splitLength:(i+1)*splitLength-1]
    
    # Remove words outside the vocabulary
    updated_train_data = []    
    updated_train_data_length = []    
    updated_train_owner = []
    final_test_data = []
    final_test_owner = []
    for j, item in enumerate(train_data):
        current_train_filter = [word for word in item if word in vocabulary]
        if len(current_train_filter) >= min_sentence_length:  
          updated_train_data.append(current_train_filter)
          updated_train_owner.append(train_owner[j])  
          
    for j, item in enumerate(test_data):
        current_test_filter = [word for word in item if word in vocabulary]  
        if len(current_test_filter) >= min_sentence_length:
          final_test_data.append(current_test_filter)          
          final_test_owner.append(test_owner[j])          
    
    # Remove data from test set that is not there in train set
    train_owner_unique = set(updated_train_owner)
    test_owner_unique = set(final_test_owner)
    unwanted_owner = list(test_owner_unique - train_owner_unique)
    updated_test_data = []
    updated_test_owner = []
    updated_test_data_length = []
    for j in range(len(final_test_owner)):
        if final_test_owner[j] not in unwanted_owner:
            updated_test_data.append(final_test_data[j])
            updated_test_owner.append(final_test_owner[j])

    unique_train_label = list(set(updated_train_owner))
    classes = np.array(unique_train_label)
    
    # Create train and test data for deep learning + softmax
    X_train = np.empty(shape=[len(updated_train_data), max_sentence_len, embed_size_word2vec], dtype='float32')
    Y_train = np.empty(shape=[len(updated_train_owner), 1], dtype='int32')
    # 1 - start of sentence, # 2 - end of sentence, # 0 - zero padding. Hence, word indices start with 3 
    for j, curr_row in enumerate(updated_train_data):
        sequence_cnt = 0         
        for item in curr_row:
            if item in vocabulary:
                X_train[j, sequence_cnt, :] = wordvec_model[item] 
                sequence_cnt = sequence_cnt + 1                
                if sequence_cnt == max_sentence_len-1:
                          break                
        for k in range(sequence_cnt, max_sentence_len):
            X_train[j, k, :] = np.zeros((1, embed_size_word2vec))        
        Y_train[j, 0] = unique_train_label.index(updated_train_owner[j])
    
    X_test = np.empty(shape=[len(updated_test_data), max_sentence_len, embed_size_word2vec], dtype='float32')
    Y_test = np.empty(shape=[len(updated_test_owner),1], dtype='int32')
    # 1 - start of sentence, # 2 - end of sentence, # 0 - zero padding. Hence, word indices start with 3 
    for j, curr_row in enumerate(updated_test_data):
        sequence_cnt = 0          
        for item in curr_row:
            if item in vocabulary:
                X_test[j, sequence_cnt, :] = wordvec_model[item] 
                sequence_cnt = sequence_cnt + 1                
                if sequence_cnt == max_sentence_len-1:
                          break                
        for k in range(sequence_cnt, max_sentence_len):
            X_test[j, k, :] = np.zeros((1, embed_size_word2vec))        
        Y_test[j, 0] = unique_train_label.index(updated_test_owner[j])
        
    y_train = np_utils.to_categorical(Y_train, len(unique_train_label))
    y_test = np_utils.to_categorical(Y_test, len(unique_train_label))


    # TODO: Add x_train and x_test
    
    # Construct the deep learning model
    print("Creating Model")
    sequence = Input(shape=(max_sentence_len, embed_size_word2vec), dtype='float32')
    forwards_1 = LSTM(1024)(sequence)
    after_dp_forward_4 = Dropout(0.20)(forwards_1) 
    backwards_1 = LSTM(1024, go_backwards=True)(sequence)
    after_dp_backward_4 = Dropout(0.20)(backwards_1)         
    merged = merge([after_dp_forward_4, after_dp_backward_4], mode='concat', concat_axis=-1)
    after_dp = Dropout(0.5)(merged)
    output = Dense(len(unique_train_label), activation='softmax')(after_dp)                
    model = Model(input=sequence, output=output)            
    rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])    
    hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=20)  # Rename nb_epochs to epochs // Value original: 200
    
    predict = model.predict(X_test)        
    accuracy = []
    sortedIndices = []
    pred_classes = []
    if len(predict) == 0:
        exit(1)  # Avoid divide by zero
    for ll in predict:
          sortedIndices.append(sorted(range(len(ll)), key=lambda ii: ll[ii], reverse=True))
    for k in range(1, rankK + 1):
          id = 0
          trueNum = 0
          for sortedInd in sortedIndices:
            pred_classes.append(classes[sortedInd[:k]])
            if y_test[id] in classes[sortedInd[:k]]:
                  trueNum += 1            
            id += 1
          accuracy.append((float(trueNum) / len(predict)) * 100)
    print("Test accuracy: ", accuracy)       
    
    train_result = hist.history        
    print(train_result)
    del model

    
#========================================================================================
# Split cross validation sets and perform baseline classifiers
#========================================================================================    
    
totalLength = len(all_data)
splitLength = totalLength / (numCV + 1)

for i in range(1, numCV+1):
    # Split cross validation set
    print("Starting cross validation {0}".format(i))
    train_data = all_data[:i*splitLength-1]
    test_data = all_data[i*splitLength:(i+1)*splitLength-1]
    train_owner = all_owner[:i*splitLength-1]
    test_owner = all_owner[i*splitLength:(i+1)*splitLength-1]
    
    # Remove words outside the vocabulary
    updated_train_data = []    
    updated_train_data_length = []    
    updated_train_owner = []
    final_test_data = []
    final_test_owner = []
    for j, item in enumerate(train_data):
        current_train_filter = [word for word in item if word in vocabulary]
        if len(current_train_filter)>=min_sentence_length:  
          updated_train_data.append(current_train_filter)
          updated_train_owner.append(train_owner[j])  
          
    for j, item in enumerate(test_data):
        current_test_filter = [word for word in item if word in vocabulary]  
        if len(current_test_filter)>=min_sentence_length:
          final_test_data.append(current_test_filter)          
          final_test_owner.append(test_owner[j])          
    
    # Remove data from test set that is not there in train set
    train_owner_unique = set(updated_train_owner)
    test_owner_unique = set(final_test_owner)
    unwanted_owner = list(test_owner_unique - train_owner_unique)
    updated_test_data = []
    updated_test_owner = []
    updated_test_data_length = []
    for j in range(len(final_test_owner)):
        if final_test_owner[j] not in unwanted_owner:
            updated_test_data.append(final_test_data[j])
            updated_test_owner.append(final_test_owner[j])  
    
    train_data = []
    for item in updated_train_data:
          train_data.append(' '.join(item))
         
    test_data = []
    for item in updated_test_data:
          test_data.append(' '.join(item))
    
    vocab_data = []
    for item in vocabulary:
          vocab_data.append(item)
    
    # Extract tf based bag of words representation
    tfidf_transformer = TfidfTransformer(use_idf=False)
    count_vect = CountVectorizer(min_df=1, vocabulary= vocab_data,dtype=np.int32)
    
    train_counts = count_vect.fit_transform(train_data)       
    train_feats = tfidf_transformer.fit_transform(train_counts)
    print(train_feats.shape)
    
    test_counts = count_vect.transform(test_data)
    test_feats = tfidf_transformer.transform(test_counts)
    print(test_feats.shape)
    print("=" * 20)
    
    # # perform classifification
    # for classifier in range(1, 5):
    #     #classifier = 3 # 1 - Naive Bayes, 2 - Softmax, 3 - cosine distance, 4 - SVM
    #     print(classifier) 
    #     if classifier == 1:            
    #         classifierModel = MultinomialNB(alpha=0.01)        
    #         classifierModel = OneVsRestClassifier(classifierModel).fit(train_feats, updated_train_owner)
    #         predict = classifierModel.predict_proba(test_feats)  
    #         classes = classifierModel.classes_  
            
    #         accuracy = []
    #         sortedIndices = []
    #         pred_classes = []
    #         for ll in predict:
    #             sortedIndices.append(sorted(range(len(ll)), key=lambda ii: ll[ii], reverse=True))
    #         for k in range(1, rankK+1):
    #             id = 0
    #             trueNum = 0
    #             for sortedInd in sortedIndices:            
    #                 if updated_test_owner[id] in classes[sortedInd[:k]]:
    #                     trueNum += 1
    #                     pred_classes.append(classes[sortedInd[:k]])
    #                 id += 1
    #             accuracy.append((float(trueNum) / len(predict)) * 100)
    #         print accuracy                                    
    #     elif classifier == 2:            
    #         classifierModel = LogisticRegression(solver='lbfgs', penalty='l2', tol=0.01)
    #         classifierModel = OneVsRestClassifier(classifierModel).fit(train_feats, updated_train_owner)
    #         predict = classifierModel.predict(test_feats)
    #         classes = classifierModel.classes_ 
            
    #         accuracy = []
    #         sortedIndices = []
    #         pred_classes = []
    #         for ll in predict:
    #             sortedIndices.append(sorted(range(len(ll)), key=lambda ii: ll[ii], reverse=True))
    #         for k in range(1, rankK+1):
    #             id = 0
    #             trueNum = 0
    #             for sortedInd in sortedIndices:            
    #                 if updated_test_owner[id] in classes[sortedInd[:k]]:
    #                     trueNum += 1
    #                     pred_classes.append(classes[sortedInd[:k]])
    #                 id += 1
    #             accuracy.append((float(trueNum) / len(predict)) * 100)
    #         print accuracy                                   
    #     elif classifier == 3:            
    #         predict = cosine_similarity(test_feats, train_feats)
    #         classes = np.array(updated_train_owner)
    #         classifierModel = []
            
    #         accuracy = []
    #         sortedIndices = []
    #         pred_classes = []
    #         for ll in predict:
    #             sortedIndices.append(sorted(range(len(ll)), key=lambda ii: ll[ii], reverse=True))
    #         for k in range(1, rankK+1):
    #             id = 0
    #             trueNum = 0
    #             for sortedInd in sortedIndices:            
    #                 if updated_test_owner[id] in classes[sortedInd[:k]]:
    #                     trueNum += 1
    #                     pred_classes.append(classes[sortedInd[:k]])
    #                 id += 1
    #             accuracy.append((float(trueNum) / len(predict)) * 100)
    #         print accuracy                        
    #     elif classifier == 4:            
    #         classifierModel = svm.SVC(probability=True, verbose=False, decision_function_shape='ovr', random_state=42)
    #         classifierModel.fit(train_feats, updated_train_owner)
    #         predict = classifierModel.predict(test_feats)
    #         classes = classifierModel.classes_ 
        
    #         accuracy = []
    #         sortedIndices = []
    #         pred_classes = []
    #         for ll in predict:
    #             sortedIndices.append(sorted(range(len(ll)), key=lambda ii: ll[ii], reverse=True))
    #         for k in range(1, rankK+1):
    #             id = 0
    #             trueNum = 0
    #             for sortedInd in sortedIndices:            
    #                 if updated_test_owner[id] in classes[sortedInd[:k]]:
    #                     trueNum += 1
    #                     pred_classes.append(classes[sortedInd[:k]])
    #                 id += 1
    #             accuracy.append((float(trueNum) / len(predict)) * 100)
    #         print accuracy                        
