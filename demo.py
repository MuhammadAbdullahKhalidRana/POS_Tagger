import nltk
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split

from keras.preprocessing.sequence import pad_sequences

from keras import backend as K

from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam
 
from keras.callbacks import Callback

from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam
from nltk.tag.util import untag
from sklearn_crfsuite import CRF
import pprint
from sklearn_crfsuite import metrics

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn import metrics 
import CrfT as crf
import BiLSTM_T as bil
import DescisionT as des


tagged_sentences = nltk.corpus.treebank.tagged_sents()
 
print(tagged_sentences[0])
print("Tagged sentences: ", len(tagged_sentences))
print("Tagged words:", len(nltk.corpus.treebank.tagged_words()))

crf.crfs(tagged_sentences)
des.dectree(tagged_sentences)
bil.sblm(tagged_sentences)


