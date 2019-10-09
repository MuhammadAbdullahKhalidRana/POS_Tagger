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




     
   

def sblm(tagged_sentences):
     
    sentences, sentence_tags =[], [] 
    for tagged_sentence in tagged_sentences:
        sentence, tags = zip(*tagged_sentence)
        sentences.append(np.array(sentence))
        sentence_tags.append(np.array(tags))
     
    # Let's see how a sequence looks
     
    print(sentences[5])
    print(sentence_tags[5])



    #print(metrics.confusion_matrix(sentences[5],sentences[5]))
     
     
    (train_sentences, 
     test_sentences, 
     train_tags, 
     test_tags) = train_test_split(sentences, sentence_tags, test_size=0.2)

    words, tags = set([]), set([])

    for s in train_sentences:
        for w in s:
            words.add(w.lower())
     
    for ts in train_tags:
        for t in ts:
            tags.add(t)
     
     
    word2index = {w: i + 2 for i, w in enumerate(list(words))}
    word2index['-PAD-'] = 0  # The special value used for padding
    word2index['-OOV-'] = 1  # The special value used for OOVs
     
    tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
    tag2index['-PAD-'] = 0  # The special value used to padding

    print(len(tag2index))

    train_sentences_X, test_sentences_X, train_tags_y, test_tags_y = [], [], [], []
     
    for s in train_sentences:
        s_int = []
        for w in s:
            try:
                s_int.append(word2index[w.lower()])
            except KeyError:
                s_int.append(word2index['-OOV-'])
     
        train_sentences_X.append(s_int)
     
    for s in test_sentences:
        s_int = []
        for w in s:
            try:
                s_int.append(word2index[w.lower()])
            except KeyError:
                s_int.append(word2index['-OOV-'])
     
        test_sentences_X.append(s_int)
     
    for s in train_tags:
        train_tags_y.append([tag2index[t] for t in s])
     
    for s in test_tags:
        test_tags_y.append([tag2index[t] for t in s])
     
    print(train_sentences_X[0])
    print(test_sentences_X[0])
    print(train_tags_y[0])
    print(test_tags_y[0])

    MAX_LENGTH = len(max(train_sentences_X, key=len))
    print(MAX_LENGTH) 
     
     
    train_sentences_X = pad_sequences(train_sentences_X, maxlen=MAX_LENGTH, padding='post')
    test_sentences_X = pad_sequences(test_sentences_X, maxlen=MAX_LENGTH, padding='post')
    train_tags_y = pad_sequences(train_tags_y, maxlen=MAX_LENGTH, padding='post')
    test_tags_y = pad_sequences(test_tags_y, maxlen=MAX_LENGTH, padding='post')
     
    print(train_sentences_X[0])
    print(test_sentences_X[0])
    print(train_tags_y[0])
    print(test_tags_y[0])


     
    model = Sequential()
    model.add(InputLayer(input_shape=(MAX_LENGTH, )))
    model.add(Embedding(len(word2index), 128))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(TimeDistributed(Dense(len(tag2index))))
    model.add(Activation('softmax'))


    def ignore_class_accuracy(to_ignore=0):
        def ignore_accuracy(y_true, y_pred):
            y_true_class = K.argmax(y_true, axis=-1)
            y_pred_class = K.argmax(y_pred, axis=-1)
     
            ig = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
            matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ig
            accuracy = K.sum(matches) / K.maximum(K.sum(ig), 1)
            return accuracy
        return ignore_accuracy

    def recall_m(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon() )
            return recall

    def precision_m(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon() )
            return precision

    def f1_m(y_true, y_pred):
        precision = precision_m(y_true, y_pred)
        recall = recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))

    # compile the model
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])

     
    model.compile(loss='categorical_crossentropy',optimizer=Adam(0.001),metrics=['acc',ignore_class_accuracy(0)])
     
    model.summary()






    def to_categorical(sequences, categories):
        cat_sequences = []
        for s in sequences:
            cats = []
            for item in s:
                cats.append(np.zeros(categories))
                cats[-1][item] = 1.0
            cat_sequences.append(cats)
        return np.array(cat_sequences)

    cat_train_tags_y = to_categorical(train_tags_y, len(tag2index))
    print(cat_train_tags_y[0])




    # fit the model
    #history = model.fit(Xtrain, ytrain, validation_split=0.3, epochs=10, verbose=0)

    # evaluate the model
    #loss, accuracy, f1_score, precision, recall = model.evaluate(Xtest, ytest, verbose=0)    






    #print(metrics.precision_score(test[:, 0], pred[:, 0]))



    class Metrics(Callback):


     def on_train_begin(self, logs={}):
      self.val_f1s = []
      self.val_recalls = []
      self.val_precisions = []
     
     def on_epoch_end(self, epoch, logs={}):
      val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
      val_targ = self.model.validation_data[1]
      _val_f1 = f1_score(val_targ, val_predict)
      _val_recall = recall_score(val_targ, val_predict)
      _val_precision = precision_score(val_targ, val_predict)
      self.val_f1s.append(_val_f1)
      self.val_recalls.append(_val_recall)
      self.val_precisions.append(_val_precision)
      print (" — val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))
      return



    #metrics = Metrics()

    #print (Metrics().val_f1s)
    model.fit(train_sentences_X, to_categorical(train_tags_y, len(tag2index)), batch_size=128, epochs=10, validation_split=0.2)

    loss, accuracy, f1_score, precision, recall = model.evaluate(test_sentences_X, to_categorical(test_tags_y, len(tag2index)))
    #matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print("f1_score")
    print(f1_score)
    print(precision)
    print(recall)
    print(accuracy)
    #print(f"{model.metrics_names[1]}: {scores[1] * 100}")   # acc: 99.09751977804825




    #print(to_categorical(test_tags_y, len(tag2index)))



    #print(matrix)

    print("Confusion matrix")
    print(model.metrics_names)




    test_samples = [
        "I am eating an apple.".split(),
        "Cleaning is very important for health".split()
    ]
    print(test_samples)
     



    test_samples_X = []
    for s in test_samples:
        s_int = []
        for w in s:
            try:
                s_int.append(word2index[w.lower()])
            except KeyError:
                s_int.append(word2index['-OOV-'])
        test_samples_X.append(s_int)
     
    test_samples_X = pad_sequences(test_samples_X, maxlen=MAX_LENGTH, padding='post')
    print(test_samples_X)

    predictions = model.predict(test_samples_X)
    print(predictions, predictions.shape)
     
    def logits_to_tokens(sequences, index):
        token_sequences = []
        for categorical_sequence in sequences:
            token_sequence = []
            for categorical in categorical_sequence:
                token_sequence.append(index[np.argmax(categorical)])
     
            token_sequences.append(token_sequence)
     
        return token_sequences


    print(logits_to_tokens(predictions, {i: t for t, i in tag2index.items()}))



     

     


     
    model = Sequential()
    model.add(InputLayer(input_shape=(MAX_LENGTH, )))
    model.add(Embedding(len(word2index), 128))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(TimeDistributed(Dense(len(tag2index))))
    model.add(Activation('softmax'))
     
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(0.001),
                  metrics=['accuracy', ignore_class_accuracy(0),f1_m,precision_m, recall_m])
     
    model.summary()

    model.fit(train_sentences_X, to_categorical(train_tags_y, len(tag2index)), batch_size=128, epochs=1, validation_split=0.2)
     
    predictions = model.predict(test_samples_X)
    print(logits_to_tokens(predictions, {i: t for t, i in tag2index.items()}))

    return 0
