import nltk
import pprint

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn import metrics 
   

def dectree(tagged_sentences):  

    def features(sentence, index):
        """ sentence: [w1, w2, ...], index: the index of the word """
        return {
            'word': sentence[index],
            'is_first': index == 0,
            'is_last': index == len(sentence) - 1,
            'is_capitalized': sentence[index][0].upper() == sentence[index][0],
            'is_all_caps': sentence[index].upper() == sentence[index],
            'is_all_lower': sentence[index].lower() == sentence[index],
            'prefix-1': sentence[index][0],
            'prefix-2': sentence[index][:2],
            'prefix-3': sentence[index][:3],
            'suffix-1': sentence[index][-1],
            'suffix-2': sentence[index][-2:],
            'suffix-3': sentence[index][-3:],
            'prev_word': '' if index == 0 else sentence[index - 1],
            'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
            'has_hyphen': '-' in sentence[index],
            'is_numeric': sentence[index].isdigit(),
            'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
        }
     
     
    pprint.pprint(features(['This', 'is', 'a', 'sentence'], 2))

    def untag(tagged_sentence):
        return [w for w, t in tagged_sentence]

    cutoff = int(.75 * len(tagged_sentences))
    training_sentences = tagged_sentences[:cutoff]
    test_sentences = tagged_sentences[cutoff:]
     
    print (len(training_sentences))   # 2935
    print  ( len(test_sentences) )        # 979
     
    def transform_to_dataset(tagged_sentences):
        X, y = [], []
     
        for tagged in tagged_sentences:
            for index in range(len(tagged)):
                X.append(features(untag(tagged), index))
                y.append(tagged[index][1])
     
        return X, y
     
    X, y = transform_to_dataset(training_sentences)

     
    clf = Pipeline([
        ('vectorizer', DictVectorizer(sparse=False)),
        ('classifier', DecisionTreeClassifier(criterion='entropy'))
    ])
     
    clf.fit(X[:10000], y[:10000])   # Use only the first 10K samples if you're running it multiple times. It takes a fair bit :)
     
    print ('Training completed')

     
    X_test, y_test = transform_to_dataset(test_sentences)
    print("DecisionTree Accuracy:", clf.score(X_test, y_test))
   # y_pred = clf.predict(X_test)
   
    return 0

     
