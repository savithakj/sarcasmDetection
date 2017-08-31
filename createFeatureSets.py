"""
This file is responsible to take the raw sentences as input and then extract
features from them.
These feature are then stored as a numpy array in to file, that can be easily
consumed by the model.
This file needs two files with positive and regular data that is cleaned and
pre-processed. The names of the files should be 'negproc.npy' and 'posproc.npy'

@Author Sanjay Haresh Khatwani (sxk6714@rit.edu)
@Author Savitha Jayasankar (skj9180@rit.edu)
@Author Saurabh Parekh (sbp4709@rit.edu)
"""

import numpy as np
from textblob import TextBlob
import nltk
import string
import exp_replace
import random


class CreateFeatureSet:
    # Read the data from numpy files into arrays
    def __init__(self):
        self.sarcastic_data = np.load('posproc.npy')
        self.regular_data = np.load('negproc.npy')

        self.featuresets = []
        self.classes = ["SARCASTIC", "REGULAR"]

    def extract_features(self):
        """
        This method drives the feature extraction which are stored in featuresets
        array.
        :return:
        """
        print("We have " + str(len(self.sarcastic_data)) + " Sarcastic sentences.")
        print("We have " + str(len(self.regular_data)) + " Regular sentences.")

        print("Extracting features for negative set")
        # We have 4 times more Regular data as Positive data. Hence we only take
        # every 4th sentence from the Regular data.
        for x in self.regular_data[::4]:
            features = self.extract_feature_of_sentence(x)
            self.featuresets.append([features, [0, 1]])

        print("Extracting features for positive set")
        for x in self.sarcastic_data:
            features = self.extract_feature_of_sentence(x)
            self.featuresets.append([features, [1, 0]])

        # Shuffle the feature sets so that thy are not in any particular order
        random.shuffle(self.featuresets)
        featuresets1 = np.array(self.featuresets)

        # Save the features into a numpy file.
        np.save('feature sets', featuresets1)

    def extract_feature_of_sentence(self, sen):
        # type: (object) -> object
        """
        This method extracts features of a single sentence.
        We have following list of features being extracted.
        1. Full sentence Polarity
        2. Full sentence Subjectivity
        3. Half sentence Polarity (1/2 and 2/2)
        4. Half sentence Subjectivity (1/2 and 2/2)
        5. Difference between polarities of two halves
        6. Third sentence Polarity (1/3, 2/3 and 3/3)
        7. Third sentence Subjectivity (1/3, 2/3 and 3/3)
        8. Difference between max and min polarity of the thirds.
        9. Fourth sentence Polarity (1/4, 2/4, 3/4 and 4/4)
        10. Fourth sentence Subjectivity (1/4, 2/4, 3/4 and 4/4)
        11. Difference between max and min polarities of the fourths.

        Like this we extract 23 features of a single sentence.
        :param sen:
        :return:
        """
        features = []

        # Tokenize the sentence and then convert everything to lower case.
        tokens = nltk.word_tokenize(exp_replace.replace_emo(str(sen)))
        tokens = [(t.lower()) for t in tokens]

        # Extract features of full sentence.
        fullBlob = TextBlob(self.join_tokens(tokens))
        features.append(fullBlob.sentiment.polarity)
        features.append(fullBlob.sentiment.subjectivity)

        # Extract features of halves.
        size = len(tokens) // 2
        parts = []
        i = 0
        while i <= len(tokens):
            if i == size:
                parts.append(tokens[i:])
                break
            else:
                parts.append(tokens[i:i + size])
                i += size
        for x in range(0, len(parts)):
            part = parts[x]
            halfBlob = TextBlob(self.join_tokens(part))
            features.append(halfBlob.sentiment.polarity)
            features.append(halfBlob.sentiment.subjectivity)
        features.append(np.abs(features[-2] - features[-4]))

        # Extract features of thirds.
        size = len(tokens) // 3
        parts = []
        i = 0
        while i <= len(tokens):
            if i == 2 * size:
                parts.append(tokens[i:])
                break
            else:
                parts.append(tokens[i:i + size])
                i += size

        ma = -2
        mi = 2
        for x in range(0, len(parts)):
            part = parts[x]
            thirdsBlob = TextBlob(self.join_tokens(part))
            pol = thirdsBlob.sentiment.polarity
            sub = thirdsBlob.sentiment.subjectivity
            if pol > ma:
                ma = pol
            if pol < mi:
                mi = pol
            features.append(pol)
            features.append(sub)
        features.append(np.abs(ma - mi))

        # Extract features of fourths.
        size = len(tokens) // 4
        parts = []
        i = 0
        while i <= len(tokens):
            if i == 3 * size:
                parts.append(tokens[i:])
                break
            else:
                parts.append(tokens[i:i + size])
                i += size
        ma = -2
        mi = 2
        for x in range(0, len(parts)):
            part = parts[x]
            fourths_blob = TextBlob(self.join_tokens(part))
            pol = fourths_blob.sentiment.polarity
            sub = fourths_blob.sentiment.subjectivity
            if pol > ma:
                ma = pol
            if pol < mi:
                mi = pol
            features.append(pol)
            features.append(sub)
        features.append(np.abs(ma - mi))

        return features

    def join_tokens(self, t):
        """
        This method joins tokes into a single text avoiding punctuations and
        special characters as required by the textblob api.
        :param t:
        :return:
        """
        s = ""
        for i in t:
            if i not in string.punctuation and not i.startswith("'"):
                s += (" " + i)
        return s.strip()


if __name__ == '__main__':
    CreateFeatureSet().extract_features()
