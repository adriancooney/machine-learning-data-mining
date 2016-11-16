import numpy as np
from itertools import groupby
from sklearn.base import BaseEstimator, ClassifierMixin
from SVC import SVC


class MulticlassSVC(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.classifiers = {} # Store the classifiers <label:str>: <LinearSVC>
        self.labels = None
    
    def fit(self, X, y):
        """Fit the data to the classifier.
        
        This generates N classifier's where N is the amount of features within the data.
        We then generate hyperplanes within each classifier to which can classify whether
        a new data point *is the feature* and *not the feature* (binary).
        
        Args:
        
            X: (numpy.ndarray)
            
            The n*m dimensional array where n is the amount of features
            and m is the amount of samples.
            
            y: (numpy.array)
            
            An array of labels for the samples. Length == sample_count
        """
        self.labels = MulticlassSVC.labels(y)
        sample_count, feature_count = X.shape
        
        print "Fitting {} samples with {} features and {} labels: {}".format(sample_count, feature_count, len(self.labels), self.labels)
        
        # Loop over each label in the set and generate a classifier
        # that can decide between is label or is not label.
        for label in self.labels:
            # Convert the labels to boolean
            ny = np.array([l == label for l in y])
            
            # Create the classifier
            classifier = SVC()
            
            # Fit the data
            classifier.fit(X, ny)
            
            # And save it for voting in the prediction
            self.classifiers[label] = classifier
    
    def predict(self, X):
        """Predict the label for each sample in X
        
        Args:
            
            X: (numpy.ndarray)
            
            The sample to predict the label for.
            
        Returns:
            y: (numpy.ndarray)
            
            Returns labels for each sample
        """
        y = []
        
        print self.classifiers

        # Loop over each sample in X
        for sample in X:
            # Loop over each classifier and check if the label returns true
            for label, classifier in self.classifiers.iteritems():
                # Prediect the label in the classifier
                [prediction] = classifier.predict([sample])
                
                # Set the sample = label if the classifier returns true
                if prediction > 0:
                    y.append(label)
                    break
            
        return np.array(y)

    def __repr__(self):
        output = "MulticlassSVC: {} labels \"{}\"\n".format(len(self.labels), "\", \"".join(self.labels))

        for label, classifier in self.classifiers.iteritems():
            output += "  {:>12} -> {}\n".format(label, repr(classifier))

        return output
    
    @staticmethod
    def labels(y):
        """Group the labels and return them.
        
        Args:
            y: (list) List of labels.
        """
        
        return [label for label, ls in groupby(sorted(y))]