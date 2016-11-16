import pandas
import numpy as np
from os import path
from MulticlassSVC import MulticlassSVC
from sklearn.cross_validation import cross_val_score

# Read in the data from the CSV file
data = pandas.read_csv(path.join(path.dirname(__file__), "../data/owls.csv"))

def test_multiclass_svc_cv():
    owl_classifier = MulticlassSVC()

    X = data[["body-length", "body-width", "wing-length", "wing-width"]].values
    y = data["species"].values
    folds = 5

    print "Cross validation ({} folds) score: {:.3f}".format(
        folds, np.mean(cross_val_score(owl_classifier, X, y, cv=folds, verbose=5))
    )

def test_multiclass_svc():
    owl_classifier = MulticlassSVC()

    X = data[["body-length", "body-width", "wing-length", "wing-width"]].values
    y = data["species"].values

    np.random.seed(13)
    index = np.arange(len(X))
    split = int(np.floor(len(index) * 0.66))

    def score_classifier():
        np.random.shuffle(index)

        # Split the samples in train/test
        train_index, test_index = index[:split], index[split:]
        train_features, train_labels = X[train_index], y[train_index]
        test_features, test_labels = X[test_index], y[test_index]

        # Fit the training data
        owl_classifier.fit(train_features, train_labels)

        # Score the classifier (inherited from ClassifierMixin)
        return owl_classifier.score(test_features, test_labels)

    scores = [score_classifier() for i in range(10)]

    print "Train/test split: %d/%d (%d total)" % (split, len(X) - split, len(X))
    print "Scores: " + ", ".join([("%.2f" % s) for s in scores])
    print "Mean score: %.3f" % np.mean(np.array(scores))
