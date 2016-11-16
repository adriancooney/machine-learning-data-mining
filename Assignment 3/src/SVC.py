import numpy as np
import cvxopt
from cvxopt import matrix, solvers
from sklearn.base import BaseEstimator, ClassifierMixin

class SVC(BaseEstimator, ClassifierMixin):
    @staticmethod
    def gaussian_kernel(x, y):
        # Formula for the kernel: 
        #   K(x, x') = exp( -\sqrt{\frac{|| x - x' ||^2}{2\sigma^2}})
        return np.exp(-np.sqrt(np.linalg.norm(x - y) ** 2 / (2 * 0.5 ** 2)))
    
    def __init__(self, kernel = None):
        if not kernel:
            # Default to Radial Basis Function (RBF) kernel
            kernel = SVC.gaussian_kernel
            
        self.kernel = kernel
        self.b = 0
        
    def fit(self, X, y):
        """Fit the data into the classifier:
        
        Args:
        
            X: (numpy.ndarray)
            
            The n*m dimensional array where n is the amount of features
            and m is the amount of samples.
            
            y: (numpy.ndarray)
            
            Array of labels for the sample. This is a BINARY classifier (`bool(label)`).
        """

        # Ensure the labels are binary. Convert labels to True/False. 
        # WARNING: None, 0 and False are the considered False and any other
        # value is considered True. See what python's truthy values for more
        # information. Here we convert the labels to 1 or -1
        y = np.array([1 if bool(n) else -1 for n in y])
        
        sample_count, feature_count = X.shape
        
        # Create the memory for the kernel space. This is a matrix
        # that contains the simliarity of each vector to each other.
        K = np.zeros((sample_count, sample_count))
        
        # Generate a matrix of the similarities of each
        # feature against each other.
        for i in range(sample_count**2):
            j = i % sample_count
            k = int(np.floor(i / sample_count))

            # Apply the kernel trick (RBF in our case)
            K[j, k] = self.kernel(X[j], X[k])

        # Since the Support Vector machine is a convex optimization
        # problem, we can solve it using a convex optimzation package,
        # and in our case CVXOPT. We have to mangle our input to suit
        # the standard form QP:
        # 
        #   min_x   \frac{1}{2} x^T Px + q^T x
        #   s.t.    Gx \leq h
        #           Ax = b
        #           
        # See: https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf
        P = np.outer(y, y) * K
        q = np.ones(sample_count) * -1

        # Inequality constraint for each sample
        #    -I^(sample_count*sample_count) < 0
        G = np.diag(np.ones(sample_count) * -1)
        h = np.zeros(sample_count) 

        # Equality constraint for each samples label
        A = matrix(y, (1, sample_count), "d") # Cast to double
        b = 0.0

        # Solve the quadratic programming problem using CVXOPT
        solved = solvers.qp(*tuple(
            [matrix(u) if not isinstance(u, cvxopt.base.matrix) else u \
                for u in [P, q, G, h, A, b]]
            )
        )

        # Grab the multipliers
        multipliers = np.array(solved["x"]).flatten()

        # Now select the indices of the support vectors that are over the threshold
        selection = multipliers > 1e-5
        index = np.arange(len(multipliers))[selection]

        # Grab the support vectors
        self.support_vectors = X[selection]
        self.support_vector_multipliers = multipliers[selection]
        self.support_vector_labels = y[selection]

        # Now calculate the intercept b
        b = 0
        for i in range(len(self.support_vector_multipliers)):
            b += self.support_vector_labels[i]
            b -= np.sum(self.support_vector_multipliers * \
                self.support_vector_labels * \
                K[index[i], selection]) 

        # Normalize and save b for prediction
        self.b = b / len(self.support_vector_multipliers)

    def predict(self, X):
        """Predict the label y for each sample in X"""
        labels = []

        for i in range(len(X)):
            prediction = 0

            for multiplier, label, support_vector in zip(self.support_vector_multipliers, 
                self.support_vector_labels, self.support_vectors):

                # Calculate the label by summing all the lagrange multipliers
                prediction += multiplier * label * self.kernel(X[i], support_vector)

            labels.append(prediction)

        labels = np.array(labels) + self.b

        return np.sign(labels)

    def __repr__(self):
        """Pretty print the classifier"""
        return "SVC(kernel={}, b={})".format(self.kernel.__name__, self.b)