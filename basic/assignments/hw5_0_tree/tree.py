import numpy as np
from sklearn.base import BaseEstimator


def entropy(y):
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset

    Returns
    -------
    float
        Entropy of the provided subset
    """

    eps = 0.0005
    p_class = np.sum(y, axis=0) / np.sum(y)  # probability of classes
    return -np.sum(np.multiply(p_class, np.log(p_class + eps)))


def gini(y):
    """
    Computes the Gini impurity of the provided distribution

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset

    Returns
    -------
    float
        Gini impurity of the provided subset
    """

    p_class = np.sum(y, axis=0) / np.sum(y)  # probability of classes
    return 1 - np.sum(np.square(p_class))


def variance(y):
    """
    Computes the variance the provided target values subset

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector

    Returns
    -------
    float
        Variance of the provided target vector
    """

    return np.var(y)


def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector

    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """

    return np.mean(np.abs(y - np.median(y)))


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """

    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None


class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True),  # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2,
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(
            self.all_criterions.keys())

        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None  # Use the Node class to initialize it later
        self.debug = debug

    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold

        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset

        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the provided subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the provided subset where selected feature x^j >= threshold
        """

        left_indexes = np.argwhere(X_subset[:, feature_index] < threshold).ravel()
        right_indexes = np.argwhere(X_subset[:, feature_index] >= threshold).ravel()

        y_left = y_subset[left_indexes, :]
        y_right = y_subset[right_indexes, :]

        X_left = X_subset[left_indexes, :]
        X_right = X_subset[right_indexes, :]

        return (X_left, y_left), (X_right, y_right)

    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold

        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset

        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """

        left_indexes = np.argwhere(X_subset[:, feature_index] < threshold).ravel()
        right_indexes = np.argwhere(X_subset[:, feature_index] >= threshold).ravel()

        y_left = y_subset[left_indexes, :]
        y_right = y_subset[right_indexes, :]

        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion

        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset

        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """

        n_samples = len(X_subset)
        n_features = X_subset.shape[1]

        g_opt = 0
        h_q = self.criterion(y_subset)

        feature_index = None
        threshold = None

        for feature in range(n_features):
            unique_samples = np.unique(X_subset[:, feature])
            max_value = np.max(unique_samples)
            min_value = np.min(unique_samples)
            for value in unique_samples:
                if (value == min_value) or (value == max_value):
                    continue
                (X_left, y_left), (X_right, y_right) = self.make_split(feature, value, X_subset, y_subset)
                g_curr = h_q - (
                        len(y_left) * self.criterion(y_left) + len(y_right) * self.criterion(y_right)) / n_samples
                if g_curr > g_opt:
                    g_opt = g_curr
                    feature_index = feature
                    threshold = value

        return feature_index, threshold

    def make_tree(self, X_subset, y_subset):
        """
        Recursively builds the tree

        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset

        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """

        if (self.max_depth <= self.depth) or (len(X_subset) < self.min_samples_split):
            if self.classification:
                proba = np.sum(y_subset, axis=0) / np.sum(y_subset)
            else:
                proba = np.mean(y_subset) if self.criterion_name == 'variance' else np.median(y_subset)
            return Node(feature_index=None, threshold=None, proba=proba)

        feature_index, threshold = self.choose_best_split(X_subset, y_subset)

        if (feature_index is not None) and (threshold is not None):
            cur_node = Node(feature_index, threshold)
            (X_left, y_left), (X_right, y_right) = self.make_split(feature_index, threshold, X_subset, y_subset)
            self.depth += 1
            cur_node.left_child = self.make_tree(X_left, y_left)
            cur_node.right_child = self.make_tree(X_right, y_right)
            self.depth -= 1
        else:
            if self.classification:
                proba = np.sum(y_subset, axis=0) / np.sum(y_subset)
            else:
                proba = np.mean(y_subset) if self.criterion_name == 'variance' else np.median(y_subset)
            return Node(feature_index=None, threshold=None, proba=proba)

        return cur_node

    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression

        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)

    def predict(self, X):
        """
        Predict the target value or class label the model from scratch using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression

        """

        n_samples = len(X)
        y_predicted = np.zeros(n_samples)

        for i in range(n_samples):
            curr = self.root
            while curr.left_child is not None:
                if X[i, curr.feature_index] < curr.value:
                    curr = curr.left_child
                else:
                    curr = curr.right_child
            if self.classification:
                y_predicted[i] = np.argmax(curr.proba)
            else:
                y_predicted[i] = curr.proba

        return y_predicted

    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects

        """
        assert self.classification, 'Available only for classification problem'

        n_samples = len(X)
        y_predicted_probs = np.zeros((n_samples, self.n_classes))

        for i in range(n_samples):
            curr_node = self.root
            while curr_node.left_child is not None:
                if X[i, curr_node.feature_index] < curr_node.value:
                    curr_node = curr_node.left_child
                else:
                    curr_node = curr_node.right_child

            y_predicted_probs[i] = curr_node.proba

        return y_predicted_probs
