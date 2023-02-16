from sklearn.tree import DecisionTreeRegressor


class RandomForestRegressor:

    def __init__(self, n_trees=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None,
                 oob_score=False, random_state=None):
        # Initialize hyperparameters and model attributes
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.oob_score = oob_score
        self.random_state = random_state
        self.trees = []
        self.oob_predictions = None

    def fit(self, X, y):
        # Fit the random forest to the training data
        n_samples, n_features = X.shape

        # Set the default value for max_features to be the square root of the number of features
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        # Grow each decision tree in the forest
        for i in range(self.n_trees):
            # Generate a random sample of the data with replacement
            random_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_random = X[random_indices, :]
            y_random = y[random_indices]

            # Sample a subset of features
            # feature_indices = np.random.choice(n_features, self.max_features, replace=False)
            # X_random = X_random[:, feature_indices]

            # Fit a decision tree to the random sample
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
            tree.fit(X_random, y_random)

            # Store the tree
            self.trees.append(tree)

        # Calculate out-of-bag predictions
        if self.oob_score:
            # Initialize arrays to hold the predictions and number of predictions for each sample
            self.oob_predictions = np.zeros((n_samples,))
            n_predictions = np.zeros((n_samples,))

            # Loop over the trees and their corresponding out-of-bag samples
            for i in range(self.n_trees):
                tree = self.trees[i]
                oob_indices = np.setdiff1d(np.arange(n_samples), random_indices)
                if oob_indices.size > 0:
                    # Calculate the out-of-bag predictions for this tree
                    X_oob = X[oob_indices, :]
                    X_oob = X_oob[:, feature_indices]
                    y_oob_pred = tree.predict(X_oob)
                    self.oob_predictions[oob_indices] += y_oob_pred
                    n_predictions[oob_indices] += 1

            # Calculate the mean out-of-bag prediction and error
            self.oob_predictions /= n_predictions
            self.oob_error = np.sqrt(np.mean((y - self.oob_predictions) ** 2))

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        # Use the random forest to make predictions on new data
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)
        for tree in self.trees:
            y_pred += tree.predict(X)
        y_pred /= self.n_trees
        return y_pred
