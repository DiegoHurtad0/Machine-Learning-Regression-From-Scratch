from sklearn.tree import DecisionTreeRegressor

# The class constructor takes four parameters:
# learning_rate: the step size shrinkage used in update to prevent overfitting. Default is 0.1
# n_estimators: number of trees to be built. Default is 100
# max_depth: maximum depth of the decision tree. Default is 3
# random_state: seed for the random number generator used to initialize the decision trees.

class XGBoostRegressor:
    def __init__(self, learning_rate=0.1, n_estimators=100, max_depth=3, random_state=None):
        # initialize the class parameters
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []

    # fit method trains the XGBoostRegressor model on the input data X and target variable y.
    def fit(self, X, y):
        # for each tree in the ensemble
        for i in range(self.n_estimators):
            # create a decision tree regressor with specified max depth and random state
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
            # fit the tree on the residuals of the current prediction
            tree.fit(X, y - self.predict(X))
            # add the tree to the list of trees
            self.trees.append(tree)

    # predict method predicts the target variable for a new input data X
    def predict(self, X):
        # initialize the prediction
        y_pred = np.zeros(X.shape[0])
        # for each tree in the ensemble
        for tree in self.trees:
            # add the weighted prediction of the tree to the final prediction
            y_pred += self.learning_rate * tree.predict(X)
        # return the final prediction
        return y_pred

    # score method calculates the coefficient of determination R^2 of the prediction
    def score(self, X, y):
        # calculate the prediction
        y_pred = self.predict(X)
        # calculate the coefficient of determination R^2
        score = 1 - np.mean((y - y_pred) ** 2) / np.var(y)
        return score