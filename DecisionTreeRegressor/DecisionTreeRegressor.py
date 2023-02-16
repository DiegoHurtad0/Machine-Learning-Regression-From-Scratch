class DecisionTreeRegressor:    
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 criterion='mse', random_state=None):
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.random_state = random_state
        self.root = None

    def fit(self, X, y):
        if self.random_state:
            np.random.seed(self.random_state)
        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._predict(x, self.root) for x in X])

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape

        # If max_depth is set, stop growing the tree if current depth >= max_depth
        if (self.max_depth is not None and depth >= self.max_depth) or n_samples <= self.min_samples_leaf:
            node = {'leaf': True, 'value': np.mean(y)}
            return node
        
        # Stop growing the tree if there are not enough samples to split
        if n_samples < self.min_samples_split:
            node = {'leaf': True, 'value': np.mean(y)}
            return node

        # Calculate criterion for each feature and split on the feature with the best criterion
        best_feature = None
        best_threshold = None
        best_score = float('inf')
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                mask = X[:, feature] <= threshold
                if self.criterion == 'mse':
                    score = np.mean((y[mask] - np.mean(y[mask])) ** 2) + np.mean((y[~mask] - np.mean(y[~mask])) ** 2)
                if score < best_score:
                    best_feature = feature
                    best_threshold = threshold
                    best_score = score

        mask = X[:, best_feature] <= best_threshold
        node = {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': self._build_tree(X[mask], y[mask], depth=depth + 1),
            'right': self._build_tree(X[~mask], y[~mask], depth=depth + 1),
        }
        return node

    def _predict(self, x, node):
        if node['leaf']:
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._predict(x, node['left'])
        else:
            return self._predict(x, node['right'])