import numpy as np

class LRDEstimator:
    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors
        self.N, self.d = vectors.shape

        # Cached data
        self.X_unpooled = None          # Normalized unpooled embeddings, shape (N, d)
        self.S_unpooled = None          # Prefix sums of X_unpooled, shape (N+1, d)
        self.current_pool_order = None  # Which pool_order is cached
        self.X_pooled = None            # Normalized pooled embeddings, shape (N, d)
        self.S_pooled = None            # Prefix sums of X_pooled, shape (N+1, d)

        self.polarities = None          # For correlation-based method
    
    def calculate_polarities(self, standardize: bool = False):
        if standardize:
            means = np.mean(self.vectors, axis=1, keepdims=True)
            stds = np.std(self.vectors, axis=1, keepdims=True)
            stds[stds == 0] = 1.0 # Avoid division by zero - if variance is zero then calculate V - EV
            vectors_standardized = (self.vectors - means) / stds
            self.polarities = np.sum(vectors_standardized, axis=1)
        else:
            self.polarities = np.sum(self.vectors, axis=1)
        
    def calculate_corr(self, lag: int, standardize: bool = False):
        if self.polarities is None:
            self.calculate_polarities(standardize)
        return np.corrcoef(self.polarities[:-lag], self.polarities[lag:])[0,1]

    def compute_unpooled(self):
        """
        Compute and cache normalized unpooled embeddings and their prefix sums if not already cached.
        """
        if self.X_unpooled is None:
            # Compute norms and normalized embeddings
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0 # Avoid division by zero
            self.X_unpooled = self.vectors / norms

            # Build prefix sums
            self.S_unpooled = np.zeros((self.N + 1, self.d))
            self.S_unpooled[1:] = np.cumsum(self.X_unpooled, axis=0)
    
    def compute_pooled(self, pool_order: int):
        """
        Compute and cache normalized pooled embeddings and their prefix sums if not already cached.
        """
        # If we already have computed them then do nothing
        if self.current_pool_order == pool_order and self.X_pooled is not None:
            return

        # Build prefix sums: P[i] = F[0] + F[1] + ... + F[i-1]
        P = np.zeros((self.N + 1, self.d))
        P[1:] = np.cumsum(self.vectors, axis=0)

        # Pool embeddings
        pooled = np.zeros((self.N, self.d))
        for i in range(self.N):
            m = min(i + pool_order, self.N - 1)
            pooled[i] = P[m + 1] - P[i]

        # Normalize pooled embeddings
        pooled_norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled_norms[pooled_norms == 0] = 1.0
        self.X_pooled = pooled / pooled_norms

        # Build prefix sums
        self.S_pooled = np.zeros((self.N + 1, self.d))
        self.S_pooled[1:] = np.cumsum(self.X_pooled, axis=0)

        # Update pool order cache
        self.current_pool_order = pool_order

    def pool_embeddings(self, pool_order: int) -> np.ndarray:
        # Build prefix sums: P[i] = F[0] + F[1] + ... + F[i-1]
        P = np.zeros((self.N + 1, self.d))
        P[1:] = np.cumsum(self.vectors, axis=0)

        # Allocate memory for pooled embeddings
        vectors_pooled = np.zeros((self.N, self.d))

        for i in range(self.N):
            m = min(i + pool_order, self.N - 1)
            vectors_pooled[i] = P[m + 1] - P[i]
        
        return vectors_pooled
    
    def calculate_coco(self, lag: int, pool_order: int = 0) -> float:
        length = self.N - lag
        if length <= 0:
            raise ValueError(f"lag={lag} is too large for sequence of length {self.N}")

        if pool_order == 0:
            # Ensure unpooled data is computed
            self.compute_unpooled()

            # Calculate CoCo on unpooled data
            U_sum = self.S_unpooled[self.N - lag] - self.S_unpooled[0]  # sum of X[0..N-lag-1]
            V_sum = self.S_unpooled[self.N] - self.S_unpooled[lag]      # sum of X[lag..N-1]

            E_U = U_sum / length
            E_V = V_sum / length

            U = self.X_unpooled[:self.N - lag]
            V = self.X_unpooled[lag:]
            dot_products = np.sum(U * V, axis=1)
            E_UV = np.mean(dot_products)
            return E_UV - np.dot(E_U, E_V)

        else:
            # Ensure pooled data is computed
            self.compute_pooled(pool_order)

            # Calculate CoCo on pooled data
            U_sum = self.S_pooled[self.N - lag] - self.S_pooled[0]  # sum of X[0..N-lag-1]
            V_sum = self.S_pooled[self.N] - self.S_pooled[lag]      # sum of X[lag..N-1]

            E_U = U_sum / length
            E_V = V_sum / length

            U = self.X_pooled[:self.N - lag]
            V = self.X_pooled[lag:]
            dot_products = np.sum(U * V, axis=1)
            E_UV = np.mean(dot_products)
            return E_UV - np.dot(E_U, E_V)
    
    def calculate_coco_with_permutation_test(self, lag: int, pool_order: int = 0, n_permutations: int = 1000, alternative: str = "two-sided", random_state: int = None):
        rng = np.random.default_rng(seed=random_state)
        observed_coco = self.calculate_coco(lag=lag, pool_order=pool_order)

        # Generate distribution under null hypothesis
        permuted_cocos = np.zeros(n_permutations)
        for i in range(n_permutations):
            indices = rng.permutation(self.N)
            permuted_vectors = self.vectors[indices, :]
            permuted_estimator = LRDEstimator(permuted_vectors)
            permuted_cocos[i] = permuted_estimator.calculate_coco(lag=lag, pool_order=pool_order)
        
        # Calculate empirical p-value
        if alternative == "two-sided":
            p_value = np.mean(np.abs(permuted_cocos) >= np.abs(observed_coco))
        elif alternative == "greater":
            p_value = np.mean(permuted_cocos >= observed_coco)
        elif alternative == "less":
            p_value = np.mean(permuted_cocos <= observed_coco)
        else:
            raise ValueError("Alternative must be 'two-sided', 'greater' or 'less'.")
        
        return observed_coco, p_value, permuted_cocos
