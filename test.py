import torch

def pca(X_train, k):
    # Center the training data
    X_mean = torch.mean(X_train, dim=0)
    X_centered = X_train - X_mean

    # Compute the SVD
    U, S, V = torch.svd(X_centered)

    # Select the top k principal components
    U_reduced = U[:, :k]

    return U_reduced, X_mean


def project_new_data(new_data, U_reduced, X_mean):
    # Center the new data using the mean from the training data
    new_data_centered = new_data - X_mean

    # Project the new data onto the reduced subspace
    new_data_pca = torch.matmul(new_data_centered, U_reduced.T)

    return new_data_pca


# Example training data
X_train = torch.randn(100, 5)  # 100 samples, 5 features

# Perform PCA on training data
k = 2  # Number of principal components
U_reduced, X_mean = pca(X_train, k)

# Example new data point
new_data_point = torch.randn(1, 5)  # New data point with 5 features

# Project new data onto reduced subspace
new_data_point_pca = project_new_data(new_data_point, U_reduced, X_mean)

print("Projected new data point shape:", new_data_point_pca.shape)
