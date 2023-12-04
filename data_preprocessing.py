import numpy as np
from sklearn import preprocessing

# Initial Data
data = np.array([[3, -1.5,  2, -5.4], [0,  4,  -0.3, 2.1], [1,  3.3, -1.9, -4.3]])
print("Original Data:")
print(data)

# --- Mean Removal ---
print("\n--- Mean Removal ---")
mean = data.mean(axis=0)
std_dev = data.std(axis=0)
print(f"Mean: {mean}")
print(f"Standard Deviation: {std_dev}")

data_standardized = preprocessing.scale(data)
mean_standardized = data_standardized.mean(axis=0)
std_dev_standardized = data_standardized.std(axis=0)

print(f"Mean of standardized data: {mean_standardized}")
print(f"Standard Deviation of standardized data: {std_dev_standardized}")

# --- Scaling ---
print("\n--- Scaling ---")
data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled = data_scaler.fit_transform(data)

min_original = data.min(axis=0)
max_original = data.max(axis=0)

print(f"Original Min: {min_original}")
print(f"Original Max: {max_original}")

min_scaled = data_scaled.min(axis=0)
max_scaled = data_scaled.max(axis=0)

print(f"Scaled Min: {min_scaled}")
print(f"Scaled Max: {max_scaled}")

print("Scaled Data:")
print(data_scaled)

# --- Normalization ---
print("\n--- Normalization ---")
data_normalized = preprocessing.normalize(data, norm='l1', axis=0)
print("Normalized Data (L1 norm):")
print(data_normalized)

# Checking if the absolute values sum to 1 (as expected in L1 norm)
data_norm_abs = np.abs(data_normalized)
print("Sum of absolute values in each column (should be 1):")
print(data_norm_abs.sum(axis=0))

# --- Binarization ---
print("\n--- Binarization ---")
data_binarized = preprocessing.Binarizer(threshold=1.4).transform(data)
print("Binarized Data:")
print(data_binarized)

# --- One Hot Encoding ---
print("\n--- One Hot Encoding ---")
data = np.array([[1, 1, 2], [0, 2, 3], [1, 0, 1], [0, 1, 0]])
print("Data for Encoding:")
print(data)

encoder = preprocessing.OneHotEncoder()
encoder.fit(data)
encoded_vector = encoder.transform([[1, 2, 3]]).toarray()
print("Encoded Vector for [1, 2, 3]:")
print(encoded_vector)
