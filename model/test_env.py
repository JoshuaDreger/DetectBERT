import numpy as np
import pickle

print(np.__version__)
test_array = np.array([1, 2, 3])
print(test_array)

# Pickle test
with open("test.pkl", "wb") as f:
    pickle.dump(test_array, f)

with open("test.pkl", "rb") as f:
    loaded_array = pickle.load(f)
print(loaded_array)
