import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../src')
from load_data import load_UT_HAR_data

data = load_UT_HAR_data('../data')

for key, value in data.items():
    print(key, value.shape)
    

unique, counts = np.unique(data['y_train'].numpy(), return_counts=True)
print(dict(zip(unique, counts)))


fall_idx = (data['y_train'] == 1).nonzero()[0][0].item()
walk_idx = (data['y_train'] == 6).nonzero()[0][0].item()

fall_sample = data['X_train'][fall_idx, 0]  # shape 250x90
walk_sample = data['X_train'][walk_idx, 0]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].imshow(fall_sample.T, aspect='auto', cmap='viridis')
axes[0].set_title('Label 1 (assumed: fall)')
axes[1].imshow(walk_sample.T, aspect='auto', cmap='viridis')
axes[1].set_title('Label 6 (assumed: walk)')
plt.tight_layout()
plt.savefig('../results/fall_vs_walk.png')
plt.show()