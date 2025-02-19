import time
from pathlib import Path

import numpy as np

data = np.random.randn(3, 3)

filename = "test"

np.save("test", data)

print(Path(filename + ".npy").stat())

time.sleep(10)

np.load(filename + ".npy")
print(Path(filename + ".npy").stat())