import noise
import numpy as np
from tqdm import tqdm

def callback(n: float):
    print(f"Now process: {n * 100:.4f}%")

print("Hello world!")
p: np.ndarray[np.float32] = noise.batch_pnoise2(
    0, 0, 1000, 1000, 1024, 1024, 2, 30, callback
)
print("Finished.")

for lines in p:
    print(lines)

pass

