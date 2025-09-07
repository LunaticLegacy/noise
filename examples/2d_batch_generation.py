import noise
import numpy as np

print("Hello world!")
p: np.ndarray[np.float32] = noise.batch_pnoise2(0.0, 0.0, 1000.0, 1000.0, 1024.0, 1024.0, 2.0, 30.0)
print("Finished.")

for lines in p:
    print(lines)

pass

