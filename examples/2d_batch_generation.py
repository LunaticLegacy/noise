import noise
import numpy as np

import threading as th
from tqdm import tqdm

print("Hello world!")

def fetch_progress(arg: float) -> float:
    global total_steps
    steps_done = int(arg * total_steps)
    bar.n = steps_done
    bar.refresh()
    return arg

def worker():
    noise.batch_snoise3(
        0.0, 0.0, 0.0,
        1000.0, 1000.0, 1000.0,
        1024.0, 1024.0, 1024.0,
        2.0, 30.0,
        fetch_progress
    )

# total steps, for tqdm.
total_width: int = int(1000 / 30) + 1
total_height: int = int(1000 / 30) + 1
total_steps: int = total_width * total_height
bar = tqdm(range(0, total_steps), desc="Generating Perlin noise...")

working_thread = th.Thread(target=worker)
working_thread.start()
working_thread.join()

bar.close()

print("Finished.")
