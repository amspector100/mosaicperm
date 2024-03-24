import time
import numpy as np
from tqdm.auto import tqdm


def elapsed(t0):
	return np.around(time.time() - t0, 2)

def vrange(n, verbose=False):
	if not verbose:
		return range(n)
	else:
		return tqdm(list(range(n)))

def haslength(x):
	try:
		len(x)
		return True
	except TypeError:
		return False