import os
import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import Optional

URL_DIR = "https://raw.githubusercontent.com/amspector100/mosaic_factor_paper/master"
STOCK_URL = f"{URL_DIR}/data/sample_data/sp_returns.csv"
EXPOSURE_URL = f"{URL_DIR}/data/sample_data/sp_exposures.csv"

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

def load_sample_dataset(
	cache: bool=True,
	data_home: Optional[str]=None,
	**kwargs
):
	"""
	Load an example dataset (requires internet).

	Parameters
	----------
	cache : bool
		If True, caches the loaded dataset (~20 MB).
	data_home : str
		Optional directory to save/load the dataset from.

	Returns
	-------
	outcomes : pd.DataFrame
		outcome data for S&P 500, 2013-2022
	exposures : pd.DataFrame	
		exposure data for S&P 500, 2013-2022

	Notes
	-----
	This function mimics :func:`seaborn.load_dataset`.
	"""
	from urllib.request import urlretrieve
	from seaborn import get_data_home
	if cache:
		# Cache locations
		data_home = get_data_home(data_home).replace("seaborn", "mosaicperm")
		if not os.path.exists(data_home):
			os.makedirs(data_home)
		stock_path = os.path.join(data_home, 'sp_returns.csv')
		exposure_path = os.path.join(data_home, 'sp_exposures.csv')
		# download if needed
		if not os.path.exists(stock_path):
			urlretrieve(STOCK_URL, stock_path)
		if not os.path.exists(exposure_path):
			urlretrieve(EXPOSURE_URL, exposure_path)
	else:
		stock_path = STOCK_URL
		exposure_path = EXPOSURE_URL
	# read data
	outcomes = pd.read_csv(stock_path, **kwargs).set_index("Date")
	outcomes.index = pd.to_datetime(outcomes.index)
	outcomes.columns.name = 'Symbol'
	exposures = pd.read_csv(exposure_path, **kwargs).set_index("Symbol")
	exposures.columns.name = 'Factor'
	# return
	return outcomes, exposures