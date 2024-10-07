import os
import time
import numpy as np
import pandas as pd
import scipy.sparse
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

def get_dummies_sparse(
	data: pd.DataFrame,
	missing_pattern: Optional[np.array]=None,
	drop_first: bool=True,
):
	"""
	Creates sparse dummies based on a dataframe.
	
	Parameters
	----------
	data : pd.DataFrame
		DataFrame of discrete variables to create dummies.
	missing_pattern : np.array
		Boolean n-length array indicating missing outcomes.
		I.e., outcome i is missing iff missing_pattern[i] = True.
	drop_first : bool
		Whether to get k-1 dummies out of k categorical levels by 
		removing the first level.
	
	Returns
	-------
	dummies : scipy.sparse.csr_matrix

	Notes
	-----
	The pandas variant of this function has a known bug
	where it does not actually produce sparse matrices.
	"""
	if missing_pattern is None:
		missing_pattern = np.zeros(len(data), dtype=bool)
	sparse_colnum = 0
	xinds = []
	yinds = []
	for column in data.columns:
		# Find unique values
		values = data[column].values
		unq_vals = np.unique(values)
		if drop_first: 
			unq_vals = unq_vals[0:-1]
		# Loop through and create sparse matrix
		for value in unq_vals:
			x = np.where((values == value) & (~missing_pattern))[0]
			xinds.append(x)
			yinds.append(sparse_colnum * np.ones(len(x), dtype=int))
			sparse_colnum += 1

	# Put it all together
	xinds = np.concatenate(xinds)
	yinds = np.concatenate(yinds)
	return scipy.sparse.csr_matrix(
		(np.ones(len(xinds)), (xinds, yinds)), shape=(len(data), sparse_colnum)
	)