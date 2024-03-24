import numpy as np
import itertools
from typing import Optional, Union

def even_random_partition(n, k, shuffle=True):
	"""
	Partitions {0, ..., n-1} into k subsets.

	Parameters
	----------
	shuffle : bool
		If True, shuffles before partitioning 
		to produce a uniformly random partition.
	"""
	inds = np.arange(n)
	if shuffle:
		np.random.shuffle(inds)
	groups = inds // (np.ceil(n / k))
	return [
		np.where(groups == ell)[0] for ell in range(k)
	]

def random_tiles(
	n_obs: int,
	n_subjects: int,
	nbatches: int,
	ngroups: int,
):
	"""
	Parameters
	----------
	n_obs : int
		Number of observations/timepoints.
	n_subjects : int
		Number of subjects/assets.
	nbatches : int
		Number of batches to split observations into.
	ngroups : int
		Number of groups to split subjects into.

	Returns
	-------
	tiles : list
		tiles[i] is a tuple of two integer np.arrays of indices 
		defining the ith tile. I.e, the mth tile of an array ``data`` 
		is ``data[tile[i][0]][:, tile[i][1]]``.
	"""
	# partition along timepoints
	batches = even_random_partition(n=n_obs, k=nbatches, shuffle=False)
	# partition along subjects
	tiles = []
	for batch in batches:
		groups = even_random_partition(n=n_subjects, k=ngroups, shuffle=True)
		tiles.append((batch, groups))
	# return tiles
	return tiles



def default_factor_tiles(
	exposures: np.array,
	n_obs: Optional[int]=None,
	max_batchsize: Optional[int]=10,
):
	"""
	Computes default tiling for factor models.

	Parameters
	----------
	exposures : np.array
		Array of factor exposures. Two input options;
		- (n_obs, n_subects, n_factors)-shaped array (if the exposures change with time)
		- (n_subects, n_factors)-shaped array (if the exposures do not change with time)
	n_obs : int
		Number of timepoints. Optional unless exposures is 2D
	max_batchsize : int
		Maximum length (in time) of a tile.
	"""
	# Choose batches
	if len(exposures.shape) == 3:
		n_obs, n_subjects, n_factors = exposures.shape
		batches = _exposures_to_batches(exposures, max_batchsize=max_batchsize)
	elif len(exposures.shape) == 2:
		# Defaults for dimensionality
		if n_obs is None:
			raise ValueError(f"n_obs must be provided if exposures is a 2D array.")
		n_subjects, n_factors = exposures.shape
		nbatches = np.ceil(n_obs / max_batchsize)
		# Create batches
		batches = even_random_partition(n=n_obs, k=nbatches, shuffle=False)
	else:
		raise ValueError(f"exposures should be a 2D or 3D array but has shape {exposures.shape}")

	# Partition subjects and construct tiles
	ngroups = max(2, int(np.ceil(n_subjects / (5*n_factors))))
	tiles = []
	for batch in batches:
		groups = even_random_partition(n=n_subjects, k=ngroups, shuffle=True)
		tiles.append((batch, groups))
	return tiles


def _exposures_to_batches(exposures, max_batchsize=10):
	"""
	Parameters
	----------
	exposures : np.array
		Array of factor exposures of shape n_obs x n_subects x n_factors.
	max_batchsize : int
		Maximum batchsize

	Returns
	-------
	batches : list of np.arrays
		A list of batches which will define the tiles.
	"""

	# starts and ends of naive batches
	T = exposures.shape[0]
	changes = ~np.all(exposures[0:-1] == exposures[1:], axis=-1).all(axis=-1)
	ends = np.concatenate([np.where(changes)[0] + 1, [T]])
	starts = np.concatenate([[0], ends[:-1]])
	lengths = ends - starts
	nstarts = len(starts)
	# Loop through and create batches
	batches = []
	i = 0
	while i < nstarts:
		# Case 1: naive batchsize is 1
		if i < nstarts - 1 and lengths[i] == 1:
			# Case 1(a): next length is also 1
			if lengths[i+1] == 1:
				batches.append(np.arange(starts[i], ends[i+1]))
				i += 2
			# Case 1(b): next length is > 2
			elif lengths[i+1] > 2:
				batches.append(np.arange(starts[i], starts[i]+2))
				batches.append(np.arange(starts[i+1]+1, ends[i+1]))
				i += 2
			# Case 1(c): next length is == 2 (we give up and use a batchsize of 1)
			else:
				batches.append(np.array([starts[i]]))
				i += 1
			continue

		# Case 2: naive batchsize is too large
		if lengths[i] > max_batchsize:
			nsplit = int(np.ceil(lengths[i] / max_batchsize))
			new_starts = np.around(np.linspace(starts[i], ends[i], nsplit+1))
			for j in range(nsplit):
				batches.append(np.arange(new_starts[j], new_starts[j+1]))

		# Case 3: naive batchsize is the right size
		batches.append(np.arange(starts[i], ends[i]))
		i += 1

	return batches