import numpy as np
import itertools
from typing import Optional, Union

def check_valid_tiling(tiles: list) -> None:
	"""
	Checks validity of a tiling.

	Parameters
	----------
	tiles : list
		``tiles[i]`` is a tuple of two np.arrays of indices defining tile ``i``.
		I.e, tile ``i`` of an array is ``data[tile[i][0]][:, tile[i][1]]``.

	Raises
	------
		ValueError: if ``tiles`` is an invalid tiling.
	"""
	## Check disjointness and support
	n_obs = int(np.max([np.max(tile[0]) for tile in tiles]) + 1)
	n_subjects = int(np.max([np.max(tile[1]) for tile in tiles]) + 1)
	counts = np.zeros((n_obs, n_subjects))
	for m, (batch, group) in enumerate(tiles):
		if np.any(counts[np.ix_(batch, group)] != 0):
			raise ValueError(f"Tile {m} is not disjoint from tiles 0-{m-1}.")
		counts[np.ix_(batch, group)] += 1

	## Check support
	if not np.all(counts == 1):
		raise ValueError(f"Tiles are disjoint but do not partition [n_obs] x [n_subjects]")

class Tiling(list):
	"""
	A class to store tilings used in mosaic tests.

	Parameters
	----------
	tiles : list
		List of tuples of two integer np.arrays. E.g., tile 
		``i`` of an array is ``data[tile[i][0]][:, tile[i][1]]``.
	check_valid : bool
		If True, runs :func:`check_valid_tiling` during initialization.

	Raises
	------
	ValueError : If tiles does not define a partition of the cartesian
		product of ``{0, ..., n}`` x ``{0, ..., k}`` for some integers
		``n`` and ``k``. 

	Examples
	--------

	>>> import numpy as np
	>>> import mosaicperm as mp
	>>> 
	>>> # a valid tiling of {0, 1} x {0, 1, 2, 3}
	>>> tiles0 = [
	... 	(np.array([0]), np.array([0, 2])),
	... 	(np.array([0]), np.array([1, 3])),
	... 	(np.array([1]), np.array([0, 1])),
	... 	(np.array([1]), np.array([2, 3])),
	... ]
	>>> tiling = mp.tilings.Tiling(tiles0, check_valid=True)

	>>> # an invalid tiling due to repeats
	>>> tiles1 = [
	... 	(np.array([0, 2]), np.array([0, 1])),
	... 	(np.array([1]), np.array([0, 1])),
	... 	(np.array([0]), np.array([1])),
	... ]
	>>> tiling = mp.tilings.Tiling(tiles1, check_valid=True)
	Traceback (most recent call last):
		...
	ValueError: Tile 2 is not disjoint from tiles 0-1.


	Notes
	-----
	This class thinly wraps python ``list``. 
	"""

	def __init__(self, tiles: list, check_valid: bool=False):
		self.tiles = tiles
		if check_valid:
			check_valid_tiling(tiles)
		super().__init__(tiles)
		
	def __str__(self, *args, **kwargs):
		return "Tiling" + super().__str__(*args, **kwargs)

	def save(self, filename):
		"""
		Saves the tiling to ``filename.npy``.
		"""
		# The save format is an integer numpy array.
		# 1. The first element is an integer ``nbreaks``
		# 2. The next ``nbreaks`` elements specify the breaks of the (batch, group)
		# 3. The rest of the elements are a concatenation of the tiling
		# Form concatenation and count breaks
		comb = []
		breaks = [0]
		counter = 0
		for batch, group in self.tiles:
			for obj in [batch, group]:
				comb.append(obj)
				counter += len(obj)
				breaks.append(counter)
		# Concatenate
		comb = np.concatenate(comb)
		breaks = np.array(breaks)
		tosave = np.concatenate([np.array([len(breaks)]), breaks, comb]).astype(np.int64)
		np.save(filename, tosave)
		return None

	@classmethod
	def load(cls, filename, **kwargs):
		"""
		Loads a tiling from a .npy file. 

		Parameters
		----------
		filename : str
			File where the tiling is stored.
		kwargs : dict
			Other kwargs for ``__init__``.

		Notes
		-----
		The .npy file must have been generated using 
		:meth:`save` or this method will not work.
		"""
		raw = np.load(filename)
		# 1. The length of the array of breaks
		breaksize = raw[0]
		# 2. The breaks
		breaks = raw[1:(breaksize+1)]
		objsraw = raw[(breaksize+1):]
		# 3. Loop through and load objects
		objs = []
		for j in range(breaksize - 1):
			objs.append(objsraw[breaks[j]:breaks[j+1]])
		# 4. Pair the appropriate batches/groups
		tiles = []
		for jj in range(int(len(objs) / 2)):
			tiles.append((objs[2*jj], objs[2*jj+1]))
		return cls(tiles, **kwargs)

def even_random_partition(n, k, shuffle=True):
	"""
	Partitions {0, ..., n-1} into k subsets.

	Parameters
	----------
	n : int
	k : int
	shuffle : bool
		If True, shuffles before partitioning 
		to produce a uniformly random partition.
	"""
	if k > n:
		raise ValueError(f"k={k} > n={n}")
	groups = np.sort(np.arange(n) % k)
	if shuffle:
		np.random.shuffle(groups)
	return [
		np.where(groups == ell)[0] for ell in range(k)
	]

def _preprocess_partition(partition):
	"""
	Preprocesses a partition so that its groups are numbered
	0 through k, for some k.
	"""
	vals = np.unique(partition)
	output = np.zeros(len(partition), dtype=int)
	for i, v in enumerate(vals):
		output[partition == v] = i
	return output


def coarsify_partition(
	partition: np.array, 
	k: int, 
	minsize: int=0, 
	random: bool=True,
):
	"""
	Produces a random coarsening of ``partition``.

	Parameters
	----------
	partition : array
		n-length integer array, so that ``partition[i] = l``
		implies that item i is in group l of the partition. 
	k : int
		Desired number of elements of the coarsened partition.
	minsize : int
		Minimum number of elements in each set in the coarsened partition.
	random : bool
		If true, produces a randomized coarsening. Otherwise returns
		a deterministic result.

	Returns
	-------
	partition : array
		n-length integer array containing values 0 through k-1, 
		so that ``partition[i] = l`` implies that item i is 
		in group l of the partition. 
	"""
	# note: this copies partition so the results are never in-place
	partition = _preprocess_partition(partition)
	# number of groups to start
	kstart = np.max(partition) + 1
	inds = np.arange(kstart)
	# precompute sizes of each partition
	sizes = np.array([np.sum(partition==k) for k in np.unique(partition)]).astype(float)
	sizes[sizes == 0] = np.inf
	for _ in range(kstart-1):
		# Compute sizes
		# sizes = np.array([np.sum(partition==k) for k in range(kstart)]).astype(float)
		# sizes[sizes == 0] = np.inf
		# Check stopping condition
		if np.all(sizes >= minsize) and len(sizes[sizes < np.inf]) <= k:
			break
		
		## Random variant
		if random:
			# Sampling probs
			probs = 1 / sizes
			# three cases based on the number of groups below minsize
			n_too_small = np.sum(sizes < minsize)
			if n_too_small == 1:
				# Case 1: merge the smallest with another
				i0 = np.where(sizes < minsize)[0].item()
				probs[i0] = 0; probs /= probs.sum()
				i1 = np.random.choice(inds, p=probs, size=1).item()
			else:
				# Case 2: only merge groups below minsize
				if n_too_small > 1:
					probs[sizes >= minsize] = 0
				# Case 3: probs are undajusted (no groups below minsize)
				probs /= probs.sum()
				i0, i1 = np.random.choice(inds, p=probs, size=2, replace=False)
		else:
			i0, i1 = np.argpartition(sizes, 2)[:2]
		
		# Sample and merge
		sizes[i0] += np.sum(partition == i1)
		sizes[i1] = np.inf
		partition[partition == i1] = i0

	return _preprocess_partition(partition)

def random_tiles(
	n_obs: int,
	n_subjects: int,
	nbatches: int,
	ngroups: int,
	seed: int=123,
) -> Tiling:
	"""
	Partitions outcomes into ``nbatches`` x ``ngroups`` random tiles.

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
	seed : int
		Random seed.

	Returns
	-------
	tiles : mosaicperm.tilings.Tiling
		The default tiling as a :class:`.Tiling` object.
	"""
	np.random.seed(seed)
	# partition along timepoints
	batches = even_random_partition(n=n_obs, k=nbatches, shuffle=False)
	# partition along subjects
	tiles = []
	for batch in batches:
		groups = even_random_partition(n=n_subjects, k=ngroups, shuffle=True)
		tiles.extend([(batch, group) for group in groups])
	# return tiles
	return Tiling(tiles)

def default_factor_tiles(
	exposures: np.array,
	n_obs: Optional[int]=None,
	max_batchsize: Optional[int]=10,
	ngroups: Optional[int]=None,
	clusters: Optional[np.array]=None,
	seed: int=123,
) -> Tiling:
	"""
	Computes default tiling for factor models.

	Parameters
	----------
	exposures : np.array
		(``n_obs``, ``n_subjects``, ``n_factors``) array of factor exposures
		OR 
		(``n_subjects``, ``n_factors``) array of factor exposures if
		the exposures do not change with time.	
	n_obs : int
		Number of timepoints. Optional unless exposures is 2D.
	max_batchsize : int
		Maximum length (in time) of a tile.
	ngroups : int
		Number of groups to partition the subjects into.
		Default is the max integer such that each group contains
		5 times as many subjects as there are factors.
	clusters : np.array
		An ``n_subjects``-length array of cluster_ids, so 
		``clusters[i] = k`` implies subject i is in the kth cluster.
		If clusters is provided, the output will preserve the cluster
		structure.
	seed: int
		Random seed.

	Returns
	-------
	tiles : mosaicperm.tilings.Tiling
		The default tiling as a :class:`.Tiling` object.
	"""
	np.random.seed(seed)
	# Choose batches
	if len(exposures.shape) == 3:
		n_obs, n_subjects, n_factors = exposures.shape
		batches = _exposures_to_batches(exposures, max_batchsize=max_batchsize)
	elif len(exposures.shape) == 2:
		# Defaults for dimensionality
		if n_obs is None:
			raise ValueError(f"n_obs must be provided if exposures is a 2D array.")
		n_subjects, n_factors = exposures.shape
		nbatches = int(np.ceil(n_obs / max_batchsize))
		# Create batches
		batches = even_random_partition(n=n_obs, k=nbatches, shuffle=False)
	else:
		raise ValueError(f"exposures should be a 2D or 3D array but has shape {exposures.shape}")

	# Partition subjects and construct tiles
	if ngroups is None:
		ngroups = max(2, int(np.ceil(n_subjects / (5*n_factors))))

	tiles = []
	for batch in batches:
		# Create groups
		if clusters is None:
			groups = even_random_partition(n=n_subjects, k=ngroups, shuffle=True)
		else:
			coarsened = coarsify_partition(clusters, k=ngroups, minsize=2*n_factors)
			groups = [np.where(coarsened == k)[0] for k in np.unique(coarsened)]
		# Add to tiles
		tiles.extend([(batch.astype(int), group.astype(int)) for group in groups])
	return Tiling(tiles)


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
			# Case 1(b): next length is > 2, so we take one from it
			elif lengths[i+1] > 2:
				batches.append(np.arange(starts[i], ends[i]+1))
				starts[i+1] += 1
				i += 1
			# Case 1(c): next length is == 2 (we give up and use a batchsize of 1)
			else:
				batches.append(np.array([starts[i]]))
				i += 1

		# Case 2: naive batchsize is too large
		elif lengths[i] > max_batchsize:
			nsplit = int(np.ceil(lengths[i] / max_batchsize))
			new_starts = np.around(np.linspace(starts[i], ends[i], nsplit+1))
			for j in range(nsplit):
				batches.append(np.arange(new_starts[j], new_starts[j+1]))
			i += 1

		# Case 3: naive batchsize is the right size
		else:
			batches.append(np.arange(starts[i], ends[i]))
			i += 1

	return [batch.astype(int) for batch in batches]


if __name__ == "__main__":
	import doctest
	doctest.testmod()