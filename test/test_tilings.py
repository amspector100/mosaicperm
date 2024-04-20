import time
import numpy as np
import unittest
import pytest
import os
import sys
try:
	from . import context
	from .context import mosaicperm as mp
# For profiling
except ImportError:
	import context
	from context import mosaicperm as mp

class TestTiles(unittest.TestCase):
	"""
	tests generic tiling functions
	"""
	def test_random_tiles(self):
		np.random.seed(123)
		for _ in range(10):
			# Generate random tiles
			tile_args = dict(
				n_obs=np.random.randint(low=20, high=30),
				n_subjects=np.random.randint(low=40, high=80),
				nbatches=np.random.randint(low=1, high=5),
				ngroups=np.random.randint(low=2, high=5),
			)
			tiles = mp.tilings.random_tiles(**tile_args)
			# Check validity
			mp.tilings.check_valid_tiling(tiles)
			# check correct number of groups
			batch0 = set(tiles[0][0].tolist())
			subtiles = [tile for tile in tiles if set(tile[0].tolist()) == batch0]
			ngroups = len(subtiles)
			self.assertTrue(
				ngroups == tile_args['ngroups'],
				f"tiles has {ngroups} groups per batch despite ngroups={tile_args['ngroups']}"
			)
			# check correct number of batches
			nbatches = int(len(tiles) / ngroups)
			self.assertTrue(
				nbatches == tile_args['nbatches'],
				f"tiles has {nbatches} batches despite nbatches={tile_args['nbatches']}"
			)

	def test_factor_tiles_3d(self):
		np.random.seed(123)
		n_obs, n_subjects, n_factors = 1000, 10, 2
		for nstart in [10, 300, 500, 800]:
			## Create data for large scale setting
			exposures = context._create_exposures(
				n_obs=n_obs, n_subjects=n_subjects, n_factors=n_factors, nstart=nstart
			)

			# Create tiles and check validity
			max_batchsize = 10
			batches = mp.tilings._exposures_to_batches(exposures, max_batchsize=max_batchsize)
			tiles = mp.tilings.default_factor_tiles(exposures, max_batchsize=max_batchsize)
			mp.tilings.check_valid_tiling(tiles)

			## Check properties of batches
			# 1. max batchsize is enforced
			realized_max_batchsize = np.max([len(batch) for batch in batches])
			self.assertTrue(
				realized_max_batchsize <= max_batchsize,
				f"realized_max_batchsize={realized_max_batchsize} > max_batchsize={max_batchsize}"
			)
			# 2. all nonconstant exposure tiles have size 2
			for m, batch in enumerate(batches):
				if not np.all([np.all(exposures[j] == exposures[batch[0]]) for j in batch]):
					self.assertTrue(
						len(batch) == 2,
						f"batch m={batch} has non-constant exposures but length={len(batch)}"
					)
			# 3. All batches of size 1 have a batch of size 2 following them
			lengths = np.array([len(batch) for batch in batches])
			length_following_ones = lengths[np.where(lengths[:-1] == 1)[0] + 1]
			self.assertTrue(
				np.all(length_following_ones == 2),
				f"there exists a singleton batch followed by batch of length != 2"
			)

		# An example where the answer is clear
		starts = np.array([0, 2, 3, 5, 6, 9])
		ends = np.array([2, 3, 5, 6, 9, 20])
		exposures = context._create_exposures(
				starts=starts, ends=ends, n_obs=20, n_subjects=2, n_factors=1
		)
		batches = mp.tilings._exposures_to_batches(exposures, max_batchsize=max_batchsize)
		expected = [
			np.arange(2),
			np.array([2]),
			np.arange(3, 5),
			np.arange(5, 7),
			np.arange(7, 9),
			np.arange(9, 14),
			np.arange(14, 20),
		]
		self.assertTrue(
			len(expected) == len(batches), 
			f"For manual example, nbatches ({len(batches)}) != expected ({len(expected)})"
		)
		for j, eb, batch in zip(range(len(expected)), expected, batches):
			self.assertTrue(
				np.all(eb == batch),
				f"For manual example, batch j={j} equals {batch}, expected {eb}."
			)

	def test_save_tiles(self):
		SCRATCH_FILE = "tiling_scratch_file_temp.npy"

		np.random.seed(123)
		n_obs, n_subjects, n_factors = 100, 200, 2
		for nstart in [2, 10, 30]:
			# Create data
			exposures = context._create_exposures(
				n_obs=n_obs, n_subjects=n_subjects, n_factors=n_factors, nstart=nstart
			)

			# Create tiles and try to save/load
			tiles = mp.tilings.default_factor_tiles(exposures, max_batchsize=7)
			tiles.save(SCRATCH_FILE)
			tilesnew = mp.tilings.Tiling.load(SCRATCH_FILE)
			for k in range(len(tiles)):
				for i, name in zip([0, 1], ['batch', 'group']):
					np.testing.assert_array_almost_equal(
						tiles[k][i],
						tilesnew[k][i],
						decimal=5,
						err_msg=f"{name} k is not equal after saving/loading tiles",
					)

		# Clean up
		os.remove(SCRATCH_FILE)


	def test_factor_tiles_2d(self):
		exposures = np.random.randn(10, 2)
		tiles = mp.tilings.default_factor_tiles(exposures, n_obs=20, max_batchsize=8)
		mp.tilings.check_valid_tiling(tiles)

class TestClusteredTilings(unittest.TestCase):

	def test_preprocess_partition(self):
		n = 10
		for maxval in [2, 3, 5, 10, 20, 50, 1000, None]:
			if maxval is not None:
				x = np.random.randint(low=0, high=maxval, size=n)
			else:
				x = np.random.randn(n)
			# Test size
			y = mp.tilings._preprocess_partition(x)
			self.assertTrue(
				len(np.unique(y)) == len(np.unique(x)),
				f"_preprocess_partition does not preserve # of partitions with x={x}, y={y}"
			)
			# Test support
			np.testing.assert_array_almost_equal(
				np.sort(np.unique(y)),
				np.arange(len(np.unique(y))),
				decimal=5,
				err_msg=f"output of _preprocess_partition has support {np.unique(y)}"
			)
			# Test preserves group structure
			for k in np.unique(y):
				sub = x[y == k]
				self.assertTrue(
					np.all(sub == sub[0]),
					f"_preprocess_partition does not preserve structure: x[y=={k}] = {sub}"
				)

	def test_coarsify_partition(self):
		"""
		Tests that the random coarsening does correctly produces a coarsening.
		"""
		n = 200
		for maxval in [5, 10, 50, 10000]:
			for k in [2, 5, 10, 20]:
				x = np.random.randint(low=0, high=maxval, size=n)
				kstart = len(np.unique(x))	
				for minsize in [2, 5, 8]:
					for random in [False, True]:
						y = mp.tilings.coarsify_partition(
							x, k=k, minsize=minsize, random=True,
						)
						# Check the minsize condition
						sizes = np.array([np.sum(y == ell) for ell in np.unique(y)])
						self.assertTrue(
							np.min(sizes) >= minsize,
							f"Sizes {sizes} contains a size smaller than minsize={minsize}"
						)
						# Check that y is a coarsening of x
						for ell in np.unique(x):
							sub = y[x == ell]
							self.assertTrue(
								np.all(sub == sub[0]),
								f"y is not a subpartition of x with y[x == ell] = {sub}"
							)

	def test_nonrandom_coarsify_partition(self):
		"""
		Tests that the nonrandom coarsening is deterministic.
		"""
		for r in range(1, 10):
			np.random.seed(r)
			x = np.random.randint(low=0, high=10, size=200)
			kwargs = dict(partition=x, k=20, minsize=5, random=False)
			y0 = mp.tilings.coarsify_partition(**kwargs)
			y1 = mp.tilings.coarsify_partition(**kwargs)
			np.testing.assert_array_almost_equal(
				y0, 
				y1,
				decimal=5,
				err_msg=f"coarsify_partition produces random outputs with random=False"
			)

	def test_clustered_factor_tiles(self):
		"""
		Tests that default_factor_tiles() produces a good clustering
		"""
		np.random.seed(123)
		# Create outcomes/exposures
		n_obs, n_subjects, n_factors, nstart = 100, 200, 5, 20
		#outcomes = np.random.randn(n_obs, n_subjects)
		exposures = context._create_exposures(
			n_obs=n_obs, n_subjects=n_subjects, n_factors=n_factors, nstart=nstart
		)
		# Create clusters
		clusters = np.random.randint(0, 100, n_subjects)
		# create tiling
		tiles = mp.tilings.default_factor_tiles(
			exposures=exposures, n_obs=n_obs, clusters=clusters
		)
		for (batch, group) in tiles:
			# Test that each group is a union of clusters
			group_clust = np.unique(clusters[group])
			neg_group = np.array([i for i in range(n_subjects) if i not in group])
			self.assertTrue(
				np.sum([np.sum(clusters[neg_group] == k) for k in group_clust]) == 0,
				f"group={group} is not a union of clusters={clusters}"
			)
			# Test that each group has the right size
			self.assertTrue(
				len(group) >= 2*n_factors,
				f"group={group} has <= 2*n_factors elements"
			)
