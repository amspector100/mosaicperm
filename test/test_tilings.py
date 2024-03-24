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
	tests tiling functions
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

	def test_factor_tiles_2d(self):
		exposures = np.random.randn(10, 2)
		tiles = mp.tilings.default_factor_tiles(exposures, n_obs=20, max_batchsize=8)
		mp.tilings.check_valid_tiling(tiles)