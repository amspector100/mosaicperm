import time
import numpy as np
import math
import unittest
from unittest.mock import patch 
import pytest
import os
import sys
from scipy import stats
try:
	from . import context
	from .context import mosaicperm as mp
# For profiling
except ImportError:
	import context
	from context import mosaicperm as mp

def fast_maxcorr_stat(residuals):
	"""
	Computes max correlation. No zero-SD checks for speed.
	Adds a tiny bit of noise to break ties.
	"""
	return np.abs(
		np.corrcoef(residuals.T) - np.eye(residuals.shape[1])
	).max() + np.random.uniform(0, 0.001)

class TestOLSFactorResids(unittest.TestCase):
	"""tests ols residual helper"""
	def test_ols_resids_3d(self):
		np.random.seed(123)
		n_obs, n_subjects, n_factors = 30, 40, 5
		exposures = np.random.randn(n_obs, n_subjects, n_factors)
		outcomes = np.random.randn(n_obs, n_subjects)
		# residuals
		residuals = mp.factor.ols_residuals(outcomes, exposures)
		# test accuracy
		inds = np.random.choice(n_obs, size=10, replace=False)
		for i in inds:
			X = exposures[i] # n_subjects x n_factors
			hatbeta_i = np.linalg.inv(X.T @ X) @ X.T @ outcomes[i]
			expected = outcomes[i] - X @ hatbeta_i
			np.testing.assert_array_almost_equal(
				residuals[i], expected,
				decimal=5,
				err_msg=f"OLS factor residuals are inaccurate with 3d exposures"
			)

	def test_ols_resids_2d(self):
		n_obs, n_subjects, n_factors = 30, 40, 5
		exposures = np.random.randn(n_subjects, n_factors)
		outcomes = np.random.randn(n_obs, n_subjects)
		# residuals
		residuals = mp.factor.ols_residuals(outcomes, exposures)
		# test accuracy
		H = exposures @ np.linalg.inv(exposures.T @ exposures) @ exposures.T
		H = np.eye(n_subjects) - H
		expected = outcomes @ H
		np.testing.assert_array_almost_equal(
			residuals, expected, decimal=5, 
			err_msg=f"OLS factor residuals are inaccurate with 2d exposures"
		)

class TestMosaicFactorTest(context.MosaicTest):
	"""
	tests mosaic factor test class
	"""

	def test_residual_orthogonality(self):
		n_obs, n_subjects, n_factors = 150, 200, 20
		for nstart in [0, 5, 15, 50]:
			# Create exposure data
			if nstart > 0:
				exposures = context._create_exposures(
					n_obs=n_obs, n_subjects=n_subjects, n_factors=n_factors, nstart=nstart,
				)
			else:
				exposures = np.random.randn(n_subjects, n_factors)
			outcomes = np.random.randn(n_obs, n_subjects)
			# Construct residuals
			mpt = mp.factor.MosaicFactorTest(
				outcomes=outcomes, exposures=exposures, test_stat=None
			)
			mpt.compute_mosaic_residuals()
			mpt.permute_residuals()
			# Check orthogonality
			for resids, name in zip([mpt.residuals, mpt._rtilde], ['Residuals', 'Permuted residuals']):
				for (batch, group) in mpt.tiles:
					for j in batch:
						if nstart > 0:
							tilej_exposures = exposures[j, group]
						else:
							tilej_exposures = exposures[group]
						np.testing.assert_array_almost_equal(
							tilej_exposures.T @ resids[j, group],
							np.zeros(n_factors),
							decimal=5,
							err_msg=f"{name} for j={j}, group={group} are not orthogonal to exposures with nstart={nstart}."
						)

	def test_reduction_to_ols(self):
		"""
		Tests that with one tile and constant exposures, the residuals are OLS residuals.
		"""
		n_obs, n_subjects, n_factors = 150, 200, 20
		exposures = np.random.randn(n_subjects, n_factors)
		outcomes = np.random.randn(n_obs, n_subjects)
		tiles = [(np.arange(n_obs), np.arange(n_subjects))]
		## compute mosaic residuals with one large tile
		mpt = mp.factor.MosaicFactorTest(
			outcomes=outcomes, exposures=exposures, test_stat=None, tiles=tiles
		)
		mpt.compute_mosaic_residuals()
		## Compute OLS residuals
		# Use inversion instead of QR just to add variety
		H = exposures @ np.linalg.inv(exposures.T @ exposures) @ exposures.T
		H = np.eye(n_subjects) - H
		ols_resid = outcomes @ H
		np.testing.assert_array_almost_equal(
			mpt.residuals, ols_resid, decimal=5, err_msg=f"Mosaic residuals do not reduce to OLS resids with one tile"
		)

	def test_permute_residuals(self):
		# data
		np.random.seed(123)
		n_obs, n_subjects, n_factors = 150, 200, 20
		exposures = context._create_exposures(
			n_obs=n_obs, n_subjects=n_subjects, n_factors=n_factors, nstart=int(n_obs / 5),
		)
		outcomes = np.random.randn(n_obs, n_subjects)
		# Construct and permute residuals
		mpt = mp.factor.MosaicFactorTest(outcomes=outcomes, exposures=exposures, test_stat=None)
		mpt.compute_mosaic_residuals()
		for method in ['ix', 'argsort']:
			mpt.permute_residuals(method=method)
			# check equality within tiles
			for tilenum, (batch, group) in enumerate(mpt.tiles):
				# Same elements in different order
				np.testing.assert_array_almost_equal(
					np.sort(np.unique(mpt.residuals[np.ix_(batch, group)])), 
					np.sort(np.unique(mpt._rtilde[np.ix_(batch, group)])),
					decimal=5,
					err_msg=f"for tilenum={tilenum}, mpt residuals and _rtilde do not have the same elements in different orders"
				)
				# The tiles have the same rows but in different orders
				tile = mpt.residuals[np.ix_(batch, group)]
				tile_perm = mpt._rtilde[np.ix_(batch, group)]
				intersection = np.unique(np.concatenate([tile, tile_perm], axis=0), axis=0)
				self.assertTrue(
					intersection.shape == tile.shape,
					f"For tile={tilenum}, the permuted tile does not have the same rows as the original tile"
				)

			# Make sure globally the residuals are accurate
			self.assertTrue(
				np.any(mpt.residuals != mpt._rtilde),
				"After permuting, _rtilde=residuals everywhere."
			)

	def test_permute_residuals_joint_law(self):
		## Check that permuted tile orderings are uniformly random
		reps = 2000
		n_obs, n_subjects, n_factors = 5, 20, 2
		exposures = np.random.randn(n_subjects, n_factors)
		outcomes = np.random.randn(n_obs, n_subjects)
		mpt = mp.factor.MosaicFactorTest(
			outcomes=outcomes, exposures=exposures, test_stat=None, max_batchsize=3
		)
		mpt.compute_mosaic_residuals()
		for method in ['ix', 'argsort']:
			# the tiles we will look at
			tilenums = [0, 1, -1]
			all_permcounts = [np.zeros((reps, len(mpt.tiles[k][0]))) for k in tilenums]
			tile_resids = [mpt.residuals[np.ix_(*mpt.tiles[k])] for k in tilenums]
			for r in range(reps):
				mpt.permute_residuals(method=method)
				for k in tilenums:
					tilde_resids = mpt._rtilde[np.ix_(*mpt.tiles[k])]
					for ell in range(len(tilde_resids)):
						all_permcounts[k][r, ell] = np.where(
							np.all(tilde_resids[ell] == tile_resids[k], axis=1)
						)[0].item()
			# check the permutation distribution is uniform
			for k in tilenums:
				batchlen = all_permcounts[k].shape[1]
				# unique but not bijective representation of the permutations
				permcounts = all_permcounts[k] @ (batchlen**(np.arange(batchlen)))
				permcounts = np.bincount(permcounts.astype(int))
				permcounts = permcounts[permcounts > 0] # get rid of superfluous reps 
				# assert correct number of nonzero values
				self.assertTrue(
					permcounts.shape[0]==math.factorial(batchlen),
					f"For batchlen={batchlen}, tile={k}, permcounts has shape {permcounts.shape} <= batchlen factorial"
				)
				# check uniform dist
				self.pearson_chi_square_test(
					counts=permcounts, test_name='Marginal law of permutations is uniform'
				)
			# check the permutation distribution is independent
			batchlen0 = all_permcounts[0].shape[1]
			for k in tilenums[1:]:
				permcounts = all_permcounts[0][:, 0] + all_permcounts[k][:, 0] * batchlen0
				permcounts = np.bincount(permcounts.astype(int))
				self.pearson_chi_square_test(
					counts=permcounts, test_name='Law of permutations is independent'
				)
				
	def test_binary_pval(self):
		"""
		For maximum efficiency, uses small scale data to test whether
		the p-value with one randomization is valid.
		"""
		# Very small scale data
		np.random.seed(123)
		reps = 1000
		n_obs, n_subjects, n_factors = 10, 15, 2
		# Factors and exposures
		exposures = stats.laplace.rvs(size=(n_subjects, n_factors))
		factors = stats.laplace.rvs(size=(n_obs, n_factors))
		mu = factors @ exposures.T
		# missing data pattern
		missing_flags = np.random.randn(n_obs, n_subjects)
		missing_flags = missing_flags < -1.25 

		# Including simulation setting with data missing at random
		for missing_data in [True, False]:
			pvals = np.zeros(reps)
			for r in range(reps):
				outcomes = stats.laplace.rvs(size=(n_obs, n_subjects)) + mu
				if missing_data:
					outcomes[missing_flags] = np.nan

				# construct
				mpt = mp.factor.MosaicFactorTest(
					outcomes=outcomes,
					exposures=exposures,
					test_stat=fast_maxcorr_stat,
				) 
				mpt.fit(nrand=1, verbose=False)
				pvals[r] = mpt.pval

			self.check_binary_pval_symmetric(pvals, alpha=0.001)

	def test_nan_handling_2d(self):
		# Create data
		np.random.randn(123)
		n_obs, n_subjects, n_factors, ngroups = 15, 20, 3, 2
		exposures = np.random.randn(n_subjects, n_factors)
		outcomes = np.random.randn(n_obs, n_subjects)
		# Create one missing outcome
		outcomes[1, 0] = np.nan
		# initialize class
		mpt = mp.factor.MosaicFactorTest(
			outcomes=outcomes,
			exposures=exposures,
			test_stat=fast_maxcorr_stat,
			ngroups=ngroups,
			max_batchsize=n_obs+1,
		)
		# Check that the tilings are appropriate
		self.assertTrue(
			len(mpt.tiles) == 4,
			f"mpt should have 4 tiles due to missing data pattern but has {len(mpt.tiles)}"
		)
		self.assertTrue(
			set(mpt.tiles[0][0].tolist()) == set([0,1]),
			f"first batch is {mpt.tiles[0][0]}, expected [0,1] due to nan pattern"
		)
		self.assertTrue(
			set(mpt.tiles[ngroups][0].tolist()) == set(range(2, n_obs)),
			f"second batch is {mpt.tiles[ngroups][0]}, expected {set(range(2, n_obs))} due to nan pattern"
		)

	def test_nan_handling_3d(self):
		# Create data
		np.random.randn(123)
		n_obs, n_subjects, n_factors = 15, 20, 3
		exposures = np.random.randn(n_obs, n_subjects, n_factors)
		outcomes = np.random.randn(n_obs, n_subjects)
		outcomes[outcomes < -1.25] = np.nan
		missing_pattern = np.isnan(outcomes)
		# create residuals
		mpt = mp.factor.MosaicFactorTest(
			outcomes=outcomes, exposures=exposures, test_stat=None
		)
		mpt.compute_mosaic_residuals()
		# Check that preprocessing worked
		np.testing.assert_array_almost_equal(
			mpt.exposures[np.isnan(outcomes)].flatten(),
			0,
			decimal=5,
			err_msg=f"Mosaic processed exposures are nonzero even for missing outcomes"
		)

		# check orthogonality
		for i in range(n_obs):
			subjects = np.where(~np.isnan(outcomes[i]))[0]
			np.testing.assert_array_almost_equal(
				exposures[i, subjects].T @ mpt.residuals[i, subjects],
				np.zeros(n_factors),
				decimal=5,
				err_msg=f"Mosaic with 3D exposures and missing outcomes does not enforce orthogonality"
			)

		# Check that local exchangeability is preserved
		for (batch, group) in mpt.tiles:
			for j in group:
				zero_prop = np.mean(mpt.outcomes[batch, j] == 0)
				self.assertTrue(
					zero_prop in [0.0,1.0],
					f"For asset={j}, batch={batch}, outcomes={mpt.outcomes[batch, j]} has a mix of zeros and non zeros"
				)
				expected = float(np.any(missing_pattern[batch, j] == 1))
				self.assertTrue(
					zero_prop == expected,
					f"For asset={j}, batch={batch}, zero_prop={zero_prop} should equal {expected} based on missing pattern."
				)

	def test_nans_constant_within_tiles(self):
		# Repeat with the missing pattern by patch
		n_obs, n_subjects, n_factors = 20, 40, 3
		exposures = np.random.randn(n_obs, n_subjects, n_factors)
		outcomes = np.random.randn(n_obs, n_subjects)
		# Create tiling
		batches = [np.arange(10), np.arange(10, n_obs)]
		tiles = []
		for batch in batches:
			groups = mp.tilings.even_random_partition(n_subjects, 2, shuffle=True)
			tiles.extend([(batch, group) for group in groups])
		# Constant missing pattern within tiles
		for batch in batches:
			missing = np.random.choice(n_subjects, size=int(n_subjects / 4), replace=False)
			outcomes[np.ix_(batch, missing)] = np.nan
		# create residuals
		mpt = mp.factor.MosaicFactorTest(
			outcomes=outcomes, exposures=exposures, test_stat=None, tiles=tiles
		)
		mpt.compute_mosaic_residuals()
		# check residuals has the same zero pattern as outcomes
		# this only holds when the missing pattern is constant within tiles
		np.testing.assert_array_almost_equal(
			mpt.residuals[np.isnan(outcomes)].flatten(),
			np.zeros(np.sum(np.isnan(outcomes))),
			decimal=5,
			err_msg=f"There exist nonzero Mosaic residuals whose corresponding outcomes are missing"
		)

	def test_maxcorr_stats_no_error(self):
		# Data
		n_obs, n_subjects, n_factors = 25, 20, 3
		exposures = np.random.randn(n_subjects, n_factors)
		outcomes = np.random.randn(n_obs, n_subjects)
		# Create two columns which are entirely nan
		outcomes[:, 0] = np.nan
		outcomes[:, -1] = np.nan
		outcomes[:, 3] = 1
		# test maxcorr stats
		for test_stat in [
			mp.statistics.mean_maxcorr_stat,
			mp.statistics.quantile_maxcorr_stat
		]:
			mpt = mp.factor.MosaicFactorTest(
				outcomes=outcomes, exposures=exposures, test_stat=test_stat
			)
			mpt.fit(nrand=2, verbose=False)
			self.assertFalse(
				np.any(np.isnan(mpt.null_statistics)),
				f"mpt.null_statistics contains nans"
			)
			self.assertFalse(
				np.any(np.isnan(mpt.statistic)),
				f"mpt.statistics contains nans"
			)
			# Fit time series variant and make sure no errors
			mpt.fit_tseries(nrand=2, n_timepoints=3)
			mpt.plot_tseries(show_plot=False)

	@patch("matplotlib.pyplot.show")
	def test_plots_do_not_error(self, mock_show):
		# Small-scale data
		n_obs, n_subjects, n_factors = 100, 50, 2
		exposures = np.random.randn(n_obs, n_subjects, n_factors)
		outcomes = np.random.randn(n_obs, n_subjects)
		# Fit for two test statistics
		stats = [mp.statistics.mean_maxcorr_stat, mp.statistics.quantile_maxcorr_stat]
		for stat in stats:
			
			mptest = mp.factor.MosaicFactorTest(
				outcomes=outcomes, 
				exposures=exposures,
				test_stat=stat,
			)
			mptest.fit()
			mptest.summary_plot()
			mptest.summary_plot(figsize=(10,10))
			mptest.fit_tseries()
			mptest.plot_tseries()
			fig, axes = mptest.plot_tseries(figsize=(10,10), show_plot=False)

	def test_handling_zero_factors(self):

		# Data
		n_obs, n_subjects, n_factors = 25, 20, 3
		exposures = np.random.randn(n_obs, n_subjects, n_factors)
		outcomes = np.random.randn(n_obs, n_subjects)
		# new exposures
		new_exposures = np.concatenate(
			[exposures, np.zeros((n_obs, n_subjects, 5*n_factors))], axis=-1
		)
		# Test that the results are the same
		test_stat = mp.statistics.mean_maxcorr_stat
		mpt = mp.factor.MosaicFactorTest(
			outcomes=outcomes, exposures=exposures, test_stat=test_stat
		).fit(nrand=2)
		mpt_new = mp.factor.MosaicFactorTest(
			outcomes=outcomes, exposures=new_exposures, test_stat=test_stat
		).fit(nrand=2)
		np.testing.assert_array_almost_equal(
			mpt.residuals,
			mpt_new.residuals,
			decimal=5,
			err_msg=f"Adding all zero factors changes residuals"
		)

	@patch("matplotlib.pyplot.show")
	def test_plots_do_not_error(self, mock_show):
		# Small-scale data
		n_obs, n_subjects, n_factors = 100, 50, 2
		exposures = np.random.randn(n_obs, n_subjects, n_factors)
		outcomes = np.random.randn(n_obs, n_subjects)
		# Fit for two test statistics
		stats = [mp.statistics.mean_maxcorr_stat, mp.statistics.quantile_maxcorr_stat]
		for stat in stats:
			mptest = mp.factor.MosaicFactorTest(
				outcomes=outcomes, 
				exposures=exposures,
				test_stat=stat,
			)
			mptest.fit(nrand=50)
			mptest.summary_plot()
			mptest.summary_plot(figsize=(10,10))
			mptest.fit_tseries(n_timepoints=10, nrand=50)
			mptest.plot_tseries()
			fig, axes = mptest.plot_tseries(figsize=(10,10), show_plot=False)

class TestMosaicBCV(context.MosaicTest):

	def test_agrees_with_manual_version(self):
		# Data
		np.random.seed(123)
		n_obs, n_subjects, n_factors = 50, 50, 5
		exposures = np.random.randn(n_obs, n_subjects, n_factors)
		outcomes = np.random.randn(n_obs, n_subjects) + np.random.randn(n_obs, 1) / 10
		# estimate candidate exposures on the first half of the data
		n0 = int(0.5 * n_obs)
		resid0 = mp.factor.ols_residuals(outcomes[0:n0], exposures[0:n0])
		new_exposures = mp.statistics.approximate_sparse_pcas(np.corrcoef(resid0.T))
		# fit MosaicBCV
		mpt_bcv = mp.factor.MosaicBCV(
			outcomes=outcomes[n0:], exposures=exposures[n0:], new_exposures=new_exposures,
		)
		mpt_bcv.fit(nrand=2).summary()
		# fit manual variant
		tiles = mpt_bcv.tiles
		mpt = mp.factor.MosaicFactorTest(
			outcomes=outcomes[n0:],
			exposures=exposures[n0:],
			tiles=tiles,
			test_stat=mp.statistics.adaptive_mosaic_bcv_stat,
			test_stat_kwargs={"tiles":tiles, "new_exposures":new_exposures}
		)
		mpt.fit(nrand=3).summary()
		# test that themarginal statistics are the same
		np.testing.assert_array_almost_equal(
			mpt.statistic, mpt_bcv.statistic,
			decimal=5,
			err_msg=f"MosaicBCV produces unexpected test statistic values"
		)
		# test for errors in tseries variant
		mpt_bcv.fit_tseries(nrand=3, n_timepoints=3)
		mpt_bcv.plot_tseries(show_plot=False)

if __name__ == "__main__":
	# Run tests---useful if using cprofilev
	basename = os.path.basename(os.path.abspath(__file__))
	if sys.argv[0] == f'test/{basename}':
		time0 = time.time()
		context.run_all_tests([TestMosaicBCV(), TestMosaicFactorTest(), TestOLSFactorResids()])
		elapsed = np.around(time.time() - time0, 2)
		print(f"Finished running all tests at time={elapsed}")

	# Else let unittest handle this for us
	else:
		unittest.main()