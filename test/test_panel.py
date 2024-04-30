import time
import numpy as np
import math
import unittest
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

class TestOLSPanelResids(unittest.TestCase):
	"""tests ols residual helper"""
	def test_ols_resids(self):
		np.random.seed(123)
		n_obs, n_subjects, n_cov = 30, 40, 5
		covariates = np.random.randn(n_obs, n_subjects, n_cov)
		outcomes = np.random.randn(n_obs, n_subjects)
		# residuals
		residuals = mp.panel.ols_residuals_panel(outcomes, covariates)
		# test accuracy without assuming alignment is correct
		Xflat = covariates.reshape(-1, n_cov, order='C')
		yflat = outcomes.flatten(order='C')
		hatbeta = np.linalg.inv(Xflat.T @ Xflat) @ Xflat.T @ yflat
		residflat = yflat - Xflat @ hatbeta
		for _ in range(10):
			i = np.random.choice(n_obs)
			j = np.random.choice(n_subjects)
			k = np.where(np.all(covariates[i, j] == Xflat, axis=-1))[0].item()
			np.testing.assert_array_almost_equal(
				residflat[k], 
				residuals[i, j],
				decimal=5,
				err_msg=f"Panel factor residuals are inaccurate at i={i}, j={j}"
			)

class TestMosaicPanelTest(context.MosaicTest):
	"""
	tests mosaic factor test class
	"""

	def test_residual_orthogonality(self):
		n_obs, n_subjects, n_cov = 15, 200, 20
		# Create outcomes and covariates
		outcomes = np.random.randn(n_obs, n_subjects)
		covariates = np.random.randn(n_obs, n_subjects, n_cov)
		# Construct residuals
		mpt = mp.panel.MosaicPanelTest(
			outcomes=outcomes, covariates=covariates, test_stat=None
		)
		mpt.compute_mosaic_residuals()
		mpt.permute_residuals()
		# Check orthogonality
		for resids, name in zip([mpt.residuals, mpt._rtilde], ['Residuals', 'Permuted residuals']):
			for (batch, group) in mpt.tiles:
				tile_cov = covariates[np.ix_(batch, group)].reshape(-1, n_cov, order='C')
				tile_resids = resids[np.ix_(batch, group)].flatten(order='C')
				# Test orthogonality
				np.testing.assert_array_almost_equal(
					tile_cov.T @ tile_resids,
					np.zeros(n_cov),
					decimal=5,
					err_msg=f"{name} for batch={batch}, group={group} are not orthogonal to covariates."
				)
				# Test that the right covariates are lined up with the right outcomes
				for k in range(len(tile_resids)):
					i, j = np.where(resids == tile_resids[k])
					i = i.item(); j = j.item()
					np.testing.assert_array_almost_equal(
						covariates[i, j],
						tile_cov[k],
						decimal=5,
						err_msg=f"After reshaping, covariates are not aligned with outcomes"
					)

	def test_reduction_to_ols(self):
		"""
		Tests that with one tile, the residuals are OLS residuals 
		if the design does not change except every two observations.
		"""
		# Create data
		n_obs, n_subjects, n_cov = 10, 50, 5
		outcomes = np.random.randn(n_obs, n_subjects)
		covariates = np.random.randn(n_obs, n_subjects, n_cov)
		# Ensure covariates is the same as its augmented variant
		swapinds = mp.panel.construct_swapinds(n_obs)
		covariates = np.around(covariates + covariates[swapinds], 5)

		# Create tiles
		tiles = [(np.arange(n_obs), np.arange(n_subjects))]
		## compute mosaic residuals with one large tile
		mpt = mp.panel.MosaicPanelTest(
			outcomes=outcomes, covariates=covariates, test_stat=None, tiles=tiles
		)
		mpt.compute_mosaic_residuals()
		## Compute OLS residuals
		ols_resid = mp.panel.ols_residuals_panel(outcomes, covariates=covariates)
		np.testing.assert_array_almost_equal(
			mpt.residuals, ols_resid, decimal=5, err_msg=f"Mosaic residuals do not reduce to OLS resids with one tile"
		)

	def test_permute_residuals(self):
		# data
		np.random.seed(123)
		n_obs, n_subjects, n_cov = 150, 200, 20
		covariates = np.random.randn(n_obs, n_subjects, n_cov)
		outcomes = np.random.randn(n_obs, n_subjects)
		# Construct and permute residuals
		mpt = mp.panel.MosaicPanelTest(outcomes=outcomes, covariates=covariates, test_stat=None)
		mpt.compute_mosaic_residuals()
		mpt.permute_residuals()
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
				
	def test_exact_pval(self):
		"""
		For maximum efficiency, uses small scale data to test whether
		the p-value is valid.
		"""
		# Very small scale data
		np.random.seed(123)
		reps = 500
		nrand = 9
		n_obs, n_subjects, n_cov = 10, 15, 2
		# data
		covariates = stats.laplace.rvs(size=(n_obs, n_subjects, n_cov))
		beta = stats.laplace.rvs(size=n_cov)
		mu = covariates @ beta
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
				mpt = mp.panel.MosaicPanelTest(
					outcomes=outcomes,
					covariates=covariates,
					test_stat=fast_maxcorr_stat,
				) 
				mpt.fit(nrand=nrand, verbose=False)
				pvals[r] = mpt.pval

			buckets = np.around(pvals * (nrand + 1)).astype(int)
			counts = np.bincount(buckets)
			self.assertTrue(
				counts[0] == 0,
				f"with {nrand} randomizations, mosaic panel test produces a p-value < 1/(nrand + 1)"
			)
			counts = counts[1:]
			self.assertTrue(
				len(counts) == nrand + 1,
				f"with {nrand} randomizations, pvals only has {len(counts)} unique values (unique pvals={np.unique(pvals)})"
			)

			self.pearson_chi_square_test(counts, test_name='MosaicPanelTest p-values', alpha=0.001)

class TestMosaicPanelInference(context.MosaicTest):

	def test_residual_orthogonality(self):
		n_obs, n_subjects, n_cov = 15, 200, 20
		# Create outcomes and covariates
		outcomes = np.random.randn(n_obs, n_subjects)
		covariates = np.random.randn(n_obs, n_subjects, n_cov)
		# Construct residuals
		mpi = mp.panel.MosaicPanelInference(outcomes, covariates=covariates)
		mpi.compute_mosaic_residuals()
		for (batch, group) in mpi.tiles:
			tile_cov = covariates[np.ix_(batch, group)].reshape(-1, n_cov, order='C')
			tile_resids = mpi.residuals[np.ix_(batch, group)].flatten(order='C')
			# Test orthogonality
			np.testing.assert_array_almost_equal(
				tile_cov.T @ tile_resids,
				np.zeros(n_cov),
				decimal=5,
				err_msg=f"MPI residuals for batch={batch}, group={group} are not orthogonal to covariates."
			)
			# Test that the right covariates are lined up with the right outcomes
			for k in range(len(tile_resids)):
				i, j = np.where(mpi.residuals == tile_resids[k])
				i = i.item(); j = j.item()
				np.testing.assert_array_almost_equal(
					covariates[i, j],
					tile_cov[k],
					decimal=5,
					err_msg=f"After reshaping, covariates are not aligned with outcomes"
				)

	def test_lfo_residuals(self):
		"""
		Check that LFO residuals are correct
		"""
		np.random.seed(1234)
		# Generate data
		n_obs, n_subjects, n_cov = 10, 200, 20
		outcomes = np.random.randn(n_obs, n_subjects)
		covariates = np.random.randn(n_obs, n_subjects, n_cov)
		tiles = mp.tilings.default_panel_tiles(n_obs=n_obs, n_subjects=n_subjects, n_cov=n_cov)
		# Compute residuals and LFO residuals
		mpt = mp.panel.MosaicPanelInference(
			outcomes=outcomes,
			covariates=covariates,
			tiles=tiles,
		)
		mpt.compute_mosaic_residuals()
		for feature in range(n_cov):
			# Compare LFO residuals to a naively computed variant
			lfo_resids = mpt._downdate_mosaic_residuals(feature=feature)
			neg_feature = [i for i in range(n_cov) if i != feature]
			mpt2 = mp.panel.MosaicPanelInference(
				outcomes=outcomes,
				covariates=covariates[..., neg_feature],
				tiles=tiles,
			)
			expected = mpt2.compute_mosaic_residuals()
			np.testing.assert_array_almost_equal(
				lfo_resids,
				expected,
				decimal=8,
				err_msg=f"LFO resids are incorrect with feature={feature}."
			)

	def test_slope_intercept_representation(self):
		"""
		Tests that the test statistic for testing H_0 : beta[j] = b
		can be written as stat - b * slope for stat, slope.
		"""
		np.random.seed(1234)
		# Generate data
		n_obs, n_subjects, n_cov = 10, 200, 20
		outcomes = np.random.randn(n_obs, n_subjects)
		covariates = np.random.randn(n_obs, n_subjects, n_cov)
		tiles = mp.tilings.default_panel_tiles(n_obs=n_obs, n_subjects=n_subjects, n_cov=n_cov)
		# Compute residuals + LFO residuals
		mpinf = mp.panel.MosaicPanelInference(
			outcomes=outcomes,
			covariates=covariates,
			tiles=tiles,
		)
		mpinf.compute_mosaic_residuals()
		# Loop through several features
		for feature in np.random.choice(n_cov, size=min(n_cov, 5), replace=True):
			# Compute slope/statistic
			mpinf._downdate_mosaic_residuals(feature=feature)
			slope = mpinf.ZA_sums[:, 0].sum()
			stat = mpinf.Zresid_sums[:, 0].sum()
			# Manually leave the feature out and invert the null
			neg_feature = [i for i in range(n_cov) if i != feature]
			bs = np.random.randn(5)
			for b in bs:
				Z = covariates[:, :, feature]
				new_outcomes = outcomes - b * Z
				mpinf_manual = mp.panel.MosaicPanelInference(
					outcomes=new_outcomes,
					covariates=covariates[:, :, neg_feature],
					tiles=tiles,
				)
				mpinf_manual.compute_mosaic_residuals()
				stat_new = np.sum(mpinf_manual.residuals * Z)
				np.testing.assert_array_almost_equal(
					stat - b * slope,
					stat_new,
					decimal=5,
					err_msg=f"Slope/intercept form used in inversion is inaccurate with b={b}, feature={feature}"
				)

	def test_linear_inversion(self):
		"""
		Tests that the algebra for the linear test statistic inversion is correct.
		"""
		np.random.seed(123)
		nrand, alpha = 100, 0.05
		for _ in range(10):
			slope = 100 * np.random.uniform()
			# Create statistics/null_slopes
			stat = np.random.randn()
			null_slopes = np.random.uniform(-slope, slope, size=nrand)
			null_stats = np.random.randn(nrand)
			# Queries p-value at a specific value of beta
			def query_pval(beta):
				new_stat = np.abs(stat - beta * slope)
				new_null_stats = np.abs(null_stats - beta * null_slopes)
				return (1 + np.sum(new_stat <= new_null_stats)) / (1 + nrand)
			# Compute lower and upper bound
			lower, upper = mp.panel.invert_linear_statistic(
				stat=stat, slope=slope, null_stats=null_stats, 
				null_slopes=null_slopes, alpha=alpha,
			)
			# Query
			tol = 1e-3
			for boundary, bsign, bname in zip(
				[lower, upper], [1, -1], ['lower', 'upper']
			):
				for sign in [-1, 1]:
					beta = boundary + sign * tol
					pval = query_pval(beta)
					self.assertTrue(
						bsign * sign * pval >= bsign * sign * alpha,
						f"invert_linear_statistic produces {bname}={boundary}, but at beta={beta}, pval={pval} with alpha={alpha}"
					)
			for beta in np.random.uniform(lower+tol, upper-tol, size=20):
				pval = query_pval(beta)
				self.assertTrue(
					pval >= alpha,
					f"invert_linear_statistic produces CI={[lower, upper]} but at beta={beta}, pval={pval} with alpha={alpha}"
				)

if __name__ == "__main__":
	# Run tests---useful if using cprofilev
	basename = os.path.basename(os.path.abspath(__file__))
	if sys.argv[0] == f'test/{basename}':
		time0 = time.time()
		context.run_all_tests([TestMosaicPanelTest(), TestOLSPanelResids()])
		elapsed = np.around(time.time() - time0, 2)
		print(f"Finished running all tests at time={elapsed}")

	# Else let unittest handle this for us
	else:
		unittest.main()