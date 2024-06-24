import numpy as np
import pandas as pd
import scipy.linalg
from typing import Optional, Union
from . import core, utilities, tilings

def construct_swapinds(t):
	"""
	Swaps elements t-1 and t.
	"""
	swapinds = np.arange(t)
	swapinds[1::2] -= 1
	swapinds[::2] += 1
	return np.minimum(swapinds, t-1).astype(int)

def ols_residuals_panel(
	outcomes: np.array, covariates: np.array,
):	
	"""
	Computes residuals via OLS (NOT cross-sectional OLS).

	Parameters
	----------
	outcomes : np.array
		(n_obs, n_subjects) array of outcome data.
	exposures : np.array
		(n_obs, n_subjects, n_cov) array of covariates.
	
	Returns
	-------
	residuals : np.array
		(n_obs, n_subjects) array of OLS residuals.
	"""
	n_obs, n_subjects, n_cov = covariates.shape
	# Flatten for QR decomposition
	yflat = outcomes.flatten(order='C')
	Xflat = covariates.reshape(-1, n_cov, order='C')
	Q, _ = np.linalg.qr(Xflat)
	resid = (yflat - Q @ (Q.T @ yflat))
	# Un-flatten and return
	return resid.reshape(n_obs, n_subjects, order='C')

def invert_linear_statistic(
	stat: float, 
	slope: float,
	null_stats: np.array,
	null_slopes: np.array,
	alpha: float,
):
	"""
	Finds the set of beta such that
	
	|stat - beta slope| >= Q_{alpha}(|null_stats - beta * null_slopes|)
	
	under the assumption that |null_slopes| <= |slope|.
	"""
	nrand = len(null_stats)
	if slope <= 0:
		raise ValueError(f"slope={slope} <= 0")
	max_null_slope = np.abs(null_slopes).max()
	if max_null_slope > slope:
		raise ValueError(f"max_abs_null_slope={max_null_slope}  slope={slope}")
	
	# Equi-tailed threshold
	rthresh = int(np.floor(alpha * (nrand + 1))) - 1
	if rthresh < 0:
		return -np.inf, np.inf
	# First set of inflections
	inflections0 = np.zeros(nrand) - np.inf
	denoms0 = slope - null_slopes
	flags0 = denoms0 != 0
	inflections0[flags0] = (stat - null_stats)[flags0] / denoms0[flags0]
	# Second set of inflections
	inflections1 = np.zeros(nrand) + np.inf
	denoms1 = slope + null_slopes
	flags1 = denoms1 != 0
	inflections1[flags1] = (stat + null_stats)[flags1] / denoms1[flags1]
	# Combine and sort
	inflections = np.sort(np.stack([inflections0, inflections1], axis=1), axis=1)
	# Confidence bounds
	lci = np.sort(inflections[:, 0])[rthresh]
	uci = np.sort(inflections[:, 1])[-(rthresh+1)]
	return lci, uci


class MosaicPanelTest(core.MosaicPermutationTest):
	"""
	Tests goodness-of-fit of a linear model for panel data
	under a local exchangeability and (optional) cluster
	independence assumption.
	"""
	def __init__(
		self, 
		outcomes: np.array,
		covariates: np.array,
		test_stat: callable,
		test_stat_kwargs: Optional[dict]=None,
		clusters: Optional[np.array]=None,
		tiles: Optional[tilings.Tiling]=None,
		**kwargs,
	):
		# Check that the shape of the covariates is correct
		if len(covariates.shape) != 3:
			raise ValueError(f"covariates has shape {covariates.shape} but expected a 3D array")
		# Process data
		self.outcomes, self.covariates, self.missing_pattern = core._preprocess_data(
			outcomes=outcomes, covariates=covariates
		)
		self.n_obs, self.n_subjects = self.outcomes.shape
		self.n_cov = self.covariates.shape[-1]
		# Test statistic
		self.test_stat = test_stat
		self.tstat_kwargs = test_stat_kwargs
		if self.tstat_kwargs is None:
			self.tstat_kwargs = {}
		# Tiles
		self.clusters = clusters
		self.tiles = tiles
		if self.tiles is None:
			self.tiles = tilings.default_panel_tiles(
				n_obs=self.n_obs,
				n_subjects=self.n_subjects,
				n_cov=self.n_cov,
				clusters=self.clusters,
				**kwargs
			)
		# initialize
		super().__init__()

	def compute_mosaic_residuals(self) -> np.array:
		"""
		Computes mosaic residuals for panel data.
		"""
		# Loop through tiles
		self.residuals = np.zeros((self.n_obs, self.n_subjects))
		for batch, group in self.tiles:
			# Augmented design
			Xtile = self.covariates[np.ix_(batch, group)]
			swapinds = construct_swapinds(len(batch))
			Xstar = np.concatenate([Xtile, Xtile[swapinds]], axis=-1)
			Xstar = np.unique(Xstar, axis=-1)
			# Compute OLS residuals and save in the correct position
			self.residuals[np.ix_(batch, group)] = ols_residuals_panel(
				outcomes=self.outcomes[np.ix_(batch, group)],
				covariates=Xstar,
			)

		# Initialize rtilde correctly
		self._rtilde = self.residuals.copy()
		return self.residuals

	def permute_residuals(self) -> None:
		"""
		Permutes residuals within tiles. Result is stored in ``self._rtilde``.
		
		Returns
		-------
		None : NoneType
		"""
		# Random uniforms
		U = np.random.uniform(size=len(self.tiles))
		for i, (batch, group) in enumerate(self.tiles):
			if U[i] <= 0.5:
				# This reshaping is faster than but equivalent to using np.ix_
				swapinds = construct_swapinds(len(batch)).reshape(-1, 1)
				br = batch.reshape(-1, 1)
				gr = group.reshape(1, -1)
				self._rtilde[br, gr] = self._rtilde[swapinds, gr]


class MosaicPanelInference:
	"""
	Produces confidence interval for linear models in panel data.
	"""

	def __init__(
		self,
		outcomes: np.array,
		covariates: np.array,
		clusters: Optional[np.array]=None,
		tiles: Optional[tilings.Tiling]=None,
		**kwargs,
	):
		# Check that the shape of the covariates is correct
		if len(covariates.shape) != 3:
			raise ValueError(f"covariates has shape {covariates.shape} but expected a 3D array")
		# Process data
		self.outcomes, self.covariates, self.missing_pattern = core._preprocess_data(
			outcomes=outcomes, covariates=covariates
		)
		self.n_obs, self.n_subjects = self.outcomes.shape
		self.n_cov = self.covariates.shape[-1]
		# Tiles
		self.clusters = clusters
		self.tiles = tiles
		if self.tiles is None:
			self.tiles = tilings.default_panel_tiles(
				n_obs=self.n_obs,
				n_subjects=self.n_subjects,
				n_cov=self.n_cov,
				clusters=self.clusters,
				**kwargs
			)
		# initialize
		# super().__init__()

	def compute_mosaic_residuals(self, verbose: bool=True):
		"""
		Computes full mosaic residuals and precomputes useful quantities.
		"""
		# Loop through tiles
		self.residuals = np.zeros((self.n_obs, self.n_subjects))
		self._X_aug_inv_inds = []
		self._QR_tiles = []
		for tilenum in utilities.vrange(len(self.tiles), verbose=verbose):
			batch, group = self.tiles[tilenum]
			# Augmented design
			Xtile = self.covariates[np.ix_(batch, group)].copy()
			swapinds = construct_swapinds(len(batch))
			Xstar = np.concatenate([Xtile, Xtile[swapinds]], axis=-1)
			# Flatten and compute QR decomposition
			Xstar_flat = Xstar.reshape(-1, Xstar.shape[-1], order='C')
			Xstar_flat, inv_inds = np.unique(
				Xstar_flat, axis=-1, return_inverse=True
			)
			Q, R = np.linalg.qr(Xstar_flat)
			# Compute final residuals
			yflat = self.outcomes[np.ix_(batch, group)].flatten(order='C')
			resid = (yflat - Q @ (Q.T @ yflat))
			# Un-flatten and save in the correct position
			self._QR_tiles.append((Q, R))
			self._X_aug_inv_inds.append(inv_inds)
			self.residuals[np.ix_(batch, group)] = resid.reshape(
				len(batch), len(group), order='C'
			)

		# Initialize rtilde correctly
		self._rtilde = self.residuals.copy()
		return self.residuals

	def _downdate_mosaic_residuals(self, feature: int) -> np.array:
		"""
		Efficiently computes mosaic residuals leaving 
		out feature j.

		Parameters
		----------
		feature : int
			Integer in {0, ..., n_cov - 1}.

		Returns
		-------
		lfo_resids : np.array
			Resids leaving out ``feature``

		Notes
		-----
		Also projects out the influence of the other covariates on
		``self.covariates[:, :, feature]`` and stores the result in
		``self.orthog_feature``.
		"""
		# Stores "leave feature out" residuals
		self.lfo_resids = np.zeros(self.residuals.shape)
		# Stores the feature
		self.orthog_feature = np.zeros(self.residuals.shape)
		# If Z is the orthogonalized feature, stores Z
		self.Zresid_sums = np.zeros((len(self.tiles), 2))
		self.ZA_sums = np.zeros((len(self.tiles), 2))
		for tilenum, (batch, group), (Qtile, Rtile), inv_inds in zip(
			range(len(self.tiles)), self.tiles, self._QR_tiles, self._X_aug_inv_inds
		):
			swapinds = construct_swapinds(len(batch))
			# Find which indices to eliminate from the downdate
			to_elim = np.array([inv_inds[feature], inv_inds[feature + self.n_cov]])
			# Sort in reverse order
			to_elim = np.flip(np.sort(to_elim))
			# Perform the QR downdate
			Q, R = Qtile.copy(), Rtile.copy()
			for k in to_elim:
				Q, R = scipy.linalg.qr_delete(Q, R, k=k, p=1, which='col', check_finite=True)
			# Compute residuals of outcome
			yflat = self.outcomes[np.ix_(batch, group)].flatten(order='C')
			resid = (yflat - Q @ (Q.T @ yflat)).reshape(len(batch), len(group), order='C')
			# Compute residuals of feature
			Z = self.covariates[np.ix_(batch, group)][:, :, feature]
			Zflat = Z.flatten(order='C')
			A = (Zflat - Q @ (Q.T @ Zflat)).reshape(len(batch), len(group), order='C')
			# Precompute sums used to compute slopes/intercepts within tiles
			self.Zresid_sums[tilenum, 0] = np.sum(Z * resid)
			self.Zresid_sums[tilenum, 1] = np.sum(Z * resid[swapinds])
			self.ZA_sums[tilenum, 0] = np.sum(Z * A)
			self.ZA_sums[tilenum, 1] = np.sum(Z * A[swapinds])
			# Save residuals/orthogonal features in the correct position
			self.lfo_resids[np.ix_(batch, group)] = resid
			self.orthog_feature[np.ix_(batch, group)] = A
		# return
		return self.lfo_resids

	def _compute_confidence_intervals(
		self, 
		alpha: int,
		nrand: int,
		features: Optional[np.array]=None,
		verbose: bool=True,
	):
		"""
		Computes underlying confidence intervals for :meth:`fit`.

		Parameters
		----------
		alpha : int
			Desired nominal Type I error rate.
		nrand : int
			Number of randomizations.
		features : np.array
			The list of features to compute CIs for. 
			Defaults to all features.
		verbose : bool
			If True, show progress bars.
		"""
		# process features
		if features is None:
			features = np.arange(self.n_cov).astype(int)
		if not utilities.haslength(features):
			features = np.array([features]).astype(int)

		# To save the confidence intervals
		columns = ['Estimate', 'Lower', 'Upper', 'p-value']
		self._summary = pd.DataFrame(
			np.zeros((len(features), 4)),
			index=features,
			columns=columns,
		)
		self._summary.values[:] = np.nan

		# Save the slopes and statistics
		self._null_estimates = np.zeros((self.n_cov, nrand))
		self._null_estimate_slopes = np.zeros((self.n_cov, nrand))


		# Loop through and compute
		ntiles = len(self.tiles)
		xinds = np.stack([np.arange(ntiles) for _ in range(nrand)])
		for j in utilities.vrange(len(features), verbose=verbose):
			feature = features[j]
			self._downdate_mosaic_residuals(feature=feature)
			## Todo: is this a bottleneck? If so, it is valid to move it outside the for loop
			flips = np.random.binomial(1, 0.5, size=(nrand, ntiles)).astype(int)
			# Observe test statistic + slope as beta changes
			stat = self.Zresid_sums[:, 0].sum()
			slope = self.ZA_sums[:, 0].sum()
			# Compute null test statistics + slopes as beta changes
			null_stats = self.Zresid_sums[(xinds, flips)].sum(axis=1)
			null_slopes = self.ZA_sums[(xinds, flips)].sum(axis=1)
			# Compute CI
			lower, upper = invert_linear_statistic(
				stat=stat, 
				slope=slope,
				null_stats=null_stats,
				null_slopes=null_slopes,
				alpha=alpha,
			)
			# Save
			self._null_estimates = null_stats / slope
			self._null_estimate_slopes = null_slopes / slope
			# Compute p-value
			pval = (1 + np.sum(np.abs(stat) <= np.abs(null_stats))) / (1 + nrand)
			# Estimate
			estimate = stat / slope
			# Save
			self._summary.loc[feature] = pd.Series(
				[estimate, lower, upper, pval],
				index=columns
			)

	def fit(
		self,
		nrand: int=10000,
		alpha: int=0.05,
		features: Optional[np.array]=None,
		verbose: bool=True,
	):
		"""
		Fits the linear model and returns confidence intervals.

		Parameters
		----------
		alpha : int
			Desired nominal Type I error rate.
		nrand : int
			Number of randomizations.
		features : np.array
			The list of features to compute CIs for. 
			Defaults to all features.
		verbose : bool
			If True, show progress bars.

		Returns
		-------
		self : object
		"""
		if verbose:
			print("Computing mosaic residuals.")
		self.compute_mosaic_residuals(verbose=verbose)
		if verbose:
			print("Computing confidence intervals.")
		self._compute_confidence_intervals(
			nrand=nrand, alpha=alpha, features=features, verbose=verbose
		)
		return self

	@property
	def summary(self):
		"""
		Produces a summary of key inferential results.
		"""
		return self._summary
