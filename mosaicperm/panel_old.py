# import numpy as np
# import pandas as pd
# import scipy.linalg
# from typing import Optional, Union
# from . import core, utilities, tilings
# # fast fixed effects via sparse arrays
# import sparse
# import sklearn.linear_model as lm

# def construct_swapinds(t):
# 	"""
# 	Swaps elements t-1 and t.
# 	"""
# 	swapinds = np.arange(t)
# 	swapinds[1::2] -= 1
# 	swapinds[::2] += 1
# 	return np.minimum(swapinds, t-1).astype(int)

# ### Handling fixed effects
# def _parse_fixed_effects_formula(formula: str) -> list:
# 	"""
# 	Examples:
# 		- '0' becomes [(0,)]
# 		- '0+1*2*3' becomes [(0,), (1,2,3,)] 
# 		- '0+1*2 + 3 ' becomes [(0,), (1,2), (3,)]
# 	"""
# 	formula = formula.replace(" ", "")
# 	formula = formula.split("+")
# 	output = []
# 	for x in formula:
# 		output.append(tuple([int(y) for y in x.split('*')]))
# 	return output

# def _construct_fex_clusters(outcome_shape: tuple, fex_axes: tuple) -> np.array:
# 	"""
# 	Parameters
# 	----------
# 	outcome_shape : shape 
# 	fex_axes : tuple of axes to use as fixed effects.
# 	E.g., (1,2) means a fixed effect by axis 1/2.

# 	Returns
# 	-------
# 	clusters : np.array
# 		Array of shape outcome_shape which defines clusters for each fixed effect.
# 		Two observations are in the same cluster iff they have the same value.
# 	"""
# 	clusters = np.zeros(outcome_shape, dtype=int)
# 	for axis in fex_axes:
# 		axsize = outcome_shape[axis]
# 		arange_shape = [1 if j != axis else axsize for j in range(len(outcome_shape))]
# 		clusters = clusters * (axsize + 1) + np.arange(axsize).reshape(*arange_shape)
# 	return clusters

# def _clusters_to_sparse_fex_dummies(
# 	clusters: np.array, 
# 	missing_pattern: Optional[np.array]=None,
# ) -> sparse.COO:
# 	"""
# 	Given clusters (as produced in the previous function), constructs dummies.

# 	missing_pattern : np.array
# 		Boolean array of the same shape as clulsters. Observation (i,j)
# 		is missing iff missing_pattern[i,j] = True.
# 	"""
# 	# Loop through and create
# 	dummies = []
# 	for v in np.unique(clusters):
# 		flags = clusters == v
# 		# Missing values (later imputed as zero) should not be used to estimate fixed effects
# 		if missing_pattern is not None:
# 			flags = flags & (~missing_pattern)
# 		dummies.append(
# 			sparse.COO(coords=np.stack(np.where(flags), axis=0), data=np.ones(np.sum(flags)), shape=clusters.shape)
# 		)
# 	return sparse.stack(dummies, axis=-1)

# def _dummies_from_fex_formula(outcome_shape: tuple, fex_formula: str, missing_pattern: Optional[np.array]=None):
# 	dummies = []
# 	fex_axes_list = _parse_fixed_effects_formula(fex_formula)
# 	for fex_axes in fex_axes_list:
# 		clusters = _construct_fex_clusters(outcome_shape, fex_axes=fex_axes)
# 		dummies.append(_clusters_to_sparse_fex_dummies(clusters, missing_pattern=missing_pattern))
# 	return sparse.concatenate(dummies, axis=-1)


# def ols_residuals_panel(
# 	outcomes: np.array, 
# 	covariates: np.array,
# 	fex_formula: Optional[str]=None,
# 	missing_pattern: Optional[np.array]=None,
# ):	
# 	"""
# 	Computes residuals via OLS (NOT cross-sectional OLS).

# 	Parameters
# 	----------
# 	outcomes : np.array
# 		(n_obs, n_subjects, ...) N-d array of outcome data.
# 	covariates : np.array
# 		(n_obs, n_subjects, ..., n_cov) array of covariates.
# 	fex_formula : str
# 		Optional formula specifying fixed effects by axis.
# 		E.g., '0+1' specifies standard two-way fixed effects,
# 	missing_pattern : np.array
# 		Array specifying which elements of ``outcomes`` are missing.
# 	Returns
# 	-------
# 	residuals : np.array
# 		(n_obs, n_subjects) array of OLS residuals.
# 	"""
# 	# Possibly add axis-aligned fixed effects. 
# 	# Note that these never need to be duplicated to preserve
# 	# local exchangeability. 
# 	if fex_formula is not None:
# 		dummies = _dummies_from_fex_formula(
# 			outcomes.shape, fex_formula=fex_formula, missing_pattern=missing_pattern
# 		)
# 		covariates = sparse.concatenate([sparse.COO(covariates), dummies], axis=-1)
# 	n_cov = covariates.shape[-1]
# 	yflat = outcomes.flatten(order='C')
# 	# Two cases: sparse and dense
# 	# Case 1: dense.
# 	if isinstance(covariates, np.ndarray):
# 		# Flatten for QR decomposition
# 		Xflat = covariates.reshape(-1, n_cov, order='C')
# 		Q, _ = np.linalg.qr(Xflat)
# 		resid = (yflat - Q @ (Q.T @ yflat))
# 	# Case 2: sparse
# 	elif isinstance(covariates, sparse.COO):
# 		Xflat = covariates.reshape((-1, n_cov), order='C').to_scipy_sparse()
# 		ols = lm.LinearRegression(fit_intercept=False)
# 		ols.fit(y=yflat, X=Xflat)
# 		resid = yflat - ols.predict(Xflat)
# 	return resid.reshape(*outcomes.shape, order='C')

# class FlatMosaicPanelTest(core.MosaicPermutationTest):
# 	"""
# 	Note to self: this is an extension that should go in the appendix IMO.
# 	"""

# 	def __init__(
# 		self,
# 		data: pd.DataFrame,
# 		outcome_col: str,
# 		test_stat: callable,
# 		cluster_col: str,
# 		time_col: str, # column denoting time
# 		subject_col: str, # column denoting subject
# 		cts_covariates: Optional[list]=None,
# 		discrete_covariates: Optional[list]=None,
# 		tstat_kwargs: Optional[dict]=None,
# 		ntiles: int=10,
# 	):
# 		# Sort data by subjects and time
# 		data = data.sort_values([subject_col, time_col])
# 		# Save
# 		self.times = data[time_col].values
# 		self.subjects = tilings._preprocess_partition(data[subject_col].values)
# 		self.clusters = tilings._preprocess_partition(data[cluster_col].values)
# 		self.outcomes = data[outcome_col].values
# 		self.n = len(self.outcomes)
# 		# Construct covariates
# 		if cts_covariates is not None:
# 			self.covariates = data[cts_covariates].values ##TODO fixed effects + sparse matrices, whatever
# 		else:
# 			self.covariates = np.ones((self.n, 1))
# 		# Possibly add discrete covariates
# 		if discrete_covariates is not None:
# 			dummies = []
# 			for col in discrete_covariates:
# 				dummies.append(_clusters_to_sparse_fex_dummies(data[col]))
# 			dummies = sparse.concatenate(dummies, axis=-1)
# 			self.covariates = sparse.concatenate(
# 				[sparse.COO(self.covariates), dummies], axis=-1
# 			)
# 		self.n_cov = self.covariates.shape[1]
# 		self.test_stat = test_stat
# 		self.tstat_kwargs = dict() if tstat_kwargs is None else tstat_kwargs
# 		# Create tiles
# 		self._construct_full_swapinds()
# 		self.tile_ids = tilings.coarsify_partition(self.clusters, k=ntiles, random=False)
# 		self.tile_inds = [
# 			np.where(self.tile_ids == tile_id)[0] for tile_id in np.unique(self.tile_ids)
# 		]

# 	def _construct_full_swapinds(self):
# 		breaks = np.where(self.subjects[:-1] != self.subjects[1:])[0]
# 		breaks = np.concatenate([[0], breaks, [self.n]])
# 		self.swapinds = []
# 		for j in range(1, len(breaks)):
# 		    self.swapinds.append(construct_swapinds(breaks[j] - breaks[j-1]) + breaks[j-1])
# 		self.swapinds = np.concatenate(self.swapinds)
# 		# Never swap residuals which are not in the same cluster
# 		to_reset = np.where(self.clusters != self.clusters[self.swapinds])[0]
# 		self.swapinds[to_reset] = np.arange(self.n)[to_reset]

# 	def compute_mosaic_residuals(self):
# 		self.residuals = np.zeros(self.n)
# 		for inds in self.tile_inds:
# 			self.residuals[inds] = ols_residuals_panel(
# 				outcomes=self.outcomes[inds], # this is flat, still fine with this function call
# 				covariates=self.covariates[inds],
# 			)
# 		self._rtilde = self.residuals.copy()


# 	def permute_residuals(self):
# 		# unifroms
# 		U = np.random.uniform(size=len(self.tile_inds))
# 		# Loop through and possibly permute
# 		for i, inds in enumerate(self.tile_inds):
# 			if U[i] <= 0.5:
# 				self._rtilde[inds] = self.residuals[self.swapinds[inds]]
# 			else:
# 				self._rtilde[inds] = self.residuals[inds]


# class MosaicPanelTest(core.MosaicPermutationTest):
# 	"""
# 	Tests goodness-of-fit of a linear model for panel data
# 	under a local exchangeability and (optional) cluster
# 	independence assumption.
# 	"""
# 	def __init__(
# 		self, 
# 		outcomes: np.array,
# 		covariates: np.array,
# 		test_stat: callable,
# 		test_stat_kwargs: Optional[dict]=None,
# 		clusters: Optional[np.array]=None,
# 		tiles: Optional[tilings.Tiling]=None,
# 		fex_formula: Optional[str]=None,
# 		**kwargs,
# 	):
# 		# Process data
# 		self.outcomes, self.covariates, self.missing_pattern = core._preprocess_data(
# 			outcomes=outcomes, covariates=covariates
# 		)
# 		self.n_obs, self.n_subjects = self.outcomes.shape[0:2]
# 		self.n_cov = self.covariates.shape[-1]
# 		self.fex_formula = fex_formula
# 		# Test statistic
# 		self.test_stat = test_stat
# 		self.tstat_kwargs = test_stat_kwargs
# 		if self.tstat_kwargs is None:
# 			self.tstat_kwargs = {}
# 		# Tiles
# 		self.clusters = clusters
# 		self.tiles = tiles
# 		if self.tiles is None:
# 			self.tiles = tilings.default_panel_tiles(
# 				n_obs=self.n_obs,
# 				n_subjects=self.n_subjects,
# 				n_cov=self.n_cov,
# 				clusters=self.clusters,
# 				**kwargs
# 			)
# 		# initialize
# 		self._enforce_local_exchangeability()
# 		super().__init__()

# 	def _enforce_local_exchangeability(self):
# 		# Readjust outcomes to ensure that missing pattern
# 		# does not cause local exchangeability violations
# 		for (batch, group) in self.tiles:
# 			missing_subjects = np.any(
# 				self.missing_pattern[np.ix_(batch, group)], 
# 				# the following line is equivalent to axis=0 if outcomes is 2D
# 				axis=tuple([x for x in range(self.outcomes.ndim) if x != 1]),
# 			)
# 			self.outcomes[np.ix_(batch, group[missing_subjects])] = 0
# 			self.covariates[np.ix_(batch, group[missing_subjects])] = 0
# 			self.missing_pattern[np.ix_(batch, group[missing_subjects])] = True

# 	def compute_mosaic_residuals(self) -> np.array:
# 		"""
# 		Computes mosaic residuals for panel data.
# 		"""
# 		# Loop through tiles
# 		self.residuals = np.zeros(self.outcomes.shape)
# 		for batch, group in self.tiles:
# 			# Augmented design
# 			Xtile = self.covariates[batch][:, group]
# 			swapinds = construct_swapinds(len(batch))
# 			Xstar = np.concatenate([Xtile, Xtile[swapinds]], axis=-1)
# 			# Check uniqueness if a dense array
# 			if isinstance(Xtile, np.ndarray):
# 				Xstar = np.unique(Xstar, axis=-1)
# 			# Compute OLS residuals and save in the correct position
# 			self.residuals[np.ix_(batch, group)] = ols_residuals_panel(
# 				outcomes=self.outcomes[batch][:, group],
# 				covariates=Xstar,
# 				fex_formula=self.fex_formula,
# 				missing_pattern=self.missing_pattern[batch][:, group],
# 			)

# 		# Initialize rtilde correctly
# 		self._rtilde = self.residuals.copy()
# 		return self.residuals

# 	def permute_residuals(self) -> None:
# 		"""
# 		Permutes residuals within tiles. Result is stored in ``self._rtilde``.
		
# 		Returns
# 		-------
# 		None : NoneType
# 		"""
# 		# Random uniforms
# 		U = np.random.uniform(size=len(self.tiles))
# 		for i, (batch, group) in enumerate(self.tiles):
# 			if U[i] <= 0.5:
# 				# This reshaping is faster than but equivalent to using np.ix_
# 				swapinds = construct_swapinds(len(batch))
# 				br = batch.reshape(-1, 1)
# 				gr = group.reshape(1, -1)
# 				self._rtilde[br, gr] = self._rtilde[br[swapinds], gr]

# ### Confidence intervals
# def invert_linear_statistic(
# 	stat: float, 
# 	slope: float,
# 	null_stats: np.array,
# 	null_slopes: np.array,
# 	alpha: float,
# ):
# 	"""
# 	Finds the set of beta such that
	
# 	|stat - beta slope| >= Q_{alpha}(|null_stats - beta * null_slopes|)
	
# 	under the assumption that |null_slopes| <= |slope|.
# 	"""
# 	nrand = len(null_stats)
# 	if slope <= 0:
# 		raise ValueError(f"slope={slope} <= 0")
# 	max_null_slope = np.abs(null_slopes).max()
# 	if max_null_slope > slope:
# 		raise ValueError(f"max_abs_null_slope={max_null_slope}  slope={slope}")
	
# 	# Equi-tailed threshold
# 	rthresh = int(np.floor(alpha * (nrand + 1))) - 1
# 	if rthresh < 0:
# 		return -np.inf, np.inf
# 	# First set of inflections
# 	inflections0 = np.zeros(nrand) - np.inf
# 	denoms0 = slope - null_slopes
# 	flags0 = denoms0 != 0
# 	inflections0[flags0] = (stat - null_stats)[flags0] / denoms0[flags0]
# 	# Second set of inflections
# 	inflections1 = np.zeros(nrand) + np.inf
# 	denoms1 = slope + null_slopes
# 	flags1 = denoms1 != 0
# 	inflections1[flags1] = (stat + null_stats)[flags1] / denoms1[flags1]
# 	# Combine and sort
# 	inflections = np.sort(np.stack([inflections0, inflections1], axis=1), axis=1)
# 	# Confidence bounds
# 	lci = np.sort(inflections[:, 0])[rthresh]
# 	uci = np.sort(inflections[:, 1])[-(rthresh+1)]
# 	return lci, uci

# class MosaicPanelInference:
# 	"""
# 	Produces confidence interval for linear models in panel data.
# 	"""

# 	def __init__(
# 		self,
# 		outcomes: np.array,
# 		covariates: np.array,
# 		clusters: Optional[np.array]=None,
# 		tiles: Optional[tilings.Tiling]=None,
# 		**kwargs,
# 	):
# 		# Check that the shape of the covariates is correct
# 		if len(covariates.shape) != 3:
# 			raise ValueError(f"covariates has shape {covariates.shape} but expected a 3D array")
# 		# Process data
# 		self.outcomes, self.covariates, self.missing_pattern = core._preprocess_data(
# 			outcomes=outcomes, covariates=covariates
# 		)
# 		self.n_obs, self.n_subjects = self.outcomes.shape
# 		self.n_cov = self.covariates.shape[-1]
# 		# Tiles
# 		self.clusters = clusters
# 		self.tiles = tiles
# 		if self.tiles is None:
# 			self.tiles = tilings.default_panel_tiles(
# 				n_obs=self.n_obs,
# 				n_subjects=self.n_subjects,
# 				n_cov=self.n_cov,
# 				clusters=self.clusters,
# 				**kwargs
# 			)
# 		# initialize
# 		# super().__init__()

# 	def compute_mosaic_residuals(self, verbose: bool=True):
# 		"""
# 		Computes full mosaic residuals and precomputes useful quantities.
# 		"""
# 		# Loop through tiles
# 		self.residuals = np.zeros((self.n_obs, self.n_subjects))
# 		self._X_aug_inv_inds = []
# 		self._QR_tiles = []
# 		for tilenum in utilities.vrange(len(self.tiles), verbose=verbose):
# 			batch, group = self.tiles[tilenum]
# 			# Augmented design
# 			Xtile = self.covariates[np.ix_(batch, group)].copy()
# 			swapinds = construct_swapinds(len(batch))
# 			Xstar = np.concatenate([Xtile, Xtile[swapinds]], axis=-1)
# 			# Flatten and compute QR decomposition
# 			Xstar_flat = Xstar.reshape(-1, Xstar.shape[-1], order='C')
# 			Xstar_flat, inv_inds = np.unique(
# 				Xstar_flat, axis=-1, return_inverse=True
# 			)
# 			Q, R = np.linalg.qr(Xstar_flat)
# 			# Compute final residuals
# 			yflat = self.outcomes[np.ix_(batch, group)].flatten(order='C')
# 			resid = (yflat - Q @ (Q.T @ yflat))
# 			# Un-flatten and save in the correct position
# 			self._QR_tiles.append((Q, R))
# 			self._X_aug_inv_inds.append(inv_inds)
# 			self.residuals[np.ix_(batch, group)] = resid.reshape(
# 				len(batch), len(group), order='C'
# 			)

# 		# Initialize rtilde correctly
# 		self._rtilde = self.residuals.copy()
# 		return self.residuals

# 	def _downdate_mosaic_residuals(self, feature: int) -> np.array:
# 		"""
# 		Efficiently computes mosaic residuals leaving 
# 		out feature j.

# 		Parameters
# 		----------
# 		feature : int
# 			Integer in {0, ..., n_cov - 1}.

# 		Returns
# 		-------
# 		lfo_resids : np.array
# 			Resids leaving out ``feature``

# 		Notes
# 		-----
# 		Also projects out the influence of the other covariates on
# 		``self.covariates[:, :, feature]`` and stores the result in
# 		``self.orthog_feature``.
# 		"""
# 		# Stores "leave feature out" residuals
# 		self.lfo_resids = np.zeros(self.residuals.shape)
# 		# Stores the feature
# 		self.orthog_feature = np.zeros(self.residuals.shape)
# 		# If Z is the orthogonalized feature, stores Z
# 		self.Zresid_sums = np.zeros((len(self.tiles), 2))
# 		self.ZA_sums = np.zeros((len(self.tiles), 2))
# 		for tilenum, (batch, group), (Qtile, Rtile), inv_inds in zip(
# 			range(len(self.tiles)), self.tiles, self._QR_tiles, self._X_aug_inv_inds
# 		):
# 			swapinds = construct_swapinds(len(batch))
# 			# Find which indices to eliminate from the downdate
# 			to_elim = np.array([inv_inds[feature], inv_inds[feature + self.n_cov]])
# 			# Sort in reverse order
# 			to_elim = np.flip(np.sort(to_elim))
# 			# Perform the QR downdate
# 			Q, R = Qtile.copy(), Rtile.copy()
# 			for k in to_elim:
# 				Q, R = scipy.linalg.qr_delete(Q, R, k=k, p=1, which='col', check_finite=True)
# 			# Compute residuals of outcome
# 			yflat = self.outcomes[np.ix_(batch, group)].flatten(order='C')
# 			resid = (yflat - Q @ (Q.T @ yflat)).reshape(len(batch), len(group), order='C')
# 			# Compute residuals of feature
# 			Z = self.covariates[np.ix_(batch, group)][:, :, feature]
# 			Zflat = Z.flatten(order='C')
# 			A = (Zflat - Q @ (Q.T @ Zflat)).reshape(len(batch), len(group), order='C')
# 			# Precompute sums used to compute slopes/intercepts within tiles
# 			self.Zresid_sums[tilenum, 0] = np.sum(Z * resid)
# 			self.Zresid_sums[tilenum, 1] = np.sum(Z * resid[swapinds])
# 			self.ZA_sums[tilenum, 0] = np.sum(Z * A)
# 			self.ZA_sums[tilenum, 1] = np.sum(Z * A[swapinds])
# 			# Save residuals/orthogonal features in the correct position
# 			self.lfo_resids[np.ix_(batch, group)] = resid
# 			self.orthog_feature[np.ix_(batch, group)] = A
# 		# return
# 		return self.lfo_resids

# 	def _compute_confidence_intervals(
# 		self, 
# 		alpha: int,
# 		nrand: int,
# 		features: Optional[np.array]=None,
# 		verbose: bool=True,
# 	):
# 		"""
# 		Computes underlying confidence intervals for :meth:`fit`.

# 		Parameters
# 		----------
# 		alpha : int
# 			Desired nominal Type I error rate.
# 		nrand : int
# 			Number of randomizations.
# 		features : np.array
# 			The list of features to compute CIs for. 
# 			Defaults to all features.
# 		verbose : bool
# 			If True, show progress bars.
# 		"""
# 		# process features
# 		if features is None:
# 			features = np.arange(self.n_cov).astype(int)
# 		if not utilities.haslength(features):
# 			features = np.array([features]).astype(int)

# 		# To save the confidence intervals
# 		columns = ['Estimate', 'Lower', 'Upper', 'p-value']
# 		self._summary = pd.DataFrame(
# 			np.zeros((len(features), 4)),
# 			index=features,
# 			columns=columns,
# 		)
# 		self._summary.values[:] = np.nan

# 		# Save the slopes and statistics
# 		self._null_estimates = np.zeros((self.n_cov, nrand))
# 		self._null_estimate_slopes = np.zeros((self.n_cov, nrand))


# 		# Loop through and compute
# 		ntiles = len(self.tiles)
# 		xinds = np.stack([np.arange(ntiles) for _ in range(nrand)])
# 		for j in utilities.vrange(len(features), verbose=verbose):
# 			feature = features[j]
# 			self._downdate_mosaic_residuals(feature=feature)
# 			## Todo: is this a bottleneck? If so, it is valid to move it outside the for loop
# 			flips = np.random.binomial(1, 0.5, size=(nrand, ntiles)).astype(int)
# 			# Observe test statistic + slope as beta changes
# 			stat = self.Zresid_sums[:, 0].sum()
# 			slope = self.ZA_sums[:, 0].sum()
# 			# Compute null test statistics + slopes as beta changes
# 			null_stats = self.Zresid_sums[(xinds, flips)].sum(axis=1)
# 			null_slopes = self.ZA_sums[(xinds, flips)].sum(axis=1)
# 			# Compute CI
# 			lower, upper = invert_linear_statistic(
# 				stat=stat, 
# 				slope=slope,
# 				null_stats=null_stats,
# 				null_slopes=null_slopes,
# 				alpha=alpha,
# 			)
# 			# Save
# 			self._null_estimates = null_stats / slope
# 			self._null_estimate_slopes = null_slopes / slope
# 			# Compute p-value
# 			pval = (1 + np.sum(np.abs(stat) <= np.abs(null_stats))) / (1 + nrand)
# 			# Estimate
# 			estimate = stat / slope
# 			# Save
# 			self._summary.loc[feature] = pd.Series(
# 				[estimate, lower, upper, pval],
# 				index=columns
# 			)

# 	def fit(
# 		self,
# 		nrand: int=10000,
# 		alpha: int=0.05,
# 		features: Optional[np.array]=None,
# 		verbose: bool=True,
# 	):
# 		"""
# 		Fits the linear model and returns confidence intervals.

# 		Parameters
# 		----------
# 		alpha : int
# 			Desired nominal Type I error rate.
# 		nrand : int
# 			Number of randomizations.
# 		features : np.array
# 			The list of features to compute CIs for. 
# 			Defaults to all features.
# 		verbose : bool
# 			If True, show progress bars.

# 		Returns
# 		-------
# 		self : object
# 		"""
# 		if verbose:
# 			print("Computing mosaic residuals.")
# 		self.compute_mosaic_residuals(verbose=verbose)
# 		if verbose:
# 			print("Computing confidence intervals.")
# 		self._compute_confidence_intervals(
# 			nrand=nrand, alpha=alpha, features=features, verbose=verbose
# 		)
# 		return self

# 	@property
# 	def summary(self):
# 		"""
# 		Produces a summary of key inferential results.
# 		"""
# 		return self._summary
