import numpy as np
from . import tilings, utilities, core, statistics
from typing import Optional

def ols_residuals(
	outcomes: np.array,
	exposures: np.array,
):
	"""
	Computes OLS residuals based on cross-sectional regressions.

	Parameters
	----------
	outcomes : np.array
		(n_obs, n_subjects) array of outcome data.
	exposures : np.array
		(n_obs, n_subjects, n_factors) array of factor exposures
		OR 
		(n_subjects, n_factors) array of factor exposures if
		the exposures do not change with time.
	
	Returns
	-------
	residuals : np.array
		(n_obs, n_subjects) array of residuals from OLS
		cross-sectional regressions.
	"""
	# Case 1: exposures do not change with time
	if len(exposures.shape) == 2:
		Q, _ = np.linalg.qr(exposures)
		return outcomes - (Q @ (Q.T @ outcomes.T)).T
	# Case 2: exposures change with time
	elif len(exposures.shape) == 3:
		residuals = np.zeros(outcomes.shape)
		for i in range(len(residuals)):
			Q, _ = np.linalg.qr(exposures[i])
			residuals[i] = outcomes[i] - Q @ (Q.T @ outcomes[i])
		return residuals
	else:
		raise ValueError(f"Exposures must be a 2D or 3D array, but has shape {exposures.shape}")


class MosaicFactorTest(core.MosaicPermutationTest):

	def __init__(
		self,
		outcomes: np.array,
		exposures: np.array,
		test_stat: callable,
		test_stat_kwargs: Optional[dict]=None,
		tiles: Optional[list]=None, 
		**kwargs,
	):
		# Data
		self.outcomes = outcomes.copy()
		self.exposures = exposures.copy()
		# Remove nans
		self.n_obs, self.n_subjects = outcomes.shape
		if np.any(np.isnan(self.outcomes)):
			# in this case, must make exposures 3-dimensional since the nan
			# pattern will causes the exposures to change with time
			if len(self.exposures.shape) == 2:
				self.exposures = np.stack(
					[self.exposures for _ in range(self.n_obs)], axis=0
				)
			# fill with zeros (provably preserving validity)
			self.exposures[np.isnan(self.outcomes)] = 0
			self.outcomes[np.isnan(self.outcomes)] = 0
		# fill additional missing exposures with zero
		self.exposures[np.isnan(self.exposures)] = 0
		# Test statistic
		self.test_stat = test_stat
		self.tstat_kwargs = test_stat_kwargs
		if self.tstat_kwargs is None:
			self.tstat_kwargs = {}
		# Tiles
		self.tiles = tiles
		if self.tiles is None:
			self.tiles = tilings.default_factor_tiles(
				exposures=self.exposures,
				n_obs=len(self.outcomes),
				**kwargs
			)
		# initialize
		super().__init__()

	def compute_mosaic_residuals(self):
		"""
		Computes mosaic-style residual estimates.
		
		Returns
		-------
		residuals : np.array
			(n_obs, n_subjects)-shaped array of residual estimates.
		"""
		self.residuals = np.zeros(self.outcomes.shape)
		for tile in self.tiles:
			batch, group = tile
			# tile-specific loadings
			# Case 1: loadings do not change with time
			if len(self.exposures.shape) == 2:
				Ltile = self.exposures[group]
			# Case 2: loadings do change with time
			else:
				# Check exposures are constant within tile. Else, adjust
				constant = np.all([
					np.all(self.exposures[j, group] == self.exposures[batch[0], group]) 
					for j in batch
				])
				if constant:
					Ltile = self.exposures[batch[0], group]
				else:
					Ltile = np.concatenate(
						[self.exposures[j, group] for j in batch],
						axis=1
					)
					# for computational efficiency, get rid of duplicate columns
					Ltile = np.unique(Ltile, axis=1)

			# Compute residuals
			Ytile = self.outcomes[np.ix_(batch, group)] # len(batch) x len(group)
			Q, _ = np.linalg.qr(Ltile) # Q is len(group) x n_factors
			self.residuals[np.ix_(batch, group)] = (Ytile.T - Q @ (Q.T @ Ytile.T)).T

		# initialize null permutation object
		self._rtilde = self.residuals.copy()
		return self.residuals

class MosaicBCV(MosaicFactorTest):
	"""
	Mosaic factor test based on a split bi-cross validation statistic.

	Parameters
	----------
	TODO
	"""

	def __init__(
		self,
		new_exposures: np.array,
		*args, 
		**kwargs, 
	):
		super().__init__(
			*args, **kwargs, test_stat=None,
		)
		# Ensure a consistent input format
		if len(new_exposures.shape) == 1:
			new_exposures = new_exposures.reshape(1, -1)
		self.new_exposures = new_exposures

		# test statistic and kwargs
		self.test_stat = statistics.adaptive_mosaic_bcv_stat
		self.tstat_kwargs['tiles'] = self.tiles
		self.tstat_kwargs['new_exposures'] = self.new_exposures