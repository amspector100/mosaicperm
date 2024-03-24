import numpy as np
from typing import Optional

DEFAULT_QS = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])

def active_subset(
	residuals: np.array, 
	subset: Optional[np.array]=None,
	min_std: float=0
):
	"""
	Returns a subset of the subjects whose residual std is above
	min_std.

	Parameters
	----------
	residuals : np.array
		(n_obs, n_subjects) array of residuals
	subset : np.array
		Optional preformed subset; the output will be a subset
		of subset.
	min_std : float
		Minimum value of the standard deviation.

	"""
	if subset is None:
		subset = np.arange(subset.shape[1])

	# threshold and combine
	subset = set(subset.tolist()).intersection(
			set(list(np.where(residuals.std(axis=0) > min_std)[0]))
	)
	return np.array(list(subset)).astype(int)

def mean_maxcorr_stat(
	residuals: np.array, 
	**kwargs
):
	"""
	Mean maximum (absolute) correlation statistic.

	Parameters
	----------
	residuals : np.array
		(n_obs, n_subjects) array of residuals
	kwargs : np.array
		kwargs for ``active_subset`` preprocessing function.`
	"""
	subset = active_subset(residuals, **kwargs)
	# max correlation
	C = np.corrcoef(hateps[:, subset].T)
	C -= np.eye(len(subset))
	maxcorrs = np.abs(C).max(axis=0)
	# return mean max corr
	return np.mean(maxcorrs)

def quantile_maxcorr_stat(
	residuals: np.array, 
	qs: Optional[np.array]=DEFAULT_QS,
	**kwargs
):
	"""
	Quantiles of maximum (absolute) correlation statistic.

	Parameters
	----------
	residuals : np.array
		(n_obs, n_subjects) array of residuals
	qs : np.array
		Array of quantiles.
	kwargs : np.array
		kwargs for ``active_subset`` preprocessing function.`
	"""
	subset = active_subset(residuals, **kwargs)
	# max correlation
	C = np.corrcoef(hateps[:, subset].T)
	C -= np.eye(len(subset))
	maxcorrs = np.abs(C).max(axis=0)
	# return mean max corr
	return np.quantile(maxcorrs, qs)