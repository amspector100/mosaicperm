import numpy as np
from typing import Optional, Union

DEFAULT_QS = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
PCA_DEFAULT_QS = np.array([0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 1])

def active_subset(
	residuals: np.array, 
	subset: Optional[np.array]=None,
	min_std: float=0
) -> np.array:
	"""
	Returns subjects whose residual stdev is above ``min_std``.

	Parameters
	----------
	residuals : np.array
		(``n_obs``, ``n_subjects``) array of residuals
	subset : np.array
		Optional preformed subset; the output will be a subset
		of subset.
	min_std : float
		Minimum value of the standard deviation.

	"""
	if subset is None:
		subset = np.arange(residuals.shape[1])

	# threshold and combine
	subset = set(subset.tolist()).intersection(
			set(list(np.where(residuals.std(axis=0) > min_std)[0]))
	)
	return np.array(list(subset)).astype(int)

def mean_maxcorr_stat(
	residuals: np.array, 
	**kwargs
) -> float:
	"""
	Mean maximum (absolute) correlation statistic.

	Parameters
	----------
	residuals : np.array
		(n_obs, n_subjects) array of residuals
	kwargs : dict
		kwargs for ``active_subset`` preprocessing function.`
	"""
	subset = active_subset(residuals, **kwargs)
	# max correlation
	C = np.corrcoef(residuals[:, subset].T)
	C -= np.eye(len(subset))
	maxcorrs = np.abs(C).max(axis=0)
	# return mean max corr
	return np.mean(maxcorrs)

def quantile_maxcorr_stat(
	residuals: np.array, 
	qs: Optional[np.array]=DEFAULT_QS,
	**kwargs
) -> np.array:
	"""
	Quantiles of maximum (absolute) correlation statistic.

	Parameters
	----------
	residuals : np.array
		(n_obs, n_subjects) array of residuals
	qs : np.array
		Array of quantiles.
	kwargs : dict
		kwargs for ``active_subset`` preprocessing function.`
	"""
	subset = active_subset(residuals, **kwargs)
	# max correlation
	C = np.corrcoef(residuals[:, subset].T)
	C -= np.eye(len(subset))
	maxcorrs = np.abs(C).max(axis=0)
	# return mean max corr
	return np.quantile(maxcorrs, qs)

def _bcv_oos_resids(
	residuals: np.array, 
	new_exposure: np.array, 
	tiles: list, 
	mus: Optional[np.array]=None,
):
	"""
	Computes ``n_obs`` x ``n_subjects`` matrix of 
	out-of-sample residuals for the mosaic BCV stat. 
	"""
	n_obs, n_subjects = residuals.shape
	oos_resids = np.zeros(residuals.shape)
	new_exposure_l2 = np.sum(new_exposure**2)
	if mus is None:
		mus = residuals.mean(axis=0)
	# Loop through tiles
	for (batch, group) in tiles:
		# All subjects not in the group
		neggroup = np.ones(n_subjects, dtype=bool)
		neggroup[group] = False
		#Predict factor returns for new exposure
		hatZ = residuals[np.ix_(batch, neggroup)] @ new_exposure[neggroup]
		hatZ += mus[group] @ new_exposure[group]
		hatZ /= new_exposure_l2
		# Predict residuals for this tile
		preds = hatZ.reshape(-1, 1) * new_exposure[group].reshape(1, -1)
		oos_resids[np.ix_(batch, group)] = residuals[np.ix_(batch, group)] - preds

	return oos_resids




def mosaic_bcv_stat(
	residuals: np.array, 
	new_exposure: np.array, 
	tiles: list, 
	mus: Optional[np.array]=None,
) -> float:
	"""
	Out-of-sample R^2 measuring improvement for an
	augmented model containing an additional exposure.

	Parameters
	----------
	residuals : np.array
		(``n_obs``, ``n_subjects``) array of residuals
	new_exposure : np.array
		``n_subject``-length array of new exposures
	tiles : mosaicperm.tilings.Tiling
		The :class:`.Tiling` used to produce the mosaic
		residuals.
	mus : np.array
		Optional ``n_subjects``-length array of estimated
		means of residuals.

	Returns
	-------
	r2 : float
		Mosaic bi-cross validation out-of-sample R^2. 
	"""
	oos_resids = _bcv_oos_resids(residuals, new_exposure=new_exposure, tiles=tiles, mus=mus)
	# Return R^2
	return 1 - np.sum(oos_resids**2) / np.sum(residuals**2)

def adaptive_mosaic_bcv_stat(
	residuals: np.array,
	new_exposures: Union[np.array, list],
	tiles: list,
	mus: Optional[np.array]=None,
) -> np.array:
	"""
	Computes a mosaic BCV statistic for several augmented models.

	Parameters
	----------
	residuals : np.array
		(``n_obs``, ``n_subjects``) array of residuals
	new_exposures : np.array
		(``n_models``, ``n_subject``)-shaped array 
		such that ``new_exposures[i]`` is an array
		of new exposures.	
	tiles : mosaicperm.tilings.Tiling
		The :class:`.Tiling` used to produce the mosaic
		residuals.
	mus : np.array
		Optional n_subjects-length array of estimated means
		of residuals.

	Returns
	-------
	r2 : np.array
		Array of mosaic bi-cross validation empirical R^2 values. 
	"""
	n_models = len(new_exposures)
	r2s = np.zeros(n_models)
	# Loop through the models and compute r2s
	for i in range(n_models):
		r2s[i] = mosaic_bcv_stat(
			residuals=residuals, 
			new_exposure=new_exposures[i],
			tiles=tiles,
			mus=mus,
		)
	return r2s

def approximate_sparse_pcas(
	Sigma: np.array,
	quantiles: Optional[np.array]=PCA_DEFAULT_QS
) -> np.array:
	"""
	Performs approximate sparse pca.

	Parameters
	----------
	Sigma : np.array
		p x p covariance matrix.
	quantiles : np.array
		Array of quantiles between zero and one. 

	Returns
	-------
	evecs : list of np.arrays
		evecs[i] is a p-length approximate sparse eigenvector
		with np.round(quantiles[i] * p) nonzero entries.
	"""
	p_orig = len(Sigma)
	# ignore assets with zero variance
	subset = np.where(np.diag(Sigma) > 1e-8)[0]
	p = len(subset)
	Sig = Sigma[np.ix_(subset, subset)]
	scale = np.sqrt(np.diag(Sig))
	# Correlation matrix
	C = Sig / np.outer(scale, scale)
	absC = np.abs(C - np.eye(len(subset)))
	macs = np.max(absC, axis=0)
	inds = np.argsort(-1*macs)
	# find subset of Sigma
	ks = np.maximum(np.minimum(
		np.round(quantiles * p).astype(int), p
	), 2)
	evecs = []
	for k in ks:
		# form submatrix
		indsk = inds[0:k]
		subC = C[np.ix_(indsk, indsk)]
		# maximum eigenvalue of submatrix
		subeig = np.linalg.eigh(subC)[1][:, -1]
		# create final eigenvalue
		evec = np.zeros((p, 1))
		evec[indsk] = subeig.reshape(-1, 1)
		evec_full = np.zeros((p_orig, 1))
		evec_full[subset] = evec * scale.reshape(-1, 1)
		evec_full /= np.sqrt(np.power(evec_full, 2).sum())
		evecs.append(evec_full.flatten())
	# return
	return np.stack(evecs, axis=0)