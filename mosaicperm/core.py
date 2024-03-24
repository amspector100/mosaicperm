import numpy as np
from . import utilities

def compute_adaptive_pval(
	statistic, null_statistics
):
	"""
	Computes an adaptive p-value based on possibly many test statistics.

	Parameters
	----------
	statistic : float or np.array
		Either a float (one statistic) or 1D array (many test statistics).
	null_statistics : np.array
		(nrand, d)-shaped array of null statistics computed based on
		null permutations, where d is the dimension of ``statistic``.
	
	Returns
	-------
	pval : float
		The adaptive p-value.
	adapt_statistic : float
		The adaptive statistic value.
	null_adapt_statistics : np.array
		nrand-shaped array of null adaptive statistic values.
	"""
	# Dimensions
	nrand, d = null_statistics.shape
	if not utilities.haslength(statistic):
		statistic = np.array([statistic])
	# concatenate and take mean/sd
	all_stats = np.concatenate(
		[statistic.reshape(1, d), null_statistics], axis=0
	)
	mu0, sd0 = all_stats.mean(axis=0), all_stats.std(axis=0)
	sd0[sd0 == 0] = 1
	# t-statistic and take maximum
	tstats = (all_stats - mu0) / sd0
	adapt_stats = np.max(tstats, axis=1)
	# return pval
	pval = np.sum(adapt_stats[0] <= adapt_stats) / (nrand + 1)
	return pval, adapt_stats[0], adapt_stats[1:]