import os
import sys
import numpy as np
from scipy import stats
import unittest

# Add path to allow import of code
file_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.split(file_directory)[0]
sys.path.insert(0, os.path.abspath(parent_directory))

# Import the actual code
import mosaicperm

# for profiling
import inspect


def _create_exposures(n_obs, n_subjects, n_factors, starts=None, ends=None, nstart=None):
	"""
	Helper function to manufacture exposures for factor tests
	"""
	# Create locations at which exposures change
	if starts is None or ends is None:
		starts = np.sort(np.random.choice(n_obs-1, size=nstart, replace=False))
		starts[0] = 0
		ends = np.concatenate([starts[1:], [n_obs]])
	# Create exposures
	exposures = np.zeros((n_obs, n_subjects, n_factors))
	for start, end in zip(starts, ends):
		exposures[start:end] = np.random.randn(n_subjects, n_factors)
	return exposures

class MosaicTest(unittest.TestCase):

	def check_binary_pval_symmetric(self, pvals, alpha=0.001):
		reps = len(pvals)
		est = np.mean(pvals < 1)
		lower_ci = est + stats.norm.ppf(alpha) * np.sqrt(est * (1 - est) / reps)
		self.assertTrue(
			lower_ci < 0.5,
			f"binary p-vals are < 1 with probability {est} > 1/2, reps={reps}, lower_ci={lower_ci}"
		)

	def pearson_chi_square_test(
		self, counts, test_name, alpha=0.001, probs=None
	):
		"""
		uses pearson chi squared test to test multinomial dist
		"""
		# default is uniform distribution
		d = len(counts)
		if probs is None:
			probs = np.ones(d) / d
		# expected counts
		expected = np.sum(counts) * probs
		# chisquared stat + p-value
		chisquared = np.sum((counts - expected)**2 / expected) 
		pval = 1 - stats.chi2(df=d - 1).cdf(chisquared)
		# test
		self.assertTrue(
			pval >= alpha,
			f"For {test_name}, pearson chisquare pval ({pval}) < alpha ({alpha}), stat={chisquared}."
		)

def run_all_tests(test_classes):
	"""
	Usage: 
	context.run_all_tests(
		[TestClass(), TestClass2()]
	)
	This is useful for making pytest play nice with cprofilev.
	"""
	def is_test(method):
		return str(method).split(".")[1][0:4] == 'test'
	for c in test_classes:
		attrs = [getattr(c, name) for name in c.__dir__()]
		test_methods = [
			x for x in attrs if inspect.ismethod(x) and is_test(x)
		]
		for method in test_methods:
			method()