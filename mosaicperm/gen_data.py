import numpy as np

def gen_panel_data(
	n_subjects: int=50,
	n_obs: int=5, 
	n_cov: int=10,
	flat=True
):
	# Generate outcomes + subject/time markers
	outcomes = np.random.randn(n_obs, n_subjects)
	# generate covariates
	covariates = np.random.randn(n_obs, n_subjects, n_cov)
	if flat:
		return dict(
			outcomes=outcomes.flatten(order='C'),
			subjects=np.stack([np.arange(n_subjects) for _ in range(n_obs)], axis=0).flatten(order='C'),
			times=np.stack([np.arange(n_obs) for _ in range(n_subjects)], axis=1).flatten(order='C'),
			covariates=covariates.reshape(-1, n_cov, order='C')
		)
	else:
		return dict(
			outcomes=outcomes,
			covariates=covariates
		)