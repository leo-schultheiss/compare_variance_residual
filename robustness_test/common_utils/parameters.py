import himalaya
import numpy as np

GROUP_CV_SOVER_PARAMS = dict(n_iter=1, n_targets_batch=100, n_targets_batch_refit=100, alphas=np.logspace(0, 4, 10),
                             score_func=himalaya.scoring.correlation_score, progress_bar=True)
