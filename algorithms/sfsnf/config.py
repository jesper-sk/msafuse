"""Config for the Structural Features with Similarity Network Fusion algorithm."""

import numpy as np

# Foote Params
config = {

    # SNF params

    "T" : 10,                           # Number of iterations of SNF
    "k_snf" : 3,                    # Number of nearest neighbors used in SNF
    "ssm_metric" : "seuclidean",         # Distance metric used for SSM computation
    "embed_next" : 4,                   # Amount of vectors from the past to embed
    "embed_prev" : 1,                   # Amount of vectors from the future to embed
    "norm_type": "min_max",             # min_max, log, np.inf,
                                        # -np.inf, float >= 0, None

    "thresh" : 0.4,

    "k_nearest" : 0.05,
    "M_gaussian" : 27,
    "Mp_adaptive" : 28,
    "offset_denom": 0.05
}

algo_id = "sfsnf"
is_boundary_type = True
is_label_type = False