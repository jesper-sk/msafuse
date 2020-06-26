"""Config for the Foote algorithm."""

import msaf

# Foote Params
config = {
    "T" : 10,                           # Number of iterations of SNF
    "k_snf" : 3,                    # Number of nearest neighbors used in SNF
    "ssm_metric" : "seuclidean",         # Distance metric used for SSM computation
    "embed_next" : 1,                   # Amount of vectors from the past to embed
    "embed_prev" : 1,                   # Amount of vectors from the future to embed
    "norm_type": "min_max",             # min_max, log, np.inf,
                                        # -np.inf, float >= 0, None
    "M_gaussian": 66,
    "m_median": 12,
    "L_peaks": 66,
    "bound_norm_feats": "min_max"  # "min_max", "log", np.inf,
                                   # -np.inf, float >= 0, None

    # Framesync
    # "M_gaussian"    : msaf.utils.seconds_to_frames(28),
    # "m_median"      : msaf.utils.seconds_to_frames(12),
    # "L_peaks"       : msaf.utils.seconds_to_frames(18)
}

algo_id = "footesnf"
is_boundary_type = True
is_label_type = False