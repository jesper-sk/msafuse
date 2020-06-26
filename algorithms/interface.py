"""Interface for all the algorithms in MSAF."""
import numpy as np
import msaf.utils as U


class SegmenterInterface:
    """This class is an interface for all the segmenter algorithms included
    in MSAF. These segmenters must inherit from it and implement one of the
    following methods:
            processFlat()
            processHierarchical()

    Additionally, two private helper functions are provided:
        - preprocess
        - postprocess

    These are meant to do common tasks for all the segmenters and they
    should be called inside the process method if needed.

    All segmenters must return estimates times for the boundaries (est_times),
    and estimated labels (est_labels), **even if they can't compute them**.

    The following three types of algorithms with their behaviors are:
        - Computes boundaries and labels:
            If in_bound_times is None:
                Compute the est_times
            Else:
                Do not compute est_times, simply use in_bound_times instead
            If in_labels is None:
                Compute the est_labels
            Else:
                Do not compute est_labels, simply use in_labels instead

        - Computes boundaries only:
            Compute boundaries and return est_labels as None.

        - Computes labels only:
            Use in_bound_times in order to compute the labels.
            Return est_times as in_bound_times and the computed labels.

    In these cases, est_times or est_labels will be empty (None).
    """
    def __init__(self, file_struct, in_bound_idxs=None, feature="pcp",
                 annot_beats=False, framesync=False, features=None, 
                 feature_default=None, features_default=None, **config):
        """Inits the Segmenter.

        Parameters
        ----------
        file_struct: `msaf.io.FileStruct`
            Object with the file paths.
        in_bound_idxs: np.array
            Array containing the frame indeces of the previously find
            boundaries. `None` for computing them.
        feature: str
            Identifier of the features (e.g., pcp, mfcc)
        annot_beats: boolean
            Whether to use annotated beats or estimated ones.
        framesync: boolean
            Whether to use frame-synchronous or beat-synchronous features.
        features: dict
            Previously computed features. `None` for reading them.
        config: dict
            Configuration for the given algorithm (see module's __config.py__).
        """
        self.file_struct = file_struct
        self.audio_file = file_struct.audio_file
        self.in_bound_idxs = in_bound_idxs
        self.feature_str = feature if not isinstance(feature, list) else "fuse"
        self.feature_default_str = feature_default
        self.annot_beats = annot_beats
        self.framesync = framesync
        self.config = config
        self.features = features
        self.features_default = features_default

    def processFlat(self):
        """Main process to obtain the flat segmentation of a given track."""
        raise NotImplementedError("This method does not return flat "
                                  "segmentations.")

    def processHierarchical(self):
        """Main process to obtian the hierarchical segmentation of a given
        track."""
        raise NotImplementedError("This method does not return hierarchical "
                                  "segmentations.")

    def _preprocess(self, valid_features=["pcp", "tonnetz", "mfcc",
                                          "cqt", "tempogram"]):
        """This method obtains the actual features."""

        # print("(interface.py:92) featdefaultstr: %s" % self.feature_default_str)

        if self.feature_str in valid_features:
            features = self.features
        elif self.feature_default_str in valid_features:
            # print("(interface.py:95) fuse feature invalid, falling back to default")
            features = self.features_default
        else:
            raise RuntimeError("Feature %s in not valid for algorithm: %s "
                                "(valid features are %s)." %
                                (self.feature_str, __name__, valid_features))

        try:
            F = features.features
        except KeyError:
            raise RuntimeError("Feature %s in not supported by MSAF" %
                                   (features.get_id()))
        
        return F

        # if self.feature_str not in valid_features:
        #     if self.feature_str == "fuse" and self.config["feature_default"] in valid_features:
                
        #     else:
        #         raise RuntimeError("Feature %s in not valid for algorithm: %s "
        #                         "(valid features are %s)." %
        #                         (self.feature_str, __name__, valid_features))
        # else:
        #     try:
        #         F = self.features.features
        #     except KeyError:
        #         raise RuntimeError("Feature %s in not supported by MSAF" %
        #                            (self.feature_str))

        # return F

    def _postprocess(self, est_idxs, est_labels):
        """Post processes the estimations from the algorithm, removing empty
        segments and making sure the lenghts of the boundaries and labels
        match."""
        # Make sure we are using the previously input bounds, if any
        if self.in_bound_idxs is not None:
            F = self._preprocess()
            est_labels = U.synchronize_labels(self.in_bound_idxs, est_idxs,
                                              est_labels, F.shape[0])
            est_idxs = self.in_bound_idxs

        # Remove empty segments if needed
        est_idxs, est_labels = U.remove_empty_segments(est_idxs, est_labels)

        assert len(est_idxs) - 1 == len(est_labels), "Number of boundaries " \
            "(%d) and number of labels(%d) don't match" % (len(est_idxs),
                                                           len(est_labels))

        # Make sure the indeces are integers
        est_idxs = np.asarray(est_idxs, dtype=int)

        return est_idxs, est_labels
