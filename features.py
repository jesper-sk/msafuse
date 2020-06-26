"""
Each feature must inherit from the base class :class:`msaf.base.Features` to be
included in the whole framework.

Here is a list of all the available features:

.. autosummary::
    :toctree: generated/

    CQT
    MFCC
    PCP
    Tonnetz
    Tempogram
    Features
"""

from builtins import super
import librosa
import numpy as np
import traceback

# Local stuff
from msaf import config
from msaf.base import Features
from msaf.exceptions import FeatureParamsError, NoFeaturesFileError

class Fuse(Features):
    """
    With this class, you can fuse multiple features 

    arrays:
        _features
        _framesync_features
        _est_beatsync_features
        _ann_beatsync_features
        _framesync_times
        _beatsync_times

    """
    @classmethod
    def is_fused(cls):
        return True

    def __init__(self, file_struct, feat_type, feat_list, sr=config.sample_rate,
                        hop_length=config.hop_size):

        #Init parent class
        super().__init__(file_struct=file_struct, sr=sr, hop_length=hop_length,
                        feat_type=feat_type)

        self._feature_classes = feat_list

    def compute_beat_sync_features(self, beat_frames, beat_times, pad):
        """Make the features beat-synchronous.

        Parameters
        ----------
        beat_frames: np.array
            The frame indeces of the beat positions.
        beat_times: np.array
            The time points of the beat positions (in seconds).
        pad: boolean
            If `True`, `beat_frames` is padded to span the full range.

        Returns
        -------
        beatsync_feats: np.array
            The beat-synchronized features.
            `None` if the beat_frames was `None`.
        beatsync_times: np.array
            The beat-synchronized times.
            `None` if the beat_frames was `None`.
        """
        if beat_frames is None:
            return None, None

        # Make the features beat-synchronous
        beatsync_feats = \
            [   librosa.util.utils.sync(feat.T,
                                        beat_frames, pad=pad).T
                for feat in self._framesync_features
            ]

        # Assign times (and add last time if padded)
        beatsync_times = \
                [   np.concatenate((np.copy(beat_times), self._framesync_times[-1])) if \
                    beat_times.shape[0] != feat.shape[0] else \
                    np.copy(beat_times) 
                    for feat in beatsync_feats
                ]

        return beatsync_feats, beatsync_times

    def _set_properties(self):
        self.dur = self._feature_classes[0].dur
        self._features = [feat._features for feat in self._feature_classes]
        self._framesync_features = [feat._framesync_features for feat in self._feature_classes]
        self._est_beatsync_features = [feat._est_beatsync_features for feat in self._feature_classes]
        self._ann_beatsync_features = [feat._ann_beatsync_features for feat in self._feature_classes]
        self._audio = self._feature_classes[0]._audio
        self._audio_harmonic = self._feature_classes[0]._audio_harmonic
        self._framesync_times = self._feature_classes[0]._framesync_times
        self._est_beatsync_times = self._feature_classes[0]._est_beatsync_times
        self._ann_beatsync_times = self._feature_classes[0]._ann_beatsync_times

    @property
    def features(self):
        """This getter will compute the actual features if they haven't
        been computed yet.

        Returns
        -------
        features: np.array
            The actual features. Each row corresponds to a feature vector.
        """
        # Compute features if needed
        if self._features is None:
            self._features = \
                [feat.features for feat in self._feature_classes]

            self._set_properties()

        return self._features

    @classmethod
    def get_id(self):
        return("fuse")

    def compute_features(self):
        raise NotImplementedError("This class doesn't have anythin to compute for itself")
        # return [feat.features for feat in self._feature_classes]

    def read_features(self):
        print("Attempting to read features from fuse...")
        # traceback.print_stack()
        #print("reading features", [feat.get_id() for feat in self._feature_classes])
        for feat in self._feature_classes:
            #print("feature", feat.get_id())
            #print("   Reading", feat.get_id())
            feat.read_features()

        self._set_properties()


    def write_features(self):
        print("Attempting to write features from fuse...")
        for feat in self._feature_classes:
            #print("writing", feat.get_id())
            feat.write_features()

    

        

class CQT(Features):
    """This class contains the implementation of the Constant-Q Transform.

    These features contain both harmonic and timbral content of the given
    audio signal.
    """
    def __init__(self, file_struct, feat_type, sr=config.sample_rate,
                 hop_length=config.hop_size, n_bins=config.cqt.bins,
                 norm=config.cqt.norm, filter_scale=config.cqt.filter_scale,
                 ref_power=config.cqt.ref_power):
        """Constructor of the class.

        Parameters
        ----------
        file_struct: `msaf.input_output.FileStruct`
            Object containing the file paths from where to extract/read
            the features.
        feat_type: `FeatureTypes`
            Enum containing the type of features.
        sr: int > 0
            Sampling rate for the analysis.
        hop_length: int > 0
            Hop size in frames for the analysis.
        n_bins: int > 0
            Number of frequency bins for the CQT.
        norm: float
            Type of norm to use for basis function normalization.
        filter_scale: float
            The scale of the filter for the CQT.
        ref_power: str
            The reference power for logarithmic scaling.
            See `configdefaults.py` for the possible values.
        """
        # Init the parent
        super().__init__(file_struct=file_struct, sr=sr, hop_length=hop_length,
                         feat_type=feat_type)
        # Init the CQT parameters
        self.n_bins = n_bins
        self.norm = norm
        self.filter_scale = filter_scale
        if ref_power == "max":
            self.ref_power = np.max
        elif ref_power == "min":
            self.ref_power = np.min
        elif ref_power == "median":
            self.ref_power = np.median
        else:
            raise FeatureParamsError("Wrong value for ref_power")

    @classmethod
    def get_id(self):
        """Identifier of these features."""
        return "cqt"

    def compute_features(self):
        """Actual implementation of the features.

        Returns
        -------
        cqt: np.array(N, F)
            The features, each row representing a feature vector for a give
            time frame/beat.
        """
        linear_cqt = np.abs(librosa.cqt(
            self._audio, sr=self.sr, hop_length=self.hop_length,
            n_bins=self.n_bins, norm=self.norm, filter_scale=self.filter_scale)
                            ) ** 2
        cqt = librosa.amplitude_to_db(linear_cqt, ref=self.ref_power).T
        return cqt


class MFCC(Features):
    """This class contains the implementation of the MFCC Features.

    The Mel-Frequency Cepstral Coefficients contain timbral content of a
    given audio signal.
    """
    def __init__(self, file_struct, feat_type, sr=config.sample_rate,
                 hop_length=config.hop_size, n_fft=config.n_fft,
                 n_mels=config.mfcc.n_mels, n_mfcc=config.mfcc.n_mfcc,
                 ref_power=config.mfcc.ref_power):
        """Constructor of the class.

        Parameters
        ----------
        file_struct: `msaf.input_output.FileStruct`
            Object containing the file paths from where to extract/read
            the features.
        feat_type: `FeatureTypes`
            Enum containing the type of features.
        sr: int > 0
            Sampling rate for the analysis.
        hop_length: int > 0
            Hop size in frames for the analysis.
        n_fft: int > 0
            Number of frames for the FFT.
        n_mels: int > 0
            Number of mel filters.
        n_mfcc: int > 0
            Number of mel coefficients.
        ref_power: function
            The reference power for logarithmic scaling.
        """
        # Init the parent
        super().__init__(file_struct=file_struct, sr=sr, hop_length=hop_length,
                         feat_type=feat_type)
        # Init the MFCC parameters
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        if ref_power == "max":
            self.ref_power = np.max
        elif ref_power == "min":
            self.ref_power = np.min
        elif ref_power == "median":
            self.ref_power = np.median
        else:
            raise FeatureParamsError("Wrong value for ref_power")

    @classmethod
    def get_id(self):
        """Identifier of these features."""
        return "mfcc"

    def compute_features(self):
        """Actual implementation of the features.

        Returns
        -------
        mfcc: np.array(N, F)
            The features, each row representing a feature vector for a give
            time frame/beat.
        """
        S = librosa.feature.melspectrogram(self._audio,
                                           sr=self.sr,
                                           n_fft=self.n_fft,
                                           hop_length=self.hop_length,
                                           n_mels=self.n_mels)
        log_S = librosa.amplitude_to_db(S, ref=self.ref_power)
        mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=self.n_mfcc).T
        return mfcc


class PCP(Features):
    """This class contains the implementation of the Pitch Class Profiles.

    The PCPs contain harmonic content of a given audio signal.
    """
    def __init__(self, file_struct, feat_type, sr=config.sample_rate,
                 hop_length=config.hop_size, n_bins=config.pcp.bins,
                 norm=config.pcp.norm, f_min=config.pcp.f_min,
                 n_octaves=config.pcp.n_octaves):
        """Constructor of the class.

        Parameters
        ----------
        file_struct: `msaf.input_output.FileStruct`
            Object containing the file paths from where to extract/read
            the features.
        feat_type: `FeatureTypes`
            Enum containing the type of features.
        sr: int > 0
            Sampling rate for the analysis.
        hop_length: int > 0
            Hop size in frames for the analysis.
        n_bins: int > 0
            Number of bins for the CQT computation.
        norm: int > 0
            Normalization parameter.
        f_min: float > 0
            Minimum frequency.
        n_octaves: int > 0
            Number of octaves.
        """
        # Init the parent
        super().__init__(file_struct=file_struct, sr=sr, hop_length=hop_length,
                         feat_type=feat_type)
        # Init the PCP parameters
        self.n_bins = n_bins
        self.norm = norm
        self.f_min = f_min
        self.n_octaves = n_octaves

    @classmethod
    def get_id(self):
        """Identifier of these features."""
        return "pcp"

    def compute_features(self):
        """Actual implementation of the features.

        Returns
        -------
        pcp: np.array(N, F)
            The features, each row representing a feature vector for a give
            time frame/beat.
        """
        audio_harmonic, _ = self.compute_HPSS()
        pcp_cqt = np.abs(librosa.hybrid_cqt(audio_harmonic,
                                            sr=self.sr,
                                            hop_length=self.hop_length,
                                            n_bins=self.n_bins,
                                            norm=self.norm,
                                            fmin=self.f_min)) ** 2
        pcp = librosa.feature.chroma_cqt(C=pcp_cqt,
                                         sr=self.sr,
                                         hop_length=self.hop_length,
                                         n_octaves=self.n_octaves,
                                         fmin=self.f_min).T
        return pcp


class Tonnetz(Features):
    """This class contains the implementation of the Tonal Centroids.

    The Tonal Centroids (or Tonnetz) contain harmonic content of a given audio
    signal.
    """
    def __init__(self, file_struct, feat_type, sr=config.sample_rate,
                 hop_length=config.hop_size, n_bins=config.tonnetz.bins,
                 norm=config.tonnetz.norm, f_min=config.tonnetz.f_min,
                 n_octaves=config.tonnetz.n_octaves):
        """Constructor of the class.

        Parameters
        ----------
        file_struct: `msaf.input_output.FileStruct`
            Object containing the file paths from where to extract/read
            the features.
        feat_type: `FeatureTypes`
            Enum containing the type of features.
        sr: int > 0
            Sampling rate for the analysis.
        hop_length: int > 0
            Hop size in frames for the analysis.
        n_bins: int > 0
            Number of bins for the CQT computation.
        norm: int > 0
            Normalization parameter.
        f_min: float > 0
            Minimum frequency.
        n_octaves: int > 0
            Number of octaves.
        """
        # Init the parent
        super().__init__(file_struct=file_struct, sr=sr, hop_length=hop_length,
                         feat_type=feat_type)
        # Init the local parameters
        self.n_bins = n_bins
        self.norm = norm
        self.f_min = f_min
        self.n_octaves = n_octaves

    @classmethod
    def get_id(self):
        """Identifier of these features."""
        return "tonnetz"

    def compute_features(self):
        """Actual implementation of the features.

        Returns
        -------
        tonnetz: np.array(N, F)
            The features, each row representing a feature vector for a give
            time frame/beat.
        """
        pcp = PCP(self.file_struct, self.feat_type, self.sr, self.hop_length,
                  self.n_bins, self.norm, self.f_min, self.n_octaves).features
        tonnetz = librosa.feature.tonnetz(chroma=pcp.T).T
        return tonnetz


class Tempogram(Features):
    """This class contains the implementation of the Tempogram feature.

    The Tempogram contains rhythmic content of a given audio signal.
    """
    def __init__(self, file_struct, feat_type, sr=config.sample_rate,
                 hop_length=config.hop_size,
                 win_length=config.tempogram.win_length):
        """Constructor of the class.

        Parameters
        ----------
        file_struct: `msaf.input_output.FileStruct`
            Object containing the file paths from where to extract/read
            the features.
        feat_type: `FeatureTypes`
            Enum containing the type of features.
        sr: int > 0
            Sampling rate for the analysis.
        hop_length: int > 0
            Hop size in frames for the analysis.
        win_length: int > 0
            The size of the window for the tempogram.
        """
        # Init the parent
        super().__init__(file_struct=file_struct, sr=sr, hop_length=hop_length,
                         feat_type=feat_type)
        # Init the local parameters
        self.win_length = win_length

    @classmethod
    def get_id(self):
        """Identifier of these features."""
        return "tempogram"

    def compute_features(self):
        """Actual implementation of the features.

        Returns
        -------
        tempogram: np.array(N, F)
            The features, each row representing a feature vector for a give
            time frame/beat.
        """
        return librosa.feature.tempogram(self._audio, sr=self.sr,
                                         hop_length=self.hop_length,
                                         win_length=self.win_length).T
