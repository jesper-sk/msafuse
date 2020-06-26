#!/usr/bin/env python
# coding: utf-8
import librosa
import logging
import numpy as np
from scipy.spatial import distance
from scipy import signal
from scipy.ndimage import filters
import pylab as plt
import timeit

import msaf
from msaf.algorithms.interface import SegmenterInterface


def median_filter(X, M=8):
    """Median filter along the first axis of the feature matrix X."""
    for i in range(X.shape[1]):
        X[:, i] = filters.median_filter(X[:, i], size=M)
    return X


def compute_gaussian_krnl(M):
    """Creates a gaussian kernel following Foote's paper."""
    g = signal.gaussian(M, M // 3., sym=True)
    G = np.dot(g.reshape(-1, 1), g.reshape(1, -1))
    G[M // 2:, :M // 2] = -G[M // 2:, :M // 2]
    G[:M // 2, M // 2:] = -G[:M // 2, M // 2:]
    return G


def compute_ssm(X, metric="seuclidean"):
    """Computes the self-similarity matrix of X."""
    D = distance.pdist(X, metric=metric)

    if np.any(np.isnan(D)):
        D = distance.pdist(X, metric="euclidean")

    D = distance.squareform(D)
    D /= D.max()

    # print(np.any(np.isnan(D)))

    return 1 - D


def compute_nc_gauss(X, G):
    """Computes the novelty curve from the self-similarity matrix X and
        the gaussian kernel G."""
    N = X.shape[0]
    M = G.shape[0]
    nc = np.zeros(N)

    for i in range(M // 2, N - M // 2 + 1):
        nc[i] = np.sum(X[i - M // 2:i + M // 2, i - M // 2:i + M // 2] * G)

    # Normalize
    # nc += nc.min()
    # nc /= nc.max()
    return nc

def compute_nc(X):
    """Computes the novelty curve from the structural features."""
    N = X.shape[0]
    # nc = np.sum(np.diff(X, axis=0), axis=1) # Difference between SF's

    nc = np.zeros(N)
    for i in range(N - 1):
        nc[i] = distance.euclidean(X[i, :], X[i + 1, :])

    # Normalize
    nc += np.abs(nc.min())
    nc /= float(nc.max())
    return nc


def pick_peaks(nc, L=16, offset_denom=0.1):
    """Obtain peaks from a novelty curve using an adaptive threshold."""
    offset = nc.mean() * float(offset_denom)

    nc = filters.gaussian_filter1d(nc, sigma=4)  # Smooth out nc

    th = filters.median_filter(nc, size=L) + offset
    #th = filters.gaussian_filter(nc, sigma=L/2., mode="nearest") + offset

    peaks = []
    for i in range(1, nc.shape[0] - 1):
        # is it a peak?
        if nc[i - 1] < nc[i] and nc[i] > nc[i + 1]:
            # is it above the threshold?
            if nc[i] > th[i]:
                peaks.append(i)
    #plt.plot(nc)
    #plt.plot(th)
    #for peak in peaks:
        #plt.axvline(peak)
    #plt.show()

    return peaks

def gaussian_filter(X, M=8, axis=0):
    """In place Gaussian filter along the first axis of the feature matrix X."""
    for i in range(X.shape[axis]):
        if axis == 1:
            X[:, i] = filters.gaussian_filter(X[:, i], sigma=M / 2.)
        elif axis == 0:
            X[i, :] = filters.gaussian_filter(X[i, :], sigma=M / 2.)

# def embedded_space(X, m, tau=1):
#     """Time-delay embedding with m dimensions and tau delays."""
#     # N is the width of the return matrix
#     N = X.shape[0] - int(np.ceil(m))
#     # Y is the return matrix of
#     Y = np.zeros((N, int(np.ceil(X.shape[1] * m))))
#     for i in range(N):
#         # print(X[i:i+m,:].flatten().shape, X.shape)
#         # print(Y[i,:].shape)
#         rem = int((m % 1) * X.shape[1])  # Reminder for float m
#         Y[i, :] = np.concatenate((X[i:i + int(m), :].flatten(),
#                                  X[i + int(m), :rem]))
#     return Y

def embedded_space(X, b, f):
    back = [np.roll(X, -(i+1), axis=0) for i in range(0,b)]
    front = [np.roll(X, i+1, axis=0) for i in range(0,f)]
    Y = np.hstack(tuple(back + [X] + front))
    return Y[b:-f,:] if f > 0 else Y[b:,:]

def imshow(F, title=None):
    plt.figure()
    plt.title(title)
    plt.imshow(F, interpolation="nearest", aspect="auto")

def plot(c, title=None):
    plt.figure()
    plt.title(title)
    plt.plot(c)

def circular_shift(X):
    """Shifts circularly the X squre matrix in order to get a
        time-lag matrix."""
    N = X.shape[0]
    L = np.zeros(X.shape)
    for i in range(N):
        L[i, :] = np.asarray([X[(i + j) % N, j] for j in range(N)])
    return L


class Segmenter(SegmenterInterface):

    def fuse(self, X):
        """
        Performs Network Similarity Fusion on the given input matrices.
        """
        #logging.info("Calculating affinity matrices...")

        F = len(X)

        P = []
        S = []

        T = self.config["T"]
        k = self.config["k_snf"]
        p_emb = self.config["embed_prev"]
        n_emb = self.config["embed_next"]

        norm = self.config["norm_type"]
        metric = self.config["ssm_metric"]

        for f in range(F):
            # plt.imshow(X[f], interpolation="nearest", aspect="auto"); plt.show()

            # Normalize and embed space
            X[f] = msaf.utils.normalize(X[f], norm_type=norm)
            #X[f] = embedded_space(X[f], 3)
            X[f] = embedded_space(X[f], p_emb, n_emb)

            # Compute self-similarity matrix
            D = compute_ssm(X[f], metric=metric)

            # imshow(D, ("SSM %s" % f))

            # plt.show()

            # Let N be array, such that N[j] is equal to the sum of
            # the k nearest neighbors of column D[:,j]. SSM is
            # symmetrical, therefore N[j] is also equal to
            # the sum of the k nearest neighbors of row D[j,:].
            N = np.partition( D - np.eye(D.shape[0])
                            , D.shape[0] - k
                            , axis = 1
                            ) [:,-k:] \
                                .sum(axis = 1)

            # Construct N, such that N[i,j] is equal to the sum of
            # the k nearest neighbors of row D[i,:], and column D[:,j]
            N = N[:,None] + N[None,:]

            # Construct bandwidth matrix
            σ = (D + N/k) / 6

            # Construct affinity matrix
            W = np.exp(-((D / σ)**2))

            # Create Pf
            Pf = W.copy()

            # Calculate case i!=j, divide Pf[i,j] by 2 * the sum
            # of row Pf[i,:] without Pf[i,j]
            np.fill_diagonal(Pf, 0)
            Pf = Pf / (2 * np.sum(Pf, axis=0)[:, None])

            # Add case i==j (diagonal), fixed 1/2
            np.fill_diagonal(Pf, 0.5)

            # Create Sf, based on W. We only need to fill in
            # S[i,j] where D[:,j] is one of the k nearest
            # neighbors of D[i,:].
            Sf = np.zeros_like(W)
            #Sf = np.eye(W.shape[0])

            # Get the indices of the k nearest neighbors, such
            # that ∀x ∈ N_ids[i,:], x is a neighbor of i as mea-
            # sured by D
            N_ids = \
                np.argpartition ( D - np.eye(D.shape[0])
                                , D.shape[0] - k
                                , axis=1
                                ) [:,-k:]

            W_c1 = np.take_along_axis(W, N_ids, axis=1)
            W_nsum = W_c1.sum(axis=1)[:, None]

            Sf_c1 = W_c1 / (2*W_nsum)

            # Take items from W according to indices of N_ids,
            # and summarize each row. Make it a olumn vector
            # such that it will broadcast over the y-axis.
            # W_nsum = \
            #     np.take_along_axis(W, N_ids, axis=1) \
            #         .sum(axis=1) \
            #             [: ,None]

            # Put these values in Sf corresponding to the
            # indices of N_ids. As W_nsum is a column vector,
            # it will be broadcasted against Sf.
            np.put_along_axis(Sf, N_ids, Sf_c1, axis=1)

            # Append resulting matrices to the lists
            P.append(Pf)
            S.append(Sf)

        # Iteration
        for t in range(T):

            Pn = P.copy()

            for f in range(F):

                M = np.sum(P[:f] + P[f+1:], axis=0)
                M = M / (F - 1)

                Pn[f] = S[f] @ M @ S[f].T

            P = Pn

        fuse = sum(P)
        fuse = fuse / F

        # Normalisation
        fuse = fuse + fuse.min()
        fuse = fuse / fuse.max()

        # imshow(fuse, "fuse")

        # plt.show(); raise Exception()

        return fuse

    def processFlat(self):

        Mp = self.config["Mp_adaptive"]   # Size of the adaptive threshold for
                                          # peak picking
        od = self.config["offset_denom"]  # Offset coefficient for adaptive
                                          # thresholding
        M = self.config["M_gaussian"]     # Size of gaussian kernel in beats
        M = M if M % 2 == 1 else M + 1

        X = self._preprocess(["fuse"])

        S = self.fuse(X)

        # imshow(S, "S")

        k = int(S.shape[0] * self.config["k_nearest"])

        R_ids = np.argpartition(S, S.shape[0] - k, axis=1) [:, -k:]

        R = np.zeros_like(S)

        np.put_along_axis(R, R_ids, np.ones_like(R_ids), axis=1)

        # imshow(R, "R")

        L = circular_shift(R)

        # imshow(L, "L")

        SF = L.T
        gaussian_filter(SF, M=M, axis=1)
        gaussian_filter(SF, M=1, axis=0)

        # imshow(SF, "SF")

        nc = compute_nc(SF)

        est_bounds = pick_peaks(nc, L=Mp, offset_denom=od)
        est_bounds = np.asarray(est_bounds) + self.config["embed_prev"] + 1

        est_idxs = np.concatenate(([0], est_bounds, [X[0].shape[0] - 1]))
        est_idxs = np.unique(est_idxs)

        assert est_idxs[0] == 0 and est_idxs[-1] == X[0].shape[0] - 1

        # Empty labels
        est_labels = np.ones(len(est_idxs) - 1) * - 1

        # Postprocess estimations
        est_idxs, est_labels = self._postprocess(est_idxs, est_labels)

        return est_idxs, est_labels


