#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WANN-DPC: Density Peaks Finding Clustering Based on Weighted Adaptive
Nearest Neighbors
=======================================================================

Overview
--------
This module implements **WANN-DPC**, a density-peak-based clustering
algorithm that improves upon ANN-DPC (Adaptive Nearest Neighbor DPC) by
introducing a *weighted* local density definition and a correction-factor
driven cluster-center selection strategy. WANN-DPC addresses two key
limitations of DPC and its variants: (1) the inability to simultaneously
identify cluster centers of dense and sparse clusters, and (2) the
"Domino Effect" caused by a single mis-assigned point propagating errors
throughout the clustering process. The key contributions realized by
this implementation are:

    1. A novel weighted local density of a point, defined by jointly
       weighting its close neighbors and far neighbors (as opposed to
       ANN-DPC, which only considers a point's adaptive/close neighbors
       and therefore introduces locality bias).
    2. A correction factor that is incorporated into the decision value
       to detect cluster centers of dense and sparse clusters in turn.
    3. A two-step (three-stage) label assignment strategy that combines
       close-neighbor propagation, weighted nearest-neighbor membership
       degrees, and a DPC-style parent-chain fallback, so that every
       sample is eventually assigned to a cluster.

The overall clustering pipeline consists of the following stages:

    1. Feature normalization and pairwise distance matrix construction.
    2. Nearest-neighbor ordering (index and distance) computation.
    3. Weighted local density estimation via adaptive near/far neighbor
       splitting (maximum-gap cutoff).
    4. Relative distance and parent-object computation for every sample.
    5. Cluster-center detection via a correction-factor-weighted decision
       value, together with close-neighbor label propagation.
    6. Label allocation strategy 1: close-neighbor-relationship-based
       propagation to build a preliminary clustering skeleton.
    7. Label allocation strategy 2: weighted nearest-neighbor membership
       degree propagation for the remaining unlabeled samples.
    8. Label allocation strategy 3: DPC-style parent-chain fallback
       label assignment for any residual unlabeled samples.

Author Information
-------------------
    Chinese Name : 严欢 (YAN Huan)
    English Name : hwanyan
    Email        : yan-huan@snnu.edu.cn
    ORCID        : https://orcid.org/0009-0000-0016-6324
    Affiliations :
        1. Tencent Cloud Computing (Chongqing) Co., Ltd.
        2. Shaanxi Normal University

How to Cite
-----------
If this implementation is useful for your research, please cite the
corresponding paper:

    Xie, J., Yan, H., Wang, M., Grant, P. W., & Pedrycz, W. (2026).
    WANN-DPC: Density peaks finding clustering based on Weighted
    Adaptive Nearest Neighbors. Pattern Recognition, 170, 111953.
    https://doi.org/10.1016/j.patcog.2025.111953

    BibTeX::

        @article{xie2026wanndpc,
            title   = {WANN-DPC: Density peaks finding clustering based
                        on Weighted Adaptive Nearest Neighbors},
            author  = {Xie, Juanying and Yan, Huan and Wang, Mingzhao and
                        Grant, Philip W. and Pedrycz, Witold},
            journal = {Pattern Recognition},
            volume  = {170},
            pages   = {111953},
            year    = {2026},
            doi     = {10.1016/j.patcog.2025.111953}
        }

License
-------
Released under the MIT License. See the ``LICENSE`` file in this
repository for full terms of use.

Dependencies
------------
- numpy
- scikit-learn (``sklearn.preprocessing.MinMaxScaler``)
"""

import numpy as np
from queue import Queue
from sklearn.preprocessing import MinMaxScaler


class WANNDPC:
    """WANN-DPC: Density Peaks Clustering Based on Weighted Adaptive Nearest Neighbors.

    The class exposes a scikit-learn-like workflow: instantiate with the
    feature matrix and the target number of clusters, then call
    :meth:`fit` to run the full clustering pipeline. After fitting,
    cluster assignments are available via :attr:`Labels` and the indices
    of the discovered cluster centers via :attr:`ClusterCenters`.

    Attributes
    ----------
    X : numpy.ndarray of shape (Size, Features)
        Feature matrix of the dataset. Overwritten in-place with its
        min-max normalized version once :meth:`fit` is executed.
    NC : int
        User-specified (preset) number of clusters to be detected.
    Size : int
        Number of samples in the dataset.
    Features : int
        Number of features (dimensionality) of the dataset.
    CF : float
        Correction factor used to weight the dependency (distance to the
        nearest higher-density labeled sample) term when selecting
        subsequent cluster centers.
    NeighborLimit : int
        Upper bound ``T`` on the number of nearest neighbors considered
        when estimating the weighted local density of a point, computed
        adaptively as ``min(0.1 * Size, 10 + Features * log(Features + 1),
        NeighborLimit)`` to balance accuracy and computational cost.
    DM : numpy.ndarray of shape (Size, Size)
        Pairwise Euclidean distance matrix.
    NNMatrix : numpy.ndarray of shape (Size, Size - 1)
        Neighbor index matrix; for each sample, the indices of all other
        samples sorted in ascending order of distance.
    NNDMatrix : numpy.ndarray of shape (Size, Size - 1)
        Neighbor distance matrix; the sorted counterpart of ``NNMatrix``.
    CutoffVector : list of int
        For each sample, the number of "close" neighbors determined by
        the maximum-gap split within its ``NeighborLimit`` nearest
        neighbors; used as the adaptive near/far neighbor boundary.
    DensityVector : numpy.ndarray of shape (Size,)
        Weighted local density of every sample.
    RelativeDistanceVector : numpy.ndarray of shape (Size,)
        Relative distance of every sample (distance to the nearest
        sample of strictly higher density).
    ObjectVector : numpy.ndarray of shape (Size,)
        For each sample, the index of its "parent" (object) sample, i.e.,
        the nearest sample of strictly higher density.
    DecisionValueVector : numpy.ndarray of shape (Size,)
        Correction-factor-weighted decision value used to select the
        next cluster center (``density * relative distance *
        dependency``).
    ClusterCenters : numpy.ndarray of shape (NC,)
        Indices of the samples selected as cluster centers.
    Labels : numpy.ndarray of shape (Size,)
        Predicted cluster label for each sample (``0`` denotes an
        as-yet-unassigned sample; final clusters are labeled from ``1``).
    INearstNeighborDic : dict of int -> list of int
        Reverse nearest-neighbor dictionary: for each sample, the list
        of samples that regard it as one of their close neighbors.
    """

    def __init__(self, X, clusterNumber, Correction=2.5, NeighborLimit=30):
        """Initialize the WANN-DPC estimator.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The feature matrix of the dataset to be clustered.
        clusterNumber : int
            The target (preset) number of clusters to detect.
        Correction : float, optional
            Correction factor applied to the dependency term during
            cluster-center selection. Defaults to ``2.5``.
        NeighborLimit : int, optional
            Upper bound on the number of nearest neighbors considered
            when estimating the weighted local density of a point.
            Defaults to ``30``.
        """
        # Feature matrix of the dataset
        self.X = X
        # Preset (target) number of clusters
        self.NC = clusterNumber
        # Number of samples in the dataset
        self.Size = len(X)
        # Number of features in the dataset
        self.Features = X.shape[1]
        # Correction factor
        self.CF = Correction
        # Adaptive upper bound on the number of neighbors considered per sample
        self.NeighborLimit = int(min(0.1 * self.Size,
                                      10 + self.Features * np.log(self.Features + 1),
                                      NeighborLimit))

        # The following attributes are all empty before clustering is performed

        # Distance matrix of the samples
        self.DM = []
        # Neighbor index matrix of the samples
        self.NNMatrix = []
        # Neighbor distance matrix of the samples
        self.NNDMatrix = []
        # Number of close (dense-side) neighbors at the maximum-gap cutoff of each sample
        self.CutoffVector = []
        # Weighted local density of each sample
        self.DensityVector = []
        # Indices of the selected cluster centers
        self.ClusterCenters = []
        # Predicted cluster label of each sample
        self.Labels = []
        # Reverse nearest-neighbor dictionary of the samples
        self.INearstNeighborDic = {}

    def __Normalization(self):
        """Standardize the feature matrix via min-max scaling.

        Returns
        -------
        None
            Overwrites ``self.X`` in place with every column linearly
            rescaled to the ``[0, 1]`` range.
        """
        # Instantiate the transformer (feature_range defines the target
        # normalization interval, i.e., [minimum, maximum])
        transfer = MinMaxScaler(feature_range=(0, 1))
        self.X = transfer.fit_transform(self.X)

    def __getDistanceMatrix(self):
        """Compute the pairwise distance matrix of the dataset.

        The pairwise Euclidean distance is computed for every pair of
        samples in the (already normalized) feature matrix.

        Returns
        -------
        None
            Populates ``self.DM`` in place.
        """
        for i in range(self.Size):
            tmpList = []
            for j in range(self.Size):
                if i == j:
                    tmpList.append(0)
                else:
                    tmpList.append(np.sum(np.power(self.X[i] - self.X[j], 2)) ** 0.5)
            self.DM.append(tmpList)
        self.DM = np.array(self.DM, dtype=float)

    def __getNeighborInfomation(self):
        """Derive nearest-neighbor ordering information from the distance matrix.

        Builds, for every sample:
            1. The neighbor index matrix, i.e., the indices of all other
               samples sorted in ascending order of distance.
            2. The corresponding sorted neighbor distance matrix.

        Returns
        -------
        None
            Populates ``self.NNMatrix`` and ``self.NNDMatrix`` in place.
        """
        # Neighbor index matrix of the samples, sorted by ascending distance
        self.NNMatrix = np.argsort(self.DM)[:, 1:]
        # Neighbor distance matrix of the samples, sorted by ascending distance
        self.NNDMatrix = np.sort(self.DM)[:, 1:]

    def __getWeightDensity(self):
        """Compute the weighted local density of every sample.

        For each sample, its ``NeighborLimit`` nearest-neighbor distances
        are scanned to locate the maximum gap between two consecutive
        neighbor distances; this gap position adaptively splits the
        neighbors into a "close" (dense-side) group and a "far"
        (sparse-side) group. The weighted local density then combines
        the mean close-neighbor similarity and the mean far-neighbor
        similarity, weighted by the relative sizes of the two groups, so
        that both nearby and distant neighborhood structure contribute
        to the estimated density (mitigating the locality bias of using
        only adaptive/close neighbors as in ANN-DPC).

        Returns
        -------
        None
            Populates ``self.CutoffVector`` and ``self.DensityVector`` in
            place.
        """
        # Construct, for each sample, the array of consecutive gaps
        # between its sorted nearest-neighbor distances
        gapMatrix = []
        for i in range(self.NeighborLimit - 1):
            gapMatrix.append(self.NNDMatrix[:, i + 1] - self.NNDMatrix[:, i])
        gapMatrix = np.array(gapMatrix)
        gapMatrix = gapMatrix.T
        # Locate the maximum-gap cutoff position for every sample
        for index in range(self.Size):
            maxGapIndex = np.argmax(gapMatrix[index])
            # Record the number of close (dense-side) neighbors of the current sample
            self.CutoffVector.append(maxGapIndex + 1)
            # Mean distance among the close (dense-side) neighbors
            nearMeanDis = np.mean(self.NNDMatrix[index][:maxGapIndex + 1])
            # Mean distance among the far (sparse-side) neighbors
            farMeanDis = np.mean(self.NNDMatrix[index][maxGapIndex + 1:self.NeighborLimit + 1])
            # Compute the weighted local density of the current sample by
            # combining close-neighbor and far-neighbor similarity,
            # weighted by their respective group proportions
            self.DensityVector.append(
                1 / (nearMeanDis + 1) * (maxGapIndex + 1) / self.NeighborLimit +
                1 / (farMeanDis + 1) * (self.NeighborLimit - maxGapIndex - 1) / self.NeighborLimit)
        self.DensityVector = np.array(self.DensityVector)

    def __getInverseNearstNeighbor(self):
        """Build the reverse nearest-neighbor dictionary of the samples.

        For every sample, records the set of other samples that include
        it among their ``CutoffVector`` close neighbors, i.e., the
        reverse (incoming) close-neighbor relationship.

        Returns
        -------
        None
            Populates ``self.INearstNeighborDic`` in place.
        """
        for index in range(self.Size):
            self.INearstNeighborDic[index] = []
        for index in range(self.Size):
            for neighbor in self.NNMatrix[index][:self.CutoffVector[index]]:
                if index not in self.INearstNeighborDic[neighbor]:
                    self.INearstNeighborDic[neighbor].append(index)

    def __getRelativeDistance(self):
        """Compute the relative distance and parent (object) sample for every sample.

        Samples are processed in descending order of local density. The
        globally densest sample (necessarily a cluster-center candidate)
        is assigned the maximum pairwise distance in the dataset as its
        relative distance. Every other sample's relative distance is
        defined as its minimum distance to any sample with strictly
        higher density that has already been processed; the
        corresponding nearest higher-density sample is recorded as its
        "object" (parent) sample.

        Returns
        -------
        None
            Populates ``self.RelativeDistanceVector`` and
            ``self.ObjectVector`` in place.
        """
        # Sort samples by density in descending order
        sortDensityIndex = np.argsort(self.DensityVector)[::-1]
        # Initialize the relative distance vector
        self.RelativeDistanceVector = np.zeros(self.Size, dtype=float)
        # The sample with the highest density (necessarily a cluster
        # center) is assigned the maximum pairwise distance as its
        # relative distance
        self.RelativeDistanceVector[sortDensityIndex[0]] = np.max(self.DM)
        # Initialize the object (parent) vector
        self.ObjectVector = np.zeros(self.Size, dtype=int)
        # Compute the relative distance and object sample for all others
        for i in range(1, self.Size):
            # The outer loop index i denotes how many higher-density
            # samples have been processed so far, scanned in descending
            # density order
            tempIndex = sortDensityIndex[i]
            # The object (parent) sample is the nearest sample among all
            # higher-density samples processed so far
            self.ObjectVector[tempIndex] = sortDensityIndex[
                np.argmin(self.DM[tempIndex][sortDensityIndex[:i]])]
            # The relative distance is the distance to that object (parent) sample
            self.RelativeDistanceVector[tempIndex] = self.DM[tempIndex][self.ObjectVector[tempIndex]]

    def __getClusterCenter(self):
        """Select cluster centers via a correction-factor-weighted decision value.

        The sample with the highest density is always selected as the
        first cluster center. Its close neighbors are then propagated
        breadth-first (via a queue), with a sample being absorbed into
        the current cluster when both:
            1. its distance to the propagating sample does not exceed
               the mean close-neighbor distance of the current cluster
               center, and
            2. its density is not lower than the minimum close-neighbor
               density of the current cluster center.

        For each subsequent cluster center (up to ``self.NC`` in total),
        a *dependency* term ``exp(CF * distance_to_nearest_labeled)`` is
        combined multiplicatively with density and relative distance to
        form the decision value; the highest-scoring unlabeled sample is
        selected as the next cluster center, and the same close-neighbor
        propagation procedure is repeated. This correction factor allows
        WANN-DPC to detect cluster centers of both dense and sparse
        clusters in turn.

        Returns
        -------
        None
            Populates ``self.Labels`` and ``self.ClusterCenters`` in
            place.
        """
        # Initialize the label vector
        self.Labels = np.zeros(self.Size, dtype=int)
        # The sample with the highest density is necessarily a cluster center
        maxDensityIndex = np.argmax(self.DensityVector)
        self.ClusterCenters.append(maxDensityIndex)
        self.Labels[maxDensityIndex] = 1
        # Extract the close-neighbor set of the cluster center
        DenseNeighbors = self.NNMatrix[maxDensityIndex][:self.CutoffVector[maxDensityIndex]]
        # Define a queue to propagate labels through close-neighbor relationships.
        # The conditions for label propagation are:
        #   1. The distance between propagating samples does not exceed
        #      the mean close-neighbor distance at the cluster center.
        #   2. The density of the receiving sample is not lower than the
        #      minimum close-neighbor density at the cluster center.
        q = Queue()
        # Perform preliminary label propagation to the close neighbors of the center
        for neighbor in DenseNeighbors:
            self.Labels[neighbor] = self.Labels[maxDensityIndex]
            q.put(neighbor)
        # Mean distance among the close neighbors of the cluster center
        meanDis = np.mean(self.DM[maxDensityIndex][DenseNeighbors])
        # Minimum density among the close neighbors of the cluster center
        minDes = np.min(self.DensityVector[DenseNeighbors])
        # Breadth-first propagation of labels through close, sufficiently
        # dense neighboring samples
        while not q.empty():
            head = q.get()
            for neighbor in self.NNMatrix[head][:self.CutoffVector[head]]:
                if self.Labels[neighbor] == 0 and self.DM[head][neighbor] <= meanDis \
                        and self.DensityVector[neighbor] >= minDes:
                    self.Labels[neighbor] = self.Labels[head]
                    q.put(neighbor)
        # Collect all samples that have already been assigned a label
        Allocationed = np.arange(self.Size)[self.Labels != 0]
        # Dependency vector: distance from every sample to its nearest labeled sample
        NCCDistanceVector = self.DM[maxDensityIndex]
        for index in range(self.Size):
            if self.Labels[index] == 0:
                NCCDistanceVector[index] = min(NCCDistanceVector[index],
                                                np.min(self.DM[index][Allocationed]))
            else:
                NCCDistanceVector[index] = 0
        # Select the remaining cluster centers one by one, introducing the
        # correction-factor-weighted dependency term at each iteration
        Cluster = 2
        while Cluster <= self.NC:
            # Compute the correction-factor-weighted dependency term
            DependencyVector = np.exp(self.CF * NCCDistanceVector)
            for index in range(self.Size):
                if self.Labels[index] != 0:
                    DependencyVector[index] = 0
            # Decision value = density * relative distance * dependency
            self.DecisionValueVector = np.multiply(
                np.multiply(self.DensityVector, self.RelativeDistanceVector), DependencyVector)
            maxDecisionIndex = np.argmax(self.DecisionValueVector)
            self.ClusterCenters.append(maxDecisionIndex)
            self.Labels[maxDecisionIndex] = Cluster
            # Extract the close-neighbor set of the new cluster center
            DenseNeighbors = self.NNMatrix[maxDecisionIndex][:self.CutoffVector[maxDecisionIndex]]
            # Perform preliminary label propagation to the close neighbors of the new center
            for neighbor in DenseNeighbors:
                self.Labels[neighbor] = self.Labels[maxDecisionIndex]
                q.put(neighbor)
            # Mean distance among the close neighbors of the new cluster center
            meanDis = np.mean(self.DM[maxDecisionIndex][DenseNeighbors])
            # Minimum density among the close neighbors of the new cluster center
            minDes = np.min(self.DensityVector[DenseNeighbors])
            # Breadth-first propagation of labels through close, sufficiently dense neighbors
            while not q.empty():
                head = q.get()
                for neighbor in self.NNMatrix[head][:self.CutoffVector[head]]:
                    if self.Labels[neighbor] == 0 and self.DM[head][neighbor] <= meanDis \
                            and self.DensityVector[neighbor] >= minDes:
                        self.Labels[neighbor] = self.Labels[head]
                        q.put(neighbor)
            # Collect all samples that have already been assigned a label
            Allocationed = np.arange(self.Size)[self.Labels != 0]
            # Update the dependency distance of every unlabeled sample to
            # its nearest labeled sample
            for index in range(self.Size):
                if self.Labels[index] == 0:
                    NCCDistanceVector[index] = min(NCCDistanceVector[index],
                                                    np.min(self.DM[index][Allocationed]))
                else:
                    NCCDistanceVector[index] = 0
            # Advance to the next cluster
            Cluster += 1
        self.ClusterCenters = np.array(self.ClusterCenters)

    def __LabelAllocation1(self):
        """Label allocation strategy 1: close-neighbor-relationship-based propagation.

        Establishes a preliminary clustering skeleton by propagating
        cluster-center labels to their surrounding lower-density close
        neighbors. The propagation conditions are:
            1. The receiving sample has not yet been assigned a label.
            2. Its distance to the propagating sample does not exceed
               the propagating sample's mean close-neighbor distance.
            3. Its density is not lower than the dataset's mean density.

        Samples that are close to an already-labeled sample but whose
        density falls below the mean density are collected as a
        secondary set; the ``CutoffVector`` (close-neighbor count) of
        their corresponding propagating samples is then expanded
        according to how often they were encountered, and labels are
        finally propagated once more from this secondary set to any
        remaining unlabeled neighbors.

        Returns
        -------
        None
            Updates ``self.Labels`` and ``self.CutoffVector`` in place.
        """
        # Mean number of close neighbors across all samples
        meanNeighbors = int(np.mean(self.CutoffVector))
        # Expand the close-neighbor count of every sample whose count
        # falls below the mean, to widen the propagation range
        for index in range(self.Size):
            if self.CutoffVector[index] < meanNeighbors:
                self.CutoffVector[index] += meanNeighbors

        # Mean density across all samples
        meanDense = np.mean(self.DensityVector)
        # Uniformly propagate the cluster-center label to all of its
        # lower-density close neighbors
        for center in self.ClusterCenters:
            for neighbor in self.NNMatrix[center][:self.CutoffVector[center]]:
                if self.Labels[neighbor] == 0 and self.DensityVector[neighbor] < self.DensityVector[center]:
                    self.Labels[neighbor] = self.Labels[center]
        # Queue used to propagate labels from already-labeled samples
        q = Queue()
        # Mean close-neighbor distance of every sample
        DenseNeighborMD = []
        # Enqueue all currently labeled samples and compute the mean
        # close-neighbor distance of every sample
        for index in range(self.Size):
            DenseNeighborMD.append(np.mean(self.NNDMatrix[index][:self.CutoffVector[index]]))
            if self.Labels[index] != 0:
                q.put(index)
        # While propagating labels through close-neighbor relationships,
        # collect low-density neighboring samples that remain unassigned
        secondary = []
        while not q.empty():
            head = q.get()
            for neighbor in self.NNMatrix[head][:self.CutoffVector[head]]:
                if self.Labels[neighbor] == 0 and self.DM[head][neighbor] <= DenseNeighborMD[
                        neighbor] and self.DensityVector[neighbor] >= meanDense:
                    self.Labels[neighbor] = self.Labels[head]
                    q.put(neighbor)
                if self.Labels[neighbor] == 0 and self.DM[head][neighbor] <= DenseNeighborMD[
                        neighbor] and self.DensityVector[neighbor] < meanDense:
                    secondary.append(head)
        # Count how many times each propagating sample was blocked by a
        # low-density unassigned neighbor
        cnt_dic = {}
        for index in secondary:
            if index not in cnt_dic:
                cnt_dic[index] = 1
            else:
                cnt_dic[index] += 1
        # Widen the close-neighbor range of those propagating samples accordingly
        for index in cnt_dic:
            self.CutoffVector[index] += cnt_dic[index]
        secondary = list(set(secondary))
        # Re-attempt label propagation from the secondary set to any
        # still-unassigned neighboring samples
        for index in secondary:
            for neighbor in self.NNMatrix[index][:self.CutoffVector[index]]:
                if self.Labels[neighbor] == 0:
                    self.Labels[neighbor] = self.Labels[index]

    def __LabelAllocation2(self):
        """Label allocation strategy 2: weighted nearest-neighbor membership propagation.

        For every sample that remains unlabeled after
        :meth:`__LabelAllocation1`, a weighted similarity to each
        existing cluster is computed from its close neighbors (grouped
        by cluster label, with per-cluster mean distance converted to a
        similarity score via a negative exponential and further weighted
        by the neighbor-count proportion). Samples are then greedily
        assigned, in decreasing order of their maximum weighted
        similarity, to the cluster achieving that maximum; each
        assignment triggers an incremental update of the weighted
        similarity of its still-unlabeled reverse close neighbors (via
        :attr:`INearstNeighborDic`).

        Returns
        -------
        None
            Updates ``self.Labels`` in place for all samples reachable,
            directly or transitively, from an already labeled neighbor
            through the close-neighbor relationship.
        """
        # Build the reverse nearest-neighbor dictionary of the samples
        self.__getInverseNearstNeighbor()
        # Identification matrix: key is the index of a sample awaiting a
        # label, value is [maximum weighted similarity obtained, the
        # corresponding cluster id]
        identifyMatrix = {}
        # Neighbor classification matrix: neighbor count per cluster
        classifyMatrix = {}
        # Neighbor total-distance matrix: cumulative distance per cluster
        distanceMatrix = {}
        # Build the identification matrix
        for index in range(self.Size):
            # Only construct identification information for samples that
            # have not yet been assigned a label
            if self.Labels[index] == 0:
                # Initialize the neighbor classification counts
                classifyMatrix[index] = np.zeros(self.NC + 1, dtype=int)
                # Initialize the neighbor total distances
                distanceMatrix[index] = np.zeros(self.NC + 1, dtype=float)
                # Initialize the neighbor mean distances
                meanDistance = np.zeros(self.NC + 1, dtype=float)
                # The mean distance to cluster 0 (unlabeled) is set to infinity
                meanDistance[0] = float('inf')
                # Accumulate neighbor statistics grouped by cluster label
                for neighbor in self.NNMatrix[index][:self.CutoffVector[index]]:
                    classifyMatrix[index][self.Labels[neighbor]] += 1
                    distanceMatrix[index][self.Labels[neighbor]] += self.DM[index][neighbor]
                # Compute the mean distance to each cluster (clusters with
                # no neighbors are assigned an infinite mean distance)
                for cluster in range(1, self.NC + 1):
                    if classifyMatrix[index][cluster] == 0:
                        meanDistance[cluster] = float('inf')
                    else:
                        meanDistance[cluster] = distanceMatrix[index][cluster] / classifyMatrix[index][cluster]
                # Convert the mean distance into a similarity score
                similarityDetail = np.exp(-meanDistance)
                # Weight the similarity score by the neighbor-count proportion
                similarityDetail = np.multiply(similarityDetail, classifyMatrix[index] / self.CutoffVector[index])
                # Store the identification information of this sample
                identifyMatrix[index] = [np.max(similarityDetail), np.argmax(similarityDetail)]
        # Iteratively assign labels based on the identification matrix
        while len(identifyMatrix) > 0:
            # Select the sample with the highest weighted similarity for label assignment
            maxSimilarityIndex = max(identifyMatrix, key=identifyMatrix.get)
            # If its best-matching cluster id is 0 (unlabeled), no further
            # progress can be made by this strategy; stop
            clusterID = identifyMatrix[maxSimilarityIndex][1]
            if clusterID == 0:
                break
            # Otherwise, assign it to the cluster with the highest weighted similarity
            self.Labels[maxSimilarityIndex] = clusterID
            # Incrementally update the identification information of every
            # unlabeled sample that regards this newly labeled sample as a close neighbor
            for host in self.INearstNeighborDic[maxSimilarityIndex]:
                # Skip samples that have already been assigned a label
                if self.Labels[host] != 0:
                    continue
                # Update the neighbor classification counts
                classifyMatrix[host][clusterID] += 1
                classifyMatrix[host][0] -= 1
                # Update the neighbor total-distance matrix
                distanceMatrix[host][clusterID] += self.DM[maxSimilarityIndex][host]
                # Recompute the weighted similarity of the host sample on the newly updated cluster
                similarity = np.exp(-distanceMatrix[host][clusterID] / classifyMatrix[host][clusterID])
                similarity *= classifyMatrix[host][clusterID] / self.CutoffVector[host]
                # Update the identification information if this improves the best match
                if similarity > identifyMatrix[host][0]:
                    identifyMatrix[host][0] = similarity
                    identifyMatrix[host][1] = clusterID
            # Remove the newly labeled sample from the identification matrix
            identifyMatrix.pop(maxSimilarityIndex)

    def __LabelAllocation3(self):
        """Label allocation strategy 3: DPC-style parent-chain fallback assignment.

        For every sample that remains unlabeled after
        :meth:`__LabelAllocation2`, follows its chain of "object" (parent)
        samples, defined in :meth:`__getRelativeDistance`, until reaching
        a sample that has already been assigned a label; every sample
        along that chain is then assigned this discovered label. This
        mirrors the classical DPC label-assignment rule and guarantees
        that every sample is eventually labeled.

        Returns
        -------
        None
            Finalizes ``self.Labels`` in place for all remaining samples.
        """
        for i in range(self.Size):
            if self.Labels[i] == 0:
                # Locate the parent-object chain that the current sample belongs to
                tmpIndex = i
                Label = 0
                # Record the chain of not-yet-labeled samples awaiting assignment
                objectChain = []
                objectChain.append(tmpIndex)
                while Label == 0:
                    # Advance to the next object (parent) sample along the chain
                    tmpIndex = self.ObjectVector[tmpIndex]
                    # Retrieve the label of the new object sample
                    Label = self.Labels[tmpIndex]
                    # If the new object sample is also unlabeled, append it to the chain
                    if Label == 0:
                        objectChain.append(tmpIndex)
                # Assign the discovered label to every sample along the chain
                for index in objectChain:
                    self.Labels[index] = Label

    def fit(self):
        """Run the complete WANN-DPC clustering pipeline.

        Executing this method sequentially performs feature
        normalization, distance-matrix construction, nearest-neighbor
        ordering, weighted local-density estimation, relative-distance
        computation, correction-factor-weighted cluster-center
        selection, and the three-stage label allocation strategy.

        Returns
        -------
        None
            After calling this method, the fitted results are available
            via ``self.Labels`` (predicted cluster label per sample) and
            ``self.ClusterCenters`` (indices of the selected cluster
            centers).
        """
        # Normalize the sample data
        self.__Normalization()
        # Compute the distance matrix of the samples
        self.__getDistanceMatrix()
        # Compute the nearest-neighbor ordering information of the samples
        self.__getNeighborInfomation()
        # Compute the weighted local density of every sample
        self.__getWeightDensity()
        # Compute the relative distance between samples
        self.__getRelativeDistance()
        # Select the cluster centers
        self.__getClusterCenter()
        # Perform the three-stage label allocation
        self.__LabelAllocation1()
        self.__LabelAllocation2()
        self.__LabelAllocation3()
