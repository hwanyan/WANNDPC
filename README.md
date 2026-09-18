# WANN-DPC: Density Peaks Finding Clustering Based on Weighted Adaptive Nearest Neighbors

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.patcog.2025.111953-blue)](https://doi.org/10.1016/j.patcog.2025.111953)

A Python implementation of **WANN-DPC**, a density-peak-based clustering
algorithm that improves upon ANN-DPC (Adaptive Nearest Neighbor DPC) by
introducing a **weighted local density** definition and a
**correction-factor-driven cluster-center selection strategy**, enabling
simultaneous detection of cluster centers in both dense and sparse
clusters while mitigating the "Domino Effect" common to DPC and its
variants.

## About This Repository

This repository is the **official source-code repository** accompanying
the paper *"WANN-DPC: Density peaks finding clustering based on Weighted
Adaptive Nearest Neighbors"*, published in *Pattern Recognition*. The
paper is a collaborative work of five co-authors (see
[Paper Authors](#paper-authors) below); this repository, however, is
**individually created and maintained by Huan Yan** (one of the paper's
co-authors, responsible for the algorithm implementation). Huan Yan is
the maintainer in charge of the code releases, issue tracking, and pull
requests for this repository, while the scientific contribution and
authorship of the underlying research remain jointly credited to all
paper co-authors as listed below.

## Abstract

> DPC (Density Peak Clustering) algorithm and most of its variants are
> unable to identify the cluster centers of dense and sparse clusters
> simultaneously. In addition, the "Domino Effect" of DPC cannot be
> entirely avoided in its variants. Despite ANN-DPC (Adaptive Nearest
> Neighbor DPC) being able to detect cluster centers of dense and sparse
> clusters, its adaptive nearest neighbors of a point may introduce bias
> in the local density, cluster centers and clustering. To address these
> limitations of ANN-DPC, the WANN-DPC (Weighted Adaptive Nearest
> Neighbor DPC) algorithm is proposed. The key contributions of
> WANN-DPC are as follows: (1) A novel weighted local density of a point
> is defined by weighting its close and far neighbors, (2) a correction
> factor is proposed to detect cluster centers in turn, and (3) a
> two-step assignment strategy is presented utilizing nearest neighbor
> relationships and weighted membership degrees. Extensive experiments
> on benchmark datasets demonstrate the superiority of the WANN-DPC over
> its peers.

**Keywords:** Weighted adapted nearest neighbors, Density peaks, Local
density, Cluster centers, Clustering

## Algorithm Pipeline

WANN-DPC completes clustering through the following stages (see the
module-level and per-method docstrings in [`WANNDPC.py`](./WANNDPC.py)
for full algorithmic details):

1. **Feature normalization & distance matrix construction** — min-max
   scaling followed by pairwise Euclidean distance computation.
2. **Nearest-neighbor ordering** — index and distance matrices sorted in
   ascending order of distance.
3. **Weighted local density estimation** — adaptive near/far neighbor
   splitting via a maximum-gap cutoff, combining close- and
   far-neighbor similarity.
4. **Relative distance & parent-object computation** for every sample.
5. **Cluster-center detection** — a correction-factor-weighted decision
   value combined with close-neighbor label propagation, allowing
   dense- and sparse-cluster centers to be detected in turn.
6. **Label allocation strategy 1** — close-neighbor-relationship-based
   propagation to build a preliminary clustering skeleton.
7. **Label allocation strategy 2** — weighted nearest-neighbor
   membership-degree propagation for remaining unlabeled samples.
8. **Label allocation strategy 3** — DPC-style parent-chain fallback
   assignment for any residual unlabeled samples.

## Requirements

- Python >= 3.7
- numpy
- scikit-learn

Install the dependencies:

```bash
pip install numpy scikit-learn
```

## Usage

```python
import numpy as np
from WANNDPC import WANNDPC

# X: feature matrix of shape (n_samples, n_features)
X = np.array([...])

# clusterNumber: the target (preset) number of clusters
model = WANNDPC(X, clusterNumber=3)
model.fit()

print("Predicted labels:", model.Labels)          # cluster label per sample
print("Cluster centers:", model.ClusterCenters)   # indices of cluster centers
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `X` | `numpy.ndarray` | — | Feature matrix of shape `(n_samples, n_features)`. |
| `clusterNumber` | `int` | — | Target (preset) number of clusters to detect. |
| `Correction` | `float` | `2.5` | Correction factor applied to the dependency term during cluster-center selection. |
| `NeighborLimit` | `int` | `30` | Upper bound on the number of nearest neighbors considered when estimating weighted local density. |

## Citation

If you use this code in your research, please cite the published paper:

> Xie, J., Yan, H., Wang, M., Grant, P. W., & Pedrycz, W. (2026).
> WANN-DPC: Density peaks finding clustering based on Weighted Adaptive
> Nearest Neighbors. *Pattern Recognition*, 170, 111953.
> https://doi.org/10.1016/j.patcog.2025.111953

BibTeX:

```bibtex
@article{yan2026wanndpc,
    title   = {WANN-DPC: Density peaks finding clustering based on
                Weighted Adaptive Nearest Neighbors},
    author  = {Xie, Juanying and Yan, Huan and Wang, Mingzhao and
                Grant, Philip W. and Pedrycz, Witold},
    journal = {Pattern Recognition},
    volume  = {170},
    pages   = {111953},
    year    = {2026},
    doi     = {10.1016/j.patcog.2025.111953}
}
```

## Paper Authors

The paper *"WANN-DPC: Density peaks finding clustering based on Weighted
Adaptive Nearest Neighbors"* is a joint effort of the following
co-authors (corresponding author marked with \*):

| Author | Affiliation |
|---|---|
| Juanying Xie\* | School of Computer Science, Shaanxi Normal University, Xi'an, 710119, China |
| **Huan Yan** | School of Computer Science, Shaanxi Normal University, Xi'an, 710119, China |
| Mingzhao Wang | School of Computer Science, Shaanxi Normal University, Xi'an, 710119, China |
| Philip W. Grant | Department of Computer Science, Swansea University, Swansea, SA2 8PP, UK |
| Witold Pedrycz | Department of Electrical and Computer Engineering, University of Alberta, Edmonton, Alberta, T6G 2R3, Canada |

> Note: This repository only reflects the maintenance responsibility of
> Huan Yan for the released code. Scientific credit for the research
> presented in the paper is shared jointly among all co-authors above.

## Repository Maintainer

| | |
|---|---|
| **Name (中文)** | 严欢 |
| **Name (English)** | Huan Yan (hwanyan) |
| **Email** | yan-huan@snnu.edu.cn |
| **ORCID** | [https://orcid.org/0009-0000-0016-6324](https://orcid.org/0009-0000-0016-6324) |
| **Affiliations** | 1. Tencent Cloud Computing (Chongqing) Co., Ltd. <br> 2. Shaanxi Normal University |
| **Role** | Paper co-author; sole creator and maintainer of this official code repository |

## License

This project is released under the [MIT License](./LICENSE).
