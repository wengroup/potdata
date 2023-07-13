"""Samplers to select a subset of objects (e.g. structures) from a sequence."""
import abc
import warnings
from typing import Any, Sequence

import numpy as np
from monty.dev import requires
from monty.json import MSONable
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from potdata.utils.dataops import serializable_slice, slice_sequence

try:
    from dscribe.descriptors import SOAP
except ImportError:
    SOAP = None

__all__ = ["BaseSampler", "RandomSampler", "SliceSampler", "DBSCANStructureSampler"]


class BaseSampler(MSONable):
    """Base class for data sampling.

    This is for general purpose sampling without any knowledge of the data. For sampling
    methods need to know structures, see `BaseStructureSampler`.
    """

    @abc.abstractmethod
    def sample(self, data: Sequence[Any]) -> list[Any]:
        """Run the sampler to sample a subset of the data.

        Args:
            data: data to sample from.

        Returns:
            A list of sampled objects.
        """

    @property
    @abc.abstractmethod
    def indices(self) -> list[int]:
        """
        Return the indices of the data that has been sampled.
        """


class BaseStructureSampler(MSONable):
    """Base sampler from a sequence of structures."""

    @abc.abstractmethod
    def sample(self, data: list[Structure]) -> list[Structure]:
        """Run the sampler to sample a subset of the data.

        Args:
            data: data to sample from.

        Returns:
            A list of sampled structures.
        """

    @property
    @abc.abstractmethod
    def indices(self) -> list[int]:
        """
        Return the indices of the structures that has been sampled.
        """


class RandomSampler(BaseSampler):
    """Randomly sample a subset.

    Args:
        size: number of data points to sample.
        seed: random seed for the sampler.
    """

    def __init__(self, size: int, seed: int = 35):
        self.size = size
        self.seed = seed
        self._indices: list[int] = None
        np.random.seed(self.seed)

    def sample(self, data: Sequence[Any]) -> list[Any]:
        data = [x for x in data]

        if self.size > len(data):
            raise ValueError(
                f"Requested number of samples `{self.size}` larger than total "
                f"number f data points `{len(data)}`."
            )

        self._indices = [
            i for i in np.random.randint(low=0, high=len(data), size=self.size)
        ]
        selected = [data[i] for i in self._indices]

        return selected

    @property
    def indices(self) -> list[int]:
        return self._indices


class SliceSampler(BaseSampler):
    """Sample a slice of the data points.

    Args:
        index: indices of the data to select. If a list of int is provided, the
            corresponding indices are selected, e.g. `index=[0, 3, 5]` will select data
            points with indices 0, 3, and 5. Alternatively, a python slice object can be
            provided, e.g. `index=slice(10, None, 2)` will select data points with
            indices 10, 12, 14, ....
    """

    # TODO, can we not use serializable_slice? and directly use slice?
    #   This depends on how we use it in potflow.
    #   This is difficult, because self.index need to be MSONable, but slice is not.
    def __init__(self, index: list[int] | serializable_slice):
        self.index = index
        self._indices: list[int] = None

    def sample(self, data: Sequence[Any]) -> list[Any]:
        selected, self._indices = slice_sequence(data, self.index)

        return selected

    @property
    def indices(self) -> list[int]:
        return self._indices


class DBSCANStructureSampler(BaseStructureSampler):
    """Sample structures using DBSCAN clustering.

    This is achieved in the below steps:
    1. convert each structure to a list of SOAP vectors, one for each atom.
    2. (Optional) select a subset of the SOAP vectors (e.g. only the vectors for the
       atoms of specific types of species).
    3. Pool the (selected) SOAP vectors for atoms in each structure into a single vector
       to get a vector representation of each structure. The pooling is done by taking
       the mean or by concatenating the vectors.
    4. (Optional) reduce the dimension of the concatenated vectors using PCA.
    5. use `sklearn.cluster.DBSCAN` to cluster the SOAP vectors. Each vector will be
       clustered into one of the following categories: `core`, `reachable`, and `noisy`.
    6. For each of the categories, randomly sample a subset of the structures.

    Args:
        soap_kwargs: arguments to pass to `dscribe.descriptors.SOAP`. Note, default
            values are provided for some of the arguments, which you can override; see
            `DEFAULT_SOAP_KWARGS`. Also, the `species` argument is not needed, as it
            will be automatically inferred from the structures.
        species_to_select: The species of the atoms for which the SOAP vectors will
            be selected for clustering. If None, all atoms will be used.
        pool_method: method to pool the SOAP vectors for atoms in each structure into a
            single vector. Can be `mean` or `concatenate`.
        pca_dim: dimension of the PCA to perform on the concatenated SOAP vectors. If
            `None`, no PCA will be performed.
        dbscan_kwargs: arguments to pass to `sklearn.cluster.DBSCAN`.
        reachable_ratio: ratio of reachable data points to sample.
            set to `n_avg_reachable/n_avg_core`, where `n_avg_reachable` is the average
            number of neighbors of all reachable data points, and `n_avg_core` is the
            average number of neighbors of all core data points.
        noisy_ratio: ratio of noisy data points to sample.
        core_ratio: ratio of core data points to sample. If `auto`, the ratio will be
        ratio: global ratio factor to be multiplied to the ratio of each category. This
            is useful to control the total number of data points to sample.
        seed: random seed for the sampling.
    """

    DEFAULT_SOAP_KWARGS = {"r_cut": 5.0, "n_max": 8, "l_max": 5, "periodic": True}

    def __init__(
        self,
        soap_kwargs: dict = None,
        species_to_select: list[str] | None = None,
        pool_method: str = "concatenate",
        pca_dim: int | None = None,
        dbscan_kwargs: dict = None,
        core_ratio: str | float = "auto",
        noisy_ratio: float = 1.0,
        reachable_ratio: float = 1.0,
        ratio: float = 1.0,
        seed: int = 35,
    ):
        self.soap_kwargs = (
            self.DEFAULT_SOAP_KWARGS.copy().update(soap_kwargs)
            if soap_kwargs is not None
            else self.DEFAULT_SOAP_KWARGS.copy()
        )

        self.species_to_select = species_to_select
        self.pool_method = pool_method
        self.pca_dim = pca_dim
        self.dbscan_kwargs = dbscan_kwargs if dbscan_kwargs is not None else {}
        self.core_ratio = core_ratio
        self.noisy_ratio = noisy_ratio
        self.reachable_ratio = reachable_ratio
        self.ratio = ratio

        # indices of all sampled points and sampled core, reachable, and noisy points
        self._indices: list[int] = None
        self._core_indices: list[int] = None
        self._reachable_indices: list[int] = None
        self._noisy_indices: list[int] = None

        # soap vector of each structure
        self._soap_vectors = None

        np.random.seed(seed)

    def sample(self, data: list[Structure]) -> list[Structure]:
        """Sample the structures."""

        # soap vectors of all atoms for all structures
        soap_vec_atoms = self._get_soap_vector_atom(data, self.soap_kwargs)

        # select soap vectors for atoms of specific species
        if self.species_to_select is not None:
            soap_vec_atoms = self._select_by_species(
                data, soap_vec_atoms, self.species_to_select
            )

        # soap vector of each structure, 2D array (n_structures, n_features)
        self._soap_vectors = self._get_soap_vector_structure(
            soap_vec_atoms, self.pool_method
        )

        # dim reduction with PCA
        if self.pca_dim is not None:
            self._soap_vectors = self._dim_reduction(self._soap_vectors, self.pca_dim)

        # classify soap vectors/structures into core, reachable, and noisy
        core_idx, reachable_idx, noisy_idx = self._cluster(self._soap_vectors)

        if isinstance(self.core_ratio, str):
            if self.core_ratio.lower() == "auto":
                core_ratio = self._estimate_core_ratio(
                    self._soap_vectors,
                    core_idx,
                    reachable_idx,
                    radius=self.dbscan_kwargs["eps"],
                )
            else:
                raise ValueError(
                    f"Unsupported core_ratio `{self.core_ratio}`. Expected either "
                    "`auto` or a float."
                )
        else:
            core_ratio = self.core_ratio

        # sample soap vectors/structures
        print("Total number of data points:", len(data))

        self._core_indices, core = self._select(data, core_idx, core_ratio * self.ratio)
        print(f"Sampled {len(core)} out of {len(core_idx)} core points.")

        self._reachable_indices, reachable = self._select(
            data, reachable_idx, self.reachable_ratio * self.ratio
        )
        print(f"Sampled {len(reachable)} out of {len(reachable_idx)} reachable points.")

        self._noisy_indices, noisy = self._select(
            data, noisy_idx, self.noisy_ratio * self.ratio
        )
        print(f"Sampled {len(noisy)} out of {len(noisy_idx)} noisy points.")

        sampled_structures = core + reachable + noisy
        self._indices = (
            self._core_indices + self._reachable_indices + self._noisy_indices
        )

        return sampled_structures

    @requires(
        SOAP,
        "`dscribe` is required to use the sampler. To install it, see "
        "https://github.com/SINGROUP/dscribe",
    )
    # This can be a staticmethod; not use because @requires does not work with it
    def _get_soap_vector_atom(
        self, data: list[Structure], soap_kwargs: dict
    ) -> list[np.ndarray]:
        """Convert structures to SOAP vectors of all atoms.

        Returns:
            SOAP vectors for all structures. For each structure, the SOAP vectors is a
            2D array of shape (n_atoms, n_features), where n_atoms is the number of
            atoms and n_features is the number of features in the SOAP vector.
        """

        species = set()
        for structure in data:
            species.update(structure.symbol_set)

        if "species" in soap_kwargs:
            raise ValueError(
                "The `species` argument in `soap_kwargs` is not allowed. "
                "The species will be automatically determined from the structures."
            )

        soap = SOAP(species=species, **soap_kwargs)
        atoms = [AseAtomsAdaptor.get_atoms(structure) for structure in data]
        soap_vectors = list(soap.create(atoms))

        return soap_vectors

    @staticmethod
    def _select_by_species(
        structures: list[Structure], vectors: list[np.ndarray], species: list[str]
    ) -> list[np.ndarray]:
        """Select SOAP vectors for atoms of specific species."""

        def select_one(struct, vec):
            indices = [i for i, s in enumerate(struct.species) if s.symbol in species]
            return vec[indices]

        selected = []
        for struct, vec in zip(structures, vectors):
            selected.append(select_one(struct, vec))

        return selected

    @staticmethod
    def _get_soap_vector_structure(vectors: list[np.ndarray], pool: str) -> np.ndarray:
        """Convert SOAP vectors of atoms to SOAP vectors of structures.

        Return a 2D array of shape (n_structures, n_features).
        """
        if pool.lower() == "mean":
            return np.asarray([np.mean(x, axis=0) for x in vectors])
        elif pool.lower() == "concatenate":
            return np.asarray([np.ravel(x) for x in vectors])
        else:
            supported = ["mean", "concatenate"]
            raise ValueError(
                f"Unsupported pooling method `{pool}`. Expected one of " f"{supported}."
            )

    @staticmethod
    def _dim_reduction(x: np.ndarray, dim: int | float):
        """Perform dimension reduction on a 2D array."""
        pca = PCA(n_components=dim)
        reduced = pca.fit_transform(x)

        return reduced

    def _cluster(self, data: np.ndarray) -> tuple[list[int], list[int], list[int]]:
        """
        Perform DBSCAN and classify the data points into core, reachable and noisy ones.
        """

        clustering = DBSCAN(**self.dbscan_kwargs).fit(data)

        # get core, reachable and noisy points
        labels = clustering.labels_
        core_indices = sorted(clustering.core_sample_indices_)
        noisy_indices = sorted(np.nonzero(labels == -1)[0])
        reachable_indices = [
            i
            for i in range(len(labels))
            if i not in core_indices and i not in noisy_indices
        ]

        if not noisy_indices:
            warnings.warn(
                "No noisy points are found. Consider decreasing `eps` or increasing "
                "`min_samples` of DBSCAN."
            )
        if not reachable_indices:
            warnings.warn(
                "No reachable points are found. Consider decreasing `eps` or "
                "increasing `min_samples` of DBSCAN."
            )
        if not core_indices:
            warnings.warn(
                "No core points are found. Consider increasing `eps` or decreasing "
                "`min_samples` of DBSCAN."
            )

        return core_indices, noisy_indices, reachable_indices

    @staticmethod
    def _estimate_core_ratio(
        soap_vectors: np.ndarray,
        core_indices: list[int],
        reachable_indices: list[int],
        radius: float,
    ) -> float:
        """Estimate the ratio of core points to all data points.

        Args:
            soap_vectors: SOAP vectors of all data points.
            core_indices: Indices of core points.
            reachable_indices: Indices of reachable points.
            radius: Radius to determine the neighbors.
        """
        neigh = NearestNeighbors(radius=radius)
        neigh.fit(soap_vectors)
        neighbors = neigh.radius_neighbors(soap_vectors, return_distance=False)

        if len(reachable_indices) == 0:
            n_avg_reachable = 1
        else:
            n_avg_reachable = np.mean([len(n) for n in neighbors[reachable_indices]])

        if len(core_indices) == 0:
            n_avg_core = 1
        else:
            n_avg_core = np.mean([len(n) for n in neighbors[core_indices]])

        ratio = n_avg_reachable / n_avg_core

        print(
            f"Estimated core ratio = num_avg_reachable/num_avg_core = "
            f"{n_avg_reachable:.2f}/{n_avg_core:.2f} = {ratio:.2f}."
        )

        return ratio

    @staticmethod
    def _select(
        data: list[Structure], indices: list[int], ratio: float
    ) -> tuple[list[int], list[Structure]]:
        """Select a subset of structures and indices."""
        sample_size = int(ratio * len(indices))
        selected_indices = sorted(np.random.choice(indices, sample_size, replace=False))
        selected_structures = [data[i] for i in selected_indices]

        return selected_indices, selected_structures

    def plot(self, show: bool = False, figname: str = "dbscan_sample.pdf"):
        """Function to plot the results of the selection.

        Args:
            show: Whether to show the plot.
            figname: Name of the figure file to save.

        """
        import matplotlib.pyplot as plt

        if self._soap_vectors is None:
            raise RuntimeError(
                "The `sample` method must be called before calling `plot`."
            )

        # Note, it is highly possible that PCA reduced soap vectors have more than
        # 2 dimensions, for example, if the input `pca_dim` is a float number.
        # Here we do PCA again to reduce the dimension to 2, merely for plotting.
        if self._soap_vectors.shape[1] > 2:
            soap_vectors = self._dim_reduction(self._soap_vectors, 2)
        else:
            soap_vectors = self._soap_vectors

        # soap vectors of sampled points
        core = soap_vectors[self._core_indices]
        reachable = soap_vectors[self._reachable_indices]
        noisy = soap_vectors[self._noisy_indices]

        plt.figure(figsize=(5, 5))
        plt.scatter(
            core[:, 0],
            core[:, 1],
            color="C0",
            alpha=0.8,
            edgecolors="white",
            label="core",
        )
        plt.scatter(
            reachable[:, 0],
            reachable[:, 1],
            color="C1",
            alpha=0.8,
            edgecolors="white",
            label="reachable",
        )
        plt.scatter(
            noisy[:, 0],
            noisy[:, 1],
            color="C2",
            alpha=0.8,
            edgecolors="white",
            label="noisy",
        )
        plt.xlabel("PC1")
        plt.ylabel("PC2")

        plt.legend()

        plt.savefig(figname, bbox_inches="tight")

        if show:
            plt.show()
