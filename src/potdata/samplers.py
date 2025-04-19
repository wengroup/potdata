"""Samplers to select a subset of objects (e.g. structures) from a sequence."""

import abc
import warnings
from typing import Any, Sequence
import random
import numpy as np
from monty.dev import requires
from monty.json import MSONable
from monty.serialization import dumpfn
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from potdata.utils.dataops import slice_sequence

try:
    from dscribe.descriptors import SOAP
except ImportError:
    SOAP = None

try:
    import pyace
except ImportError:
    pyace = None

__all__ = [
    "BaseSampler",
    "RandomSampler",
    "SliceSampler",
    "DBSCANStructureSampler",
    "KMeansStructureSampler",
    "ACEGammaSampler",
]


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
        """Return the indices of the data that has been sampled."""


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
            indices 10, 12, 14, ... `None` to select all.
    """

    def __init__(self, index: list[int] | slice | None):
        self.index = index
        self._indices: list[int] = None

    def sample(self, data: Sequence[Any]) -> list[Any]:
        selected, self._indices = slice_sequence(data, self.index)

        return selected

    @property
    def indices(self) -> list[int]:
        return self._indices

    def as_dict(self) -> dict:
        if isinstance(self.index, slice):
            # deal with `slice`, which cannot be serialized
            index = {
                "@class": "slice",
                "start": self.index.start,
                "stop": self.index.stop,
                "step": self.index.step,
            }
        else:
            index = self.index  # type: ignore

        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "index": index,
        }

        return d

    @classmethod
    def from_dict(cls, d):
        index = d["index"]
        if isinstance(index, dict) and index["@class"] == "slice":
            index = slice(index["start"], index["stop"], index["step"])
        return cls(index=index)


class BaseStructureSamplerWithSoap(BaseStructureSampler):
    """Sample structures based on their SOAP vectors.

    This is typically achieved in the below steps:
    1. get soap vectors for all atoms in each structure (one for each atom),
        called atom-vectors
    2. (Optional) select a subset of atom-vectors (e.g. only the vectors of atoms of
        specific species type)
    3. pool the (selected) atom-vectors for each structure into a single vector, called
        structure-vector
    4. (Optional) reduce the dimension of structure-vectors using PCA
    5. sample the structures based on the structure-vectors using a clustering algorithm
        (e.g. DBSCAN or KMeans)

    This class implements steps 1-4, and thus is not meant to be used directly. Instead,
    you should use one of its subclasses, such as `DBSCANStructureSampler` or
    `KMeansStructureSampler`, which implement step 5 with a specific clustering
    algorithm.

    Args:
         soap_kwargs: arguments to pass to `dscribe.descriptors.SOAP`. Note, default
            values are provided for some of the arguments, which you can override; see
            `DEFAULT_SOAP_KWARGS`. Also, the `species` argument is not needed, as it
            will be automatically inferred from the structures.
        species_to_select: The species of the atoms whose SOAP vectors will be selected
            to obtain a structure-vector. If None, all atoms will be used.
        pool_method: method to pool the (selected) atom-vectors for each structure into
            a structure-vector. Can be `mean` or `concatenate`.
        pca_dim: dimension of the PCA to perform on the concatenated SOAP vectors. If
            `None`, no PCA will be performed. If a float between 0 and 1 is provided, it
            will be treated as the explained variance ratio. If an integer is provided,
            it will be treated as the number of components to keep.
    """

    DEFAULT_SOAP_KWARGS = {"r_cut": 5.0, "n_max": 8, "l_max": 5, "periodic": True}

    def __init__(
        self,
        soap_kwargs: dict = None,
        species_to_select: list[str] | None = None,
        pool_method: str = "concatenate",
        pca_dim: int | None = None,
    ):
        self.soap_kwargs = self.DEFAULT_SOAP_KWARGS.copy()
        if soap_kwargs is not None:
            self.soap_kwargs.update(soap_kwargs)

        self.species_to_select = species_to_select
        self.pool_method = pool_method
        self.pca_dim = pca_dim

        # soap vector of each structure
        self._soap_vectors = None

        self._indices: list[int] = None

    def sample(self, data: list[Structure]) -> list[Structure]:
        """Sample the structures."""

        # 1. soap vectors of all atoms for all structures
        soap_vec_atoms = self._get_soap_vector_atom(data, self.soap_kwargs)

        # 2. select soap vectors for atoms of specific species
        if self.species_to_select is not None:
            soap_vec_atoms = self._select_by_species(
                data, soap_vec_atoms, self.species_to_select
            )

        # 3. soap vector of each structure, 2D array (n_structures, n_features)
        self._soap_vectors = self._get_soap_vector_structure(
            soap_vec_atoms, self.pool_method
        )

        # 4. dim reduction with PCA
        if self.pca_dim is not None:
            self._soap_vectors = self._dim_reduction(self._soap_vectors, self.pca_dim)

        # 5. sample based on clusters
        selected, indices = self.cluster_and_sample(data, self._soap_vectors)

        self._indices = indices

        return selected

    @property
    def indices(self) -> list[int]:
        return self._indices

    @abc.abstractmethod
    def cluster_and_sample(
        self, data: list[Structure], soap_vectors: np.ndarray
    ) -> tuple[list[Structure], list[int]]:
        """Perform clustering and then sample the structures.

        Args:
            data: list of structures.
            soap_vectors: SOAP vectors of the structures.

        Returns:
            sampled_structures: A list of sampled structures and their indices.
            sampled_indices: Indices of the sampled structures.
        """

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
        """Pool SOAP vectors of atoms to get SOAP vectors of structures.

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


class DBSCANStructureSampler(BaseStructureSamplerWithSoap):
    """Sample structures using SOAP and DBSCAN clustering.

    See `BaseStructureSamplerWithSoap` for details on some of the arguments.

    Args:
        dbscan_kwargs: arguments to pass to `sklearn.cluster_and_sample.DBSCAN`.
        reachable_ratio: ratio of reachable data points to sample.
            set to `n_avg_reachable/n_avg_core`, where `n_avg_reachable` is the average
            number of neighbors of all reachable data points, and `n_avg_core` is the
            average number of neighbors of all core data points.
        noisy_ratio: ratio of noisy data points to sample.
        core_ratio: ratio of core data points to sample. If `auto`, the ratio will be
            calculated automatically, according to the formula:
            `core_ratio = n_avg_reachable/n_avg_core`, where `n_avg_reachable` is the
            average number of neighbors of all reachable data points, and `n_avg_core`
            is the average number of neighbors of all core data points. Alternatively,
            you can provide a float value for the ratio, e.g. `core_ratio=0.5` will
            sample half of the core data points.
        ratio: global ratio factor to be multiplied to the ratio of each category. This
            is useful to control the total number of data points to sample.
        seed: random seed for the sampling.
    """

    DEFAULT_DBSCAN_KWARGS = {"eps": 0.5, "min_samples": 5}

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
        super().__init__(soap_kwargs, species_to_select, pool_method, pca_dim)

        self.dbscan_kwargs = self.DEFAULT_DBSCAN_KWARGS.copy()
        if dbscan_kwargs is not None:
            self.dbscan_kwargs.update(dbscan_kwargs)

        self.core_ratio = core_ratio
        self.noisy_ratio = noisy_ratio
        self.reachable_ratio = reachable_ratio
        self.ratio = ratio
        self.seed = seed

        # indices of all sampled points and sampled core, reachable, and noisy points
        self._core_indices: list[int] = None
        self._reachable_indices: list[int] = None
        self._noisy_indices: list[int] = None

        np.random.seed(self.seed)

    def cluster_and_sample(
        self, data: list[Structure], soap_vectors: np.ndarray
    ) -> tuple[list[Structure], list[int]]:
        # classify soap vectors/structures into core, reachable, and noisy
        core_idx, reachable_idx, noisy_idx = self._cluster(soap_vectors)

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
        print("DBSCAN sampler:")
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
        indices = self._core_indices + self._reachable_indices + self._noisy_indices

        return sampled_structures, indices

    def _cluster(self, data: np.ndarray) -> tuple[list[int], list[int], list[int]]:
        """
        Perform DBSCAN and classify the data points into core, reachable and noisy ones.
        """

        clustering = DBSCAN(**self.dbscan_kwargs)
        clustering.fit(data)

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

    def plot2(
        self,
        separation1: int,
        separation2: int,
        show: bool = False,
        plot_only_sampled: bool = True,
        figname: str = "Transition_sample.pdf",
    ):
        """Function to plot the results of the transition, which composes of groups 1 (before transition),
           group 2 (during transition) and group 3 (after transition). Free to add more groups if needed.

           How to calculate the separation index:

           Separation index = ((Specific transition step - SliceSampler.start) / SliceSampler.step) + 1

           Specific transition step (int): The specific step at which the transition of interest occurs.
           SliceSampler.start (int): The starting step for the SliceSampling process.
           SliceSampler.step (int): The step size used for SliceSampling.

           Note: Don't forget to add one because the default numbering for steps starts from 1.

        Args:
            separation1: The index marking the separation between groups 1 and 2.
            separation2: The index marking the separation between groups 2 and 3.
            show: Whether to show the plot.
            plot_only_sampled: if True, only the sampld points will be plotted; otherwise, all points will be used.
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

        if plot_only_sampled:
            group1 = np.asarray(
                [soap_vectors[i] for i in self._indices if i < separation1]
            )
            group2 = np.asarray(
                [
                    soap_vectors[i]
                    for i in self._indices
                    if separation1 < i < separation2
                ]
            )
            group3 = np.asarray(
                [soap_vectors[i] for i in self._indices if i > separation2]
            )
        else:
            # soap vectors of sampled points
            group1 = soap_vectors[:separation1]  # Before transition
            group2 = soap_vectors[separation1:separation2]  # During transition
            group3 = soap_vectors[separation2:]  # After transition

        plt.figure(figsize=(5, 5))
        plt.scatter(
            group1[:, 0],
            group1[:, 1],
            color="C0",
            alpha=0.8,
            edgecolors="white",
            label="group1 (before transition)",
        )
        plt.scatter(
            group2[:, 0],
            group2[:, 1],
            color="C1",
            alpha=0.8,
            edgecolors="white",
            label="group2 (during transition)",
        )
        plt.scatter(
            group3[:, 0],
            group3[:, 1],
            color="C2",
            alpha=0.8,
            edgecolors="white",
            label="group3 (after transition)",
        )
        plt.xlabel("PC1")
        plt.ylabel("PC2")

        plt.legend()

        plt.savefig(figname, bbox_inches="tight")

        if show:
            plt.show()


class KMeansStructureSampler(BaseStructureSamplerWithSoap):
    """Sample structures using SOAP vectors and KMeans clustering.

    This performs KMeans clustering based on the SOAP vectors of the structures, and
    then sample a subset of the structures from each cluster.

    See `BaseStructureSamplerWithSoap` for details on some of the arguments.

    Args:
        kmeans_kwargs: Arguments to pass to `sklearn.cluster_and_sample.KMeans`.
            You should provide the `n_clusters` argument to specify the number of
            clusters.
        ratio: Sampling ratio. This determines the number of structures to sample from
            each cluster. If an integer is provided, a fixed number of samples
            is chosen from each cluster. If a float between 0 and 1 is provided, a
            fraction of samples is chosen from each cluster.
    """

    DEFAULT_KMEANS_KWARGS = {"n_clusters": 8}

    def __init__(
        self,
        soap_kwargs: dict = None,
        species_to_select: list[str] | None = None,
        pool_method: str = "concatenate",
        pca_dim: int | None = None,
        kmeans_kwargs: dict = None,
        ratio: int | float = 1,
        seed: int = 35,
    ):
        super().__init__(soap_kwargs, species_to_select, pool_method, pca_dim)

        self.kmeans_kwargs = self.DEFAULT_KMEANS_KWARGS.copy()
        if kmeans_kwargs is not None:
            self.kmeans_kwargs.update(kmeans_kwargs)

        if isinstance(ratio, float) and not 0.0 < ratio < 1.0:
            raise ValueError(
                f"Expect ratio to be an integer or a float between 0.0 and 1.0. "
                f"Got {ratio}."
            )
        self.ratio = ratio
        self.seed = seed

        # Indices of all sampled points and sampled clusters
        self._indices: list[int] = None
        self._cluster_indices: list[int] = None

        np.random.seed(self.seed)

    def cluster_and_sample(
        self, data: list[Structure], soap_vectors: np.ndarray
    ) -> tuple[list[Structure], list[int]]:
        # Perform KMeans clustering and get the cluster labels
        clustering = KMeans(**self.kmeans_kwargs)
        clustering.fit(soap_vectors)
        labels = clustering.labels_

        unique_clusters = set(labels)
        if len(unique_clusters) < self.kmeans_kwargs["n_clusters"]:
            warnings.warn(
                f"Number of clusters found ({len(unique_clusters)}) is less than "
                f"specified n_clusters ({self.kmeans_kwargs['n_clusters']}). "
                "Consider adjusting the parameters of KMeans."
            )

        # Sample structures from each cluster
        selected_indices = []
        for cluster in unique_clusters:
            cluster_indices = [i for i, label in enumerate(labels) if label == cluster]
            cluster_size = (
                int(self.ratio * len(cluster_indices))
                if isinstance(self.ratio, float)
                else int(self.ratio)
            )

            if cluster_size > len(cluster_indices):
                warnings.warn(
                    f"Requested number of samples `{cluster_size}` larger than total "
                    f"number of data points `{len(cluster_indices)}` in cluster "
                    f"{cluster}. Selecting all data points in the cluster."
                )
                cluster_size = len(cluster_indices)

            selected_indices.extend(
                np.random.choice(cluster_indices, cluster_size, replace=False)
            )

        selected_structures = [data[i] for i in selected_indices]

        return selected_structures, selected_indices

    def plot(self, show: bool = False, figname: str = "kmeans_sample.pdf"):
        """Plot the results of the KMeans clustering.

        Args:
            show: Whether to show the plot.
            figname: Name of the figure file to save.
        """
        import matplotlib.pyplot as plt

        if self._soap_vectors is None:
            raise RuntimeError(
                "The `sample` method must be called before calling `plot`."
            )

        # Note: It is highly possible that PCA reduced soap vectors have more than
        # 2 dimensions, for example, if the input `pca_dim` is a float number.
        # Here we do PCA again to reduce the dimension to 2, merely for plotting.
        if self._soap_vectors.shape[1] > 2:
            soap_vectors = self._dim_reduction(self._soap_vectors, 2)
        else:
            soap_vectors = self._soap_vectors

        # soap vectors of sampled points
        kmeans_labels = KMeans(n_clusters=self.kmeans_kwargs["n_clusters"]).fit_predict(
            soap_vectors
        )

        plt.figure(figsize=(5, 5))
        for cluster_label in range(self.kmeans_kwargs["n_clusters"]):
            cluster_points = soap_vectors[kmeans_labels == cluster_label]
            plt.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                alpha=0.8,
                edgecolors="white",
                label=f"Cluster {cluster_label}",
            )

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()

        plt.savefig(figname, bbox_inches="tight")

        if show:
            plt.show()


class ACEGammaSampler(BaseStructureSampler):
    """
    Sample structures based on the extrapolation grade (gamma) from ACE potential.

    Args:
        gamma_range: range of gamma values to select structures from. Each structure
            has a structure-gamma, and only the structures with gamma values within this
            range are selected. If `None`, all structures are returned.
        gamma_reduce: method to reduce the gamma values. In fact, gamma values will be
            calculated for each atom in the structure. To select the structure, we
            obtain a structure-gamma value by reducing the atom-gamma values. This
            parameter specifies how to reduce the atom-gamma values to obtain the
            structure-gamma value. Options are "max" and "mean"; the former selects the
            maximum value of atom-gamma as the structure-gamma, and the latter computes
            the mean of the atom-gamma values as the structure-gamma. Default is "max".
            This is ignored if `gamma_range` is `None`.
        verbose: verbosity level. 0 (default) only save md trajectory and log file.
            1: also save gamma values.
    """

    def __init__(
        self,
        potential_filename: str = "output_potential.yaml",
        potential_asi_filename: str = "output_potential.asi",
        gamma_range: tuple[float, float] = None,
        gamma_reduce: str = "max",
        verbose: int = 0,
        sample_size: int = None,
    ):
        self.potential_filename = potential_filename
        self.potential_asi_filename = potential_asi_filename
        self.gamma_range = gamma_range
        self.gamma_reduce = gamma_reduce
        self.verbose = verbose
        self.sample_size = sample_size

        self._indices: list[int] = None

    @property
    def indices(self) -> list[int]:
        return self._indices

    @requires(
        pyace,
        "`ACE` is needed for this transformation. To install it, see "
        "https://pacemaker.readthedocs.io/en/latest/",
    )
    def sample(self, data: list[Structure]) -> list[Structure]:
        from pyace import PyACECalculator
        from pymatgen.io.ase import AseAtomsAdaptor

        calc = PyACECalculator(self.potential_filename)
        calc.set_active_set(self.potential_asi_filename)

        selected = []
        selected_indices = []
        gamma_data = []
        for i, s in enumerate(data):
            atoms = AseAtomsAdaptor.get_atoms(s)
            atoms.set_calculator(calc)
            atoms.get_potential_energy()
            gamma_atoms = calc.results["gamma"]

            if self.gamma_reduce == "max":
                gamma_structure = max(gamma_atoms)
            elif self.gamma_reduce == "mean":
                gamma_structure = np.mean(gamma_atoms)
            else:
                support = ["max", "mean"]
                raise ValueError(
                    f"Unknown gamma_reduce: {self.gamma_reduce}. "
                    f"Supported are: {support}",
                )

            if self.gamma_range[0] <= gamma_structure <= self.gamma_range[1]:
                selected.append(AseAtomsAdaptor.get_structure(atoms))
                selected_indices.append(i)

            if self.verbose > 0:
                gamma_data.append(
                    {
                        "index": i,
                        "gamma_atoms": gamma_atoms.tolist(),
                        "gamma_structure": gamma_structure.tolist(),
                    }
                )

        # check: do we have any structures within the gamma range?
        if len(selected) == 0:
            raise RuntimeError(
                f"No structures found within the gamma range: {self.gamma_range}."
            )

        # random sample
        if self.sample_size is not None and self.sample_size < len(selected):
            chosen = random.sample(list(zip(selected, selected_indices)), self.sample_size)
            selected, selected_indices = zip(*chosen)
            selected = list(selected)
            selected_indices = list(selected_indices)
        
        # save gamma data if requested
        if self.verbose > 0:
            dumpfn(gamma_data, "gamma_data.yaml")

        self._indices = selected_indices

        return selected
