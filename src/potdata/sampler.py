"""Samplers to select a subset of objects (e.g. structures) from a sequence."""
import abc
from typing import Any, Callable, Iterable

import numpy as np
from monty.json import MSONable
from pymatgen.core.structure import Structure
import matplotlib.pyplot as plt
from potdata.utils.dataops import serializable_slice, slice_sequence
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from dscribe.descriptors import SOAP
from pymatgen.io.ase import AseAtomsAdaptor

__all__ = ["BaseSampler", "RandomSampler", "SliceSampler", "DBSCANStructureSampler"]


class BaseSampler(MSONable):
    """Base class for data sampling.

    This is for general purpose sampling without any knowledge of the data. For sampling
    methods need to know structures, see `BaseStructureSampler`.
    """

    @abc.abstractmethod
    def sample(self, data: Iterable) -> list[Any]:
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
        self._indices: list[int] = []
        np.random.seed(self.seed)

    def sample(self, data: Iterable) -> list[Any]:
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
        self._indices: list[int] = []

    def sample(self, data: Iterable) -> list[Any]:
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
       atoms of a specific element).
    3. Concatenate the (selected) SOAP vectors for each structure into a single vector.
    4. (Optional) reduce the dimension of the concatenated vectors using PCA.
    5. use `sklearn.cluster.DBSCAN` to cluster the SOAP vectors. Each vector will be
       clustered into one of the following categories: `core`, `reachable`, and `noisy`.
    6. For each of the categories, randomly sample a subset of the structures.

    Args:
        soap_kwargs: arguments to pass to `dscribe.descriptors.SOAP`.
        post_soap_selection: a function to select a subset of the SOAP vectors. The
            function should take two arguments: a list of structures and a list of SOAP
            vectors, and return a list of selected SOAP vectors. If `None`, all SOAP
            vectors will be used.
        pca_dim: dimension of the PCA to perform on the concatenated SOAP vectors. If
            `None`, no PCA will be performed.
        dbscan_kwargs: arguments to pass to `sklearn.cluster.DBSCAN`.
        noisy_ratio: ratio of noisy data points to sample.
        reachable_ratio: ratio of reachable data points to sample.
        core_ratio: ratio of core data points to sample. If `auto`, the ratio will be
            set to `min_distance/average_num_neighbors`, where `min_distance` is the
            argument `min_samples` in `sklearn.cluster.DBSCAN`, and
            `average_num_neighbors` is the average number of neighbors of all core data
            points.
        seed: random seed for the sampling.
    """

    def __init__(
        self,
        elements: list[str],
        target_elements: list[str],
        soap_kwargs: dict = None,
        post_soap_selection: Callable[
            [list[Structure], list[np.ndarray]], list[np.ndarray]
        ] = None,
        pca_dim: int | None = None,
        dbscan_kwargs: dict = None,
        noisy_ratio: float = 1.0,
        reachable_ratio: float = 1.0,
        core_ratio: str | float = "auto",
        seed: int = 35,
    ):
        self.soap_kwargs = soap_kwargs if soap_kwargs is None else {}
        self.post_soap_selection = post_soap_selection
        self.pca_dim = pca_dim
        self.dbscan_kwargs = dbscan_kwargs if dbscan_kwargs is None else {}
        self.noisy_ratio = noisy_ratio
        self.reachable_ratio = reachable_ratio
        self.core_ratio = core_ratio
        self.core_sample_indices_ = []
        self._indices: list[int] = []
        self.elements = elements
        self.target_elements = target_elements
        self.sampled_noisy_points = []
        self.sampled_reachable_points = []
        self.core_samples_mask = None
        self.labels = []

        np.random.seed(seed)

        # Retrieve eps and min_samples from dbscan_kwargs
        self.eps = self.dbscan_kwargs.get('eps')
        self.min_samples = self.dbscan_kwargs.get('min_samples')

    def sample(self, data: list[Structure]) -> list[Structure]:
        """"""
        soap_vectors_all = self._get_soap_vectors(data)

        if self.post_soap_selection is not None:
            soap_vectors_all = self.post_soap_selection(data, soap_vectors_all)

        # TODO, need to double check whether this correct
        reduced_vectors = np.concatenate(soap_vectors_all, axis=1)

        if self.pca_dim is not None:
            reduced_vectors = [reduced_vectors]  # Wrap reduced_vectors in a list
            reduced_vectors = self._dim_reduction(reduced_vectors, self.pca_dim)

        self.reduced_vectors = reduced_vectors

        labels = self._cluster(reduced_vectors)
        core_samples_mask = self.core_samples_mask  # Define core_samples_mask attribute

        if self.core_ratio == "auto":
            core_ratio = self._compute_core_ratio()
        else:
            core_ratio = self.core_ratio

        self.core_samples_mask = np.zeros_like(labels, dtype=bool)
        self.core_samples_mask[self.core_sample_indices_] = True

        # Sample noisy points
        noisy_points = reduced_vectors[labels == -1]
        sample_size_noisy = int(self.noisy_ratio * len(noisy_points))
        sampled_noisy_points = np.random.choice(noisy_points, sample_size_noisy, replace=False)

        # Sample reachable points
        reachable_points = reduced_vectors[(labels != -1) & (~self.core_samples_mask)]
        sample_size_reachable = int(self.reachable_ratio * len(reachable_points))
        sampled_reachable_points = np.random.choice(reachable_points, sample_size_reachable, replace=False)

        # Sample core points
        core_points = reduced_vectors[core_samples_mask]
        sample_size_core = int(core_ratio * len(core_points))
        sampled_core_points = np.random.choice(core_points, sample_size_core, replace=False)

        # Store sampled noisy points as class attribute
        self.sampled_noisy_points = sampled_noisy_points
        self.sampled_reachable_points = sampled_reachable_points

        # `selected` below should be a list of pymatgen Structure
        selected = []

        # Add sampled noisy points to selected
        for point in sampled_noisy_points:
            selected.append(data[point])

        # Add sampled reachable points to selected
        for point in sampled_reachable_points:
            selected.append(data[point])

        # Add sampled core points to selected
        for point in sampled_core_points:
            selected.append(data[point])

        return selected

    def _get_soap_vectors(self, data: list[Structure]) -> list[np.ndarray]:
        """Convert structures to SOAP vectors."""

        # Get unique elements from the structures
        elements = list(set([element.symbol for structure in data for element in structure.composition.elements]))
        self.elements = elements

        # Set up the SOAP descriptor with the dynamic elements
        soap_desc = SOAP(**self.soap_kwargs)

        soap_vectors_all = []

        for structure in data:
            # Convert pymatgen Structure to ASE Atoms
            atoms = AseAtomsAdaptor.get_atoms(structure)

            # Compute the SOAP vector for the current structure
            soap_vector = soap_desc.create(atoms)

            soap_vectors_all.append(soap_vector)

        return soap_vectors_all

    def post_soap_selection(self, soap_vectors_all: list[np.ndarray]) -> np.ndarray:
        """Convert structures to SOAP vectors for specific atoms."""

        # Select specific atom indices by element
        selected_atom_indices = [
            i for i, atom in enumerate(soap_vectors_all[0][0])
            if atom.symbol in self.target_elements
        ]

        # Extract the relevant steps
        relevant_steps = soap_vectors_all[self.start_step:self.end_step:self.step_size]

        # Extract the SOAP vectors for the selected atoms at each step
        soap_vectors_selected_atoms = [soap_vectors_all[step] for step in relevant_steps]

        # Extract the SOAP vectors for all selected atoms at each step
        soap_vectors_all_selected = [vectors[selected_atom_indices] for vectors in soap_vectors_selected_atoms]

        # Convert the SOAP vectors to a numpy array
        soap_vectors_array = np.array(soap_vectors_all_selected)

        return soap_vectors_array

    def _dim_reduction(self, soap_vectors_array: list[np.ndarray], dim: int):
        """Perform dimension reduction on the SOAP vectors."""

        # Stack the SOAP vectors vertically
        soap_vectors_dim = np.vstack(soap_vectors_array)

        # Reshape the stacked SOAP vectors into a n-dimensional array
        soap_vectors_dim = soap_vectors_dim.reshape(len(soap_vectors_array), -1)

        # Perform PCA dimension reduction
        pca = PCA(n_components=dim)  # Set the desired number of components
        reduced_vectors = pca.fit_transform(soap_vectors_dim)

        return reduced_vectors

    def _cluster(self, data: list[Structure]):
        """Perform DBSCAN."""

        db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        db.fit(data)
        self.core_sample_indices_ = db.core_sample_indices_
        self.labels_ = db.labels_

        return self.labels_

        def _compute_core_ratio(self):
        """Compute the ratio of core data points to sample.

        ratio = min_samples/average_num_neighbors
        """

        core_points = self.reduced_vectors[self.core_samples_mask]

        # Compute the average number of neighbors for all core points
        neighbors_model = NearestNeighbors(**self.dbscan_kwargs)
        neighbors_model.fit(core_points)
        neighborhoods = neighbors_model.radius_neighbors(core_points, return_distance=False)
        average_neighbors_of_core_points = np.mean([len(neighbors) for neighbors in neighborhoods])

        # Calculate the ratio
        ratio = self.min_samples / average_neighbors_of_core_points

        def plot(self):
        """Function to plot the results of the selection.

        This can be called after the `sample` method to visualize the results.
        """

        labels = self.labels

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        unique_labels = set(self.labels)
        cmap = cm.get_cmap("Spectral")
        colors = [cmap(each) for each in np.linspace(0, 1, len(unique_labels))]

        # Access sampled noisy points from class attribute
        sampled_noisy_points = self.sampled_noisy_points
        sampled_reachable_points = self.sampled_reachable_points

        plt.scatter(
            [point[0] for point in sampled_noisy_points] + [point[0] for point in sampled_reachable_points],
            [point[1] for point in sampled_noisy_points] + [point[1] for point in sampled_reachable_points],
            color=['black'] * len(sampled_noisy_points) + ['gray'] * len(sampled_reachable_points),
            marker='o',
            s=30,
            label=['Noisy Points'] * len(sampled_noisy_points) + ['Reachable Points'] * len(sampled_reachable_points)
        )

        # Plot a sample of core points for each cluster
        for k, col in zip(unique_labels, colors):
            if k == -1:
                continue  # Skip the noisy points

            class_member_mask = labels == k
            core_points = self.reduced_vectors[class_member_mask & self.core_samples_mask]
            sample_size_core = int(self.core_ratio * len(core_points))
            sampled_points = np.random.choice(core_points, size=sample_size_core, replace=False)

            plt.scatter(
                [point[0] for point in sampled_points],
                [point[1] for point in sampled_points],
                color=tuple(col),
                marker='o',
                s=14,
                label=f'Cluster {k} (Sampled)'
            )

        plt.title(f"Estimated number of clusters: {n_clusters_}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()
