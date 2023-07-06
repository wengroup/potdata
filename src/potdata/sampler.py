"""Samplers to select a subset of objects (e.g. structures) from a sequence."""
import abc
from typing import Any, Callable, Iterable, List

import numpy as np
from monty.json import MSONable
from pymatgen.core.structure import Structure

from potdata.utils.dataops import serializable_slice, slice_sequence

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from ase.io import Trajectory
from dscribe.descriptors import SOAP
import random

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
        self.db_kwargs = dbscan_kwargs if dbscan_kwargs is None else {}
        self.noisy_ratio = noisy_ratio
        self.reachable_ratio = reachable_ratio
        self.core_ratio = core_ratio

        self._indices: list[int] = []

        np.random.seed(seed)

    def _get_soap_vectors(self, data: list[Structure]) -> list[np.ndarray]:
        """Convert structures to SOAP vectors.
           Take Li atom in Li3YCl6 as an example."""
        
        # Set up the SOAP descriptor
        soap_desc = SOAP(
            species=["Li", "Y", "Cl"],
            periodic=True, # Set periodicity to True for a periodic system
            r_cut=5.0,
            n_max=8,
            l_max=6,
            sigma=0.1
        )

        # Read the trajectory file
        traj = Trajectory(traj_file)

        Li_indices = [i for i, a in enumerate(traj[0]) if a.symbol == 'Li']

        # Extract the relevant steps, select every 5th in the last 6000 steps
        relevant_steps = traj[-6000::5]

        # Compute the SOAP vectors for the selected Li atoms at each step
        soap_vectors = [soap_desc.create(step) for step in relevant_steps]

        # Extract the SOAP vectors for all Li atom at each step
        soap_vectors_all = [vectors[Li_indices] for vectors in soap_vectors]

        # Convert the SOAP vectors to a numpy array
        soap_vectors_array = np.array(soap_vectors_all)

        # Save the SOAP vectors to a file
        np.save(save_file, soap_vectors_array)

        # Load the SOAP vectors from the .npy file
        soap_vectors = np.load(save_file)

        return soap_vectors

    def _dim_reduction(self, vectors: list[np.ndarray], dim: int):
        """Perform dimension reduction on the SOAP vectors."""

        # Reshape the SOAP vectors into a 2D array
        soap_vectors_2d = soap_vectors.reshape(soap_vectors.shape[0], -1)

        # Perform PCA dimension reduction
        pca = PCA(n_components=2)  # Set the desired number of components
        reduced_vectors = pca.fit_transform(soap_vectors_2d)
        
        return reduced_vectors

    def _cluster(self, data: list[Structure]):
        """Perform DBSCAN."""
        eps = self.db_kwargs.get('eps', 0.00019)
        min_samples = self.db_kwargs.get('min_samples', 4)

        db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
        labels = db.labels_

        return labels

    def sample(self, data: list[Structure]) -> list[Structure]:
        """"""
        soap_vectors = self._get_soap_vectors(data)

        if self.post_soap_selection is not None:
            soap_vectors = self.post_soap_selection(data, soap_vectors)

        # TODO, need to double check whether this correct
        vectors = np.concatenate(soap_vectors, axis=1)

        if self.pca_dim is not None:
            vectors = self._dim_reduction(vectors, self.pca_dim)

        labels = self._cluster(vectors)

        if self.core_ratio == "auto":
            core_ratio = self._compute_core_ratio()
        else:
            core_ratio = self.core_ratio

        ### TODO, select core, reachable, and noisy data points based on their ratios

        # `selected` below should be a list of pymatgen Structure
        selected = []

        return selected

    def sample_points(self, reduced_vectors, eps=0.00019, min_samples=4, noisy_ratio=1, reachable_ratio=1, core_ratio=0.3):
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(reduced_vectors)
        labels = db.labels_

        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True

        # Sample noisy points
        noisy_points = reduced_vectors[labels == -1]
        sample_size_noisy = int(noisy_ratio * len(noisy_points))
        sampled_noisy_points = random.sample(list(noisy_points), sample_size_noisy)

        # Sample reachable points
        reachable_points = reduced_vectors[np.logical_and(labels != -1, ~core_samples_mask)]
        sample_size_reachable = int(reachable_ratio * len(reachable_points))
        sampled_reachable_points = random.sample(list(reachable_points), sample_size_reachable)

        # Sample core points
        core_points = reduced_vectors[core_samples_mask]
        sample_size_core = int(core_ratio * len(core_points))
        sampled_core_points = random.sample(list(core_points), sample_size_core)

        return sampled_noisy_points, sampled_reachable_points, sampled_core_points, db
    
    def plot_clusters_with_sampling(self, eps=0.00019, min_samples=4, noisy_ratio=1, reachable_ratio=1, core_ratio=0.3):
        
        """Function to plot the results of the selection.

        This can be called after the `sample` method to visualize the results.
        """
        
        soap_vectors = self._get_soap_vectors(data)

        if self.post_soap_selection is not None:
            soap_vectors = self.post_soap_selection(data, soap_vectors)

        vectors = np.concatenate(soap_vectors, axis=1)

        if self.pca_dim is not None:
            vectors = self._dim_reduction(vectors, self.pca_dim)

        db = DBSCAN(eps=eps, min_samples=min_samples).fit(vectors)
        labels = db.labels_

        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

        # Plot noisy points and reachable points
        noisy_points = vectors[labels == -1]
        sampled_noisy_points = random.sample(list(noisy_points), int(noisy_ratio * len(noisy_points)))

        reachable_points = vectors[np.logical_and(labels != -1, ~core_samples_mask)]
        sampled_reachable_points = random.sample(list(reachable_points), int(reachable_ratio * len(reachable_points)))

        plt.scatter(
            [point[0] for point in sampled_noisy_points],
            [point[1] for point in sampled_noisy_points],
            color='black',
            marker='o',
            s=30,
            label='Noisy Points'
        )
        plt.scatter(
            [point[0] for point in sampled_reachable_points],
            [point[1] for point in sampled_reachable_points],
            color='gray',
            marker='o',
            s=30,
            label='Reachable Points'
        )

        # Plot a sample of core points for each cluster
        for k, col in zip(unique_labels, colors):
            if k == -1:
                continue  # Skip the noisy points

            class_member_mask = labels == k
            core_points = vectors[class_member_mask & core_samples_mask]
            sampled_points = random.sample(list(core_points), int(core_ratio * len(core_points)))

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
        
        plot_clusters_with_sampling(soap_vectors)

    def _compute_core_ratio(self, soap_vectors):
        """Compute the ratio of core data points to sample.

        ratio = min_samples/average_num_neighbors
        """
        
        eps = self.db_kwargs.get('eps', 0.00019)
        min_samples = self.db_kwargs.get('min_samples', 4)

        db = DBSCAN(eps=eps, min_samples=min_samples).fit(soap_vectors)
        labels = db.labels_

        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        core_points = soap_vectors[core_samples_mask]

        # Compute the average number of neighbors for all core points
        neighbors_model = NearestNeighbors(radius=db.eps)
        neighbors_model.fit(core_points)
        neighborhoods = neighbors_model.radius_neighbors(core_points, return_distance=False)
        average_neighbors_of_core_points = np.mean([len(neighbors) for neighbors in neighborhoods])

        # Calculate the ratio
        ratio = min_samples / average_neighbors_of_core_points

        return average_neighbors_of_core_points, ratio
        
        average_neighbors_of_core_points, ratio = _compute_core_ratio(soap_vectors)
        print("Average number of neighbors of core points:", average_neighbors_of_core_points)
        print("Ratio:", ratio)
