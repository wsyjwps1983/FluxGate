from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import faiss
from pydantic import ConfigDict, Field

from semantic_router.index.base import BaseIndex, IndexConfig
from semantic_router.linear import similarity_matrix, top_scores
from semantic_router.schema import ConfigParameter, SparseEmbedding, Utterance
from semantic_router.utils.logger import logger


class FaissIndex(BaseIndex):
    type: str = "faiss"
    metadata: Optional[np.ndarray] = Field(default=None, exclude=True)
    index: Any = Field(default=None, exclude=True)  # Faiss index object
    dimension: Optional[int] = Field(default=None, exclude=True)
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.metadata is None:
            self.metadata = None
        
    # Stop pydantic from complaining about Optional[np.ndarray] type hints.
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    def add(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[str],
        function_schemas: Optional[List[Dict[str, Any]]] = None,
        metadata_list: List[Dict[str, Any]] = [],
        **kwargs,
    ):
        """Add embeddings to the Faiss index.

        :param embeddings: List of embeddings to add to the index.
        :type embeddings: List[List[float]]
        :param routes: List of routes to add to the index.
        :type routes: List[str]
        :param utterances: List of utterances to add to the index.
        :type utterances: List[str]
        :param function_schemas: List of function schemas to add to the index.
        :type function_schemas: Optional[List[Dict[str, Any]]]
        :param metadata_list: List of metadata to add to the index.
        :type metadata_list: List[Dict[str, Any]]
        """
        embeds = np.array(embeddings, dtype=np.float32)  # type: ignore
        routes_arr = np.array(routes)
        if isinstance(utterances[0], str):
            utterances_arr = np.array(utterances)
        else:
            utterances_arr = np.array(utterances, dtype=object)
        
        # Initialize Faiss index if it doesn't exist
        if self.index is None:
            self.dimension = embeds.shape[1]
            # Create a flat L2 index for now (can be customized later)
            self.index = faiss.IndexFlatL2(self.dimension)
            self.routes = routes_arr
            self.utterances = utterances_arr
            self.metadata = (
                np.array(metadata_list, dtype=object)
                if metadata_list
                else np.array([{} for _ in utterances], dtype=object)
            )
            # Add embeddings to Faiss index
            self.index.add(embeds)
        else:
            # Check if dimension matches
            if embeds.shape[1] != self.dimension:
                raise ValueError(f"Embedding dimension mismatch: expected {self.dimension}, got {embeds.shape[1]}")
            # Add embeddings to Faiss index
            self.index.add(embeds)
            # Update routes, utterances, and metadata
            self.routes = np.concatenate([self.routes, routes_arr])
            self.utterances = np.concatenate([self.utterances, utterances_arr])
            if self.metadata is not None:
                self.metadata = np.concatenate(
                    [
                        self.metadata,
                        np.array(metadata_list, dtype=object)
                        if metadata_list
                        else np.array([{} for _ in utterances], dtype=object),
                    ]
                )
            else:
                self.metadata = (
                    np.array(metadata_list, dtype=object)
                    if metadata_list
                    else np.array([{} for _ in utterances], dtype=object)
                )

    def _remove_and_sync(self, routes_to_delete: dict) -> np.ndarray:
        """Remove and sync the index.

        :param routes_to_delete: Dictionary of routes to delete.
        :type routes_to_delete: dict
        :return: A numpy array of the removed route utterances.
        :rtype: np.ndarray
        """
        if self.index is None or self.routes is None or self.utterances is None:
            raise ValueError("Index, routes, or utterances are not populated.")
        
        route_utterances = np.array([self.routes, self.utterances]).T
        # Initialize our mask with all true values (ie keep all)
        mask = np.ones(len(route_utterances), dtype=bool)
        for route, utterances in routes_to_delete.items():
            for utterance in utterances:
                mask &= ~(
                    (route_utterances[:, 0] == route)
                    & (route_utterances[:, 1] == utterance)
                )
        
        # Create new Faiss index
        new_embeds = self.index.reconstruct_n(0, self.index.ntotal)[mask]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(new_embeds)
        
        # Update routes, utterances, and metadata
        self.routes = self.routes[mask]
        self.utterances = self.utterances[mask]
        if self.metadata is not None:
            self.metadata = self.metadata[mask]
        
        # Return what was removed
        return route_utterances[~mask]

    def get_utterances(self, include_metadata: bool = False) -> List[Utterance]:
        """Gets a list of route and utterance objects currently stored in the index.

        :param include_metadata: Whether to include function schemas and metadata in
        the returned Utterance objects - LocalIndex now includes metadata if present.
        :return: A list of Utterance objects.
        :rtype: List[Utterance]
        """
        if self.routes is None or self.utterances is None:
            return []
        if include_metadata and self.metadata is not None:
            return [
                Utterance(
                    route=route,
                    utterance=utterance,
                    function_schemas=None,
                    metadata=metadata,
                )
                for route, utterance, metadata in zip(
                    self.routes, self.utterances, self.metadata
                )
            ]
        else:
            return [Utterance.from_tuple(x) for x in zip(self.routes, self.utterances)]

    def describe(self) -> IndexConfig:
        """Describe the index.

        :return: An IndexConfig object.
        :rtype: IndexConfig
        """
        return IndexConfig(
            type=self.type,
            dimensions=self.dimension or 0,
            vectors=self.index.ntotal if self.index is not None else 0,
        )

    def is_ready(self) -> bool:
        """Checks if the index is ready to be used.

        :return: True if the index is ready, False otherwise.
        :rtype: bool
        """
        return self.index is not None and self.routes is not None

    async def ais_ready(self) -> bool:
        """Checks if the index is ready to be used asynchronously.

        :return: True if the index is ready, False otherwise.
        :rtype: bool
        """
        return self.index is not None and self.routes is not None

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Search the Faiss index for the query and return top_k results.

        :param vector: The vector to search for.
        :type vector: np.ndarray
        :param top_k: The number of results to return.
        :type top_k: int
        :param route_filter: The routes to filter the search by.
        :type route_filter: Optional[List[str]]
        :param sparse_vector: The sparse vector to search for.
        :type sparse_vector: dict[int, float] | SparseEmbedding | None
        :return: A tuple containing the query vector and a list of route names.
        :rtype: Tuple[np.ndarray, List[str]]
        """
        if self.index is None or self.routes is None:
            raise ValueError("Index or routes are not populated.")
        
        # Convert vector to float32
        vector = vector.astype(np.float32)
        
        if route_filter is not None:
            # Get indices of routes that match the filter
            filter_indices = np.array([i for i, route in enumerate(self.routes) if route in route_filter])
            if len(filter_indices) == 0:
                raise ValueError("No routes found matching the filter criteria.")
            
            # Create a sub-index for filtered routes
            sub_embeds = self.index.reconstruct_n(0, self.index.ntotal)[filter_indices]
            sub_index = faiss.IndexFlatL2(self.dimension)
            sub_index.add(sub_embeds)
            
            # Search in sub-index
            distances, sub_idx = sub_index.search(vector.reshape(1, -1), top_k)
            distances = distances[0]
            sub_idx = sub_idx[0]
            
            # Map back to original indices
            original_idx = filter_indices[sub_idx]
            scores = 1 / (1 + distances)  # Convert L2 distances to similarity scores
            route_names = [self.routes[i] for i in original_idx]
        else:
            # Search in full index
            distances, idx = self.index.search(vector.reshape(1, -1), top_k)
            distances = distances[0]
            idx = idx[0]
            scores = 1 / (1 + distances)  # Convert L2 distances to similarity scores
            route_names = [self.routes[i] for i in idx]
        
        return scores, route_names

    async def aquery(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: dict[int, float] | SparseEmbedding | None = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Search the index for the query and return top_k results asynchronously.

        :param vector: The vector to search for.
        :type vector: np.ndarray
        :param top_k: The number of results to return.
        :type top_k: int
        :param route_filter: The routes to filter the search by.
        :type route_filter: Optional[List[str]]
        :param sparse_vector: The sparse vector to search for.
        :type sparse_vector: dict[int, float] | SparseEmbedding | None
        :return: A tuple containing the query vector and a list of route names.
        :rtype: Tuple[np.ndarray, List[str]]
        """
        # For Faiss, async query is the same as sync query
        return self.query(vector, top_k, route_filter, sparse_vector)

    def aget_routes(self):
        """Get all routes from the index.

        :return: A list of routes.
        :rtype: List[str]
        """
        logger.error("Sync remove is not implemented for FaissIndex.")

    def _write_config(self, config: ConfigParameter):
        """Write the config to the index.

        :param config: The config to write to the index.
        :type config: ConfigParameter
        """
        logger.warning("No config is written for FaissIndex.")

    def delete(self, route_name: str):
        """Delete all records of a specific route from the index.

        :param route_name: The name of the route to delete.
        :type route_name: str
        """
        if (
            self.index is not None
            and self.routes is not None
            and self.utterances is not None
        ):
            # Get indices to delete
            delete_idx = self._get_indices_for_route(route_name=route_name)
            if not delete_idx:
                return  # No records to delete
            
            # Create a mask for records to keep
            mask = np.ones(len(self.routes), dtype=bool)
            mask[delete_idx] = False
            
            # Create new Faiss index
            new_embeds = self.index.reconstruct_n(0, self.index.ntotal)[mask]
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(new_embeds)
            
            # Update routes, utterances, and metadata
            self.routes = self.routes[mask]
            self.utterances = self.utterances[mask]
            if self.metadata is not None:
                self.metadata = self.metadata[mask]
        else:
            raise ValueError(
                "Attempted to delete route records but either index, routes or "
                "utterances is None."
            )

    async def adelete(self, route_name: str):
        """Delete all records of a specific route from the index asynchronously.

        :param route_name: The name of the route to delete.
        :type route_name: str
        """
        # For Faiss, async delete is the same as sync delete
        self.delete(route_name)

    def delete_index(self):
        """Deletes the index, effectively clearing it and setting it to None.

        :return: None
        :rtype: None
        """
        self.index = None
        self.routes = None
        self.utterances = None
        self.metadata = None
        self.dimension = None

    async def adelete_index(self):
        """Deletes the index, effectively clearing it and setting it to None asynchronously.

        :return: None
        :rtype: None
        """
        self.delete_index()

    def _get_indices_for_route(self, route_name: str):
        """Gets an array of indices for a specific route.

        :param route_name: The name of the route to get indices for.
        :type route_name: str
        :return: An array of indices for the route.
        :rtype: np.ndarray
        """
        if self.routes is None:
            raise ValueError("Routes are not populated.")
        idx = np.where(self.routes == route_name)[0]
        return idx

    def __len__(self):
        if self.index is not None:
            return self.index.ntotal
        else:
            return 0

    def get_vector_by_utterance(self, utterance: str) -> Optional[np.ndarray]:
        """Get the vector for a specific utterance.

        :param utterance: The utterance to get the vector for.
        :type utterance: str
        :return: The vector if found, None otherwise.
        :rtype: Optional[np.ndarray]
        """
        if self.index is None or self.utterances is None:
            return None
        
        # Get indices of matching utterances
        indices = np.where(self.utterances == utterance)[0]
        if len(indices) == 0:
            return None
        
        # Return the first matching vector
        return self.index.reconstruct(int(indices[0]))

    def get_utterance_by_vector(self, vector: np.ndarray, top_k: int = 1) -> List[Tuple[str, float]]:
        """Get the utterance(s) for a specific vector.

        :param vector: The vector to get the utterance for.
        :type vector: np.ndarray
        :param top_k: The number of results to return.
        :type top_k: int
        :return: A list of tuples containing utterance and similarity score.
        :rtype: List[Tuple[str, float]]
        """
        if self.index is None or self.utterances is None:
            return []
        
        # Convert vector to float32
        vector = vector.astype(np.float32)
        
        # Search for similar vectors
        distances, idx = self.index.search(vector.reshape(1, -1), top_k)
        distances = distances[0]
        idx = idx[0]
        
        # Convert distances to similarity scores
        scores = 1 / (1 + distances)
        
        # Return utterances with scores
        return [(self.utterances[i], float(scores[j])) for j, i in enumerate(idx)]