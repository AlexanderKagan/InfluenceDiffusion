import pandas as pd
import numpy as np
from typing import List, Tuple, Union, Dict
import networkx as nx

__all__ = ["Graph"]


class Graph(nx.DiGraph):
    def __init__(self, edge_list: List[Tuple[int, int]],
                 directed: bool = True,
                 weights: Union[np.array, List[float]] = None):
        """
        Initialize a directed or undirected graph from an edge list.

        Parameters
        ----------
        edge_list : List[Tuple[int, int]]
            List of tuples (source, sink) representing edges of the graph.
        directed : bool, optional
            Flag to indicate whether the graph is directed (default is True).
        weights : Union[None, List[float]], optional
            Optional weights for each edge. If None, all edges are assigned a weight of 1.
        """
        self.directed = directed
        self.edge_array = np.array(edge_list, dtype=int)

        if not self.directed:
            reverse_edge_array = self.edge_array[:, [1, 0]]
            self.edge_array = np.concatenate([self.edge_array, reverse_edge_array])

        super().__init__(self.edge_array.tolist())

        if weights is not None:
            assert len(weights) == len(edge_list), "number of edges is different from number of weights"
        else:
            weights = np.ones(len(edge_list))
        self.weights = np.array(weights) if directed else np.hstack([weights] * 2)
        self.edge_2_index = {tuple(edge): idx for idx, edge in enumerate(self.edge_array)}

    def rename_graph_vertices(self, old_2_new_vertex_name_dict: Union[None, Dict] = None) -> "Graph":
        """
        Rename graph vertices according to the provided mapping or re-index them.

        Parameters
        ----------
        old_2_new_vertex_name_dict : Union[None, Dict], optional
            Mapping of old vertex names to new names. 
            If None, re-indexes vertices to [0, 1, ..., |V|-1].

        Returns
        -------
        Graph
            The updated graph instance with renamed vertices.
        """
        if old_2_new_vertex_name_dict is None:
            old_vertex_names = sorted(list(self.get_vertices()))
            old_2_new_vertex_name_dict = {name: idx for idx, name in enumerate(old_vertex_names)}

        self.edge_array[:, 0] = list(map(lambda v: old_2_new_vertex_name_dict[v], self.edge_array[:, 0]))
        self.edge_array[:, 1] = list(map(lambda v: old_2_new_vertex_name_dict[v], self.edge_array[:, 1]))
        return self

    def set_weights(self, weights: Union[str, np.array]) -> "Graph":
        """
        Set weights for the edges in the graph.

        Parameters
        ----------
        weights : Union[str, np.array]
            Array of weights for the edges.

        Returns
        -------
        Graph
            The current graph instance with updated weights.
        
        Raises
        ------
        AssertionError
            If the number of weights does not match the number of edges.
        """
        assert len(weights) == self.count_edges(), "number of weights different from the number of edges"
        self.weights = np.array(weights)
        return self

    def get_edges(self, as_array: bool = False) -> Union[np.array, List[Tuple[int, int]]]:
        """
        Retrieve the edges of the graph.

        Parameters
        ----------
        as_array : bool, optional
            If True, returns edges as a numpy array. If False, returns as a list of tuples (default is False).

        Returns
        -------
        Union[np.array, List[Tuple[int, int]]]
            The edges of the graph in the specified format.
        """
        if as_array:
            return self.edge_array
        return [tuple(edge) for edge in self.edge_array]

    def get_vertices(self) -> set:
        """
        Get the set of all vertices in the graph.

        Returns
        -------
        set
            A set containing all vertex identifiers.
        """
        return set(self.nodes)
    
    def get_sources(self) -> set: 
        """
        Get the set of source vertices in the graph.

        Returns
        -------
        set
            A set containing all source vertices.
        """
        return set(self.edge_array[:, 0])
    
    def get_sinks(self) -> set: 
        """
        Get the set of sink vertices in the graph.

        Returns
        -------
        set
            A set containing all sink vertices.
        """
        return set(self.edge_array[:, 1])

    def count_edges(self) -> int:
        """
        Count the number of edges in the graph.

        Returns
        -------
        int
            The total number of edges.
        """
        return len(self.edge_array)

    def count_vertices(self) -> int:
        """
        Count the number of vertices in the graph.

        Returns
        -------
        int
            The total number of vertices.
        """
        return len(self.get_vertices())

    def get_children_mask(self, vertex: int) -> np.array:
        """
        Create a mask for edges originating from a specified vertex.

        Parameters
        ----------
        vertex : int
            The vertex for which to find children.

        Returns
        -------
        np.array
            A boolean array mask indicating edges originating from the vertex.
        """
        return self.edge_array[:, 0] == vertex

    def get_children(self, vertex: int, out_type: Union[set, list, np.array] = set) -> Union[set, list, np.array]:
        """
        Get the children (outgoing edges) of a specified vertex.

        Parameters
        ----------
        vertex : int
            The vertex for which to find children.
        out_type : Union[set, list, np.array], optional
            The type of output to return (default is set).

        Returns
        -------
        Union[set, list, np.array]
            The children of the vertex in the specified output type.
        """
        return out_type(self.edge_array[:, 1][self.get_children_mask(vertex)])

    def get_parents_mask(self, vertex: int) -> np.array:
        """
        Create a mask for edges leading to a specified vertex.

        Parameters
        ----------
        vertex : int
            The vertex for which to find parents.

        Returns
        -------
        np.array
            A boolean array mask indicating edges leading to the vertex.
        """
        return self.edge_array[:, 1] == vertex

    def get_parents(self, vertex: int, out_type: Union[set, list, np.array] = set) -> Union[set, list, np.array]:
        """
        Get the parents (incoming edges) of a specified vertex.

        Parameters
        ----------
        vertex : int
            The vertex for which to find parents.
        out_type : Union[set, list, np.array], optional
            The type of output to return (default is set).

        Returns
        -------
        Union[set, list, np.array]
            The parents of the vertex in the specified output type.
        """
        return out_type(self.edge_array[:, 0][self.get_parents_mask(vertex)])

    def get_vertex_2_indegree_dict(self, weighted: bool = False) -> Dict[int, Union[float, int]]:
        """
        Get a dictionary mapping vertices to their indegrees.

        Parameters
        ----------
        weighted : bool, optional
            If True, returns weighted indegrees; otherwise, returns simple counts (default is False).

        Returns
        -------
        Dict[int, Union[float, int]]
            A dictionary where keys are vertex identifiers and values are their indegrees.
        """
        return {vertex: self.get_indegree(vertex, weighted=weighted) for vertex in self.get_vertices()}

    def get_outdegree(self, vertex: int, weighted: bool = False) -> Union[float, int]:
        """
        Get the outdegree of a specified vertex.

        Parameters
        ----------
        vertex : int
            The vertex for which to find the outdegree.
        weighted : bool, optional
            If True, returns the weighted outdegree; otherwise, returns simple counts (default is False).

        Returns
        -------
        Union[float, int]
            The outdegree of the vertex.
        """
        children_weights = self.weights[self.edge_array[:, 0] == vertex]
        return np.sum(children_weights) if weighted else len(children_weights)

    def get_all_outdegrees(self, weighted: bool = False) -> np.array:
        """
        Get outdegrees for all vertices in the graph.

        Parameters
        ----------
        weighted : bool, optional
            If True, returns weighted outdegrees; otherwise, returns simple counts (default is False).

        Returns
        -------
        np.array
            An array of outdegrees for all vertices.
        """
        return np.array([self.get_outdegree(v, weighted=weighted) for v in self.get_vertices()])

    def get_avg_outdegree(self, weighted: bool = False) -> Union[float, int]:
        """
        Calculate the average outdegree of all vertices.

        Parameters
        ----------
        weighted : bool, optional
            If True, uses weighted outdegrees; otherwise, uses simple counts (default is False).

        Returns
        -------
        Union[float, int]
            The average outdegree of the vertices.
        """
        return np.mean(self.get_all_outdegrees(weighted=weighted))

    def get_max_outdegree(self, weighted: bool = False) -> Union[float, int]:
        """
        Get the maximum outdegree among all vertices.

        Parameters
        ----------
        weighted : bool, optional
            If True, considers weighted outdegrees; otherwise, uses simple counts (default is False).

        Returns
        -------
        Union[float, int]
            The maximum outdegree.
        """
        return np.max(self.get_all_outdegrees(weighted))

    def get_indegree(self, vertex: int, weighted: bool = False) -> Union[float, int]:
        """
        Get the indegree of a specified vertex.

        Parameters
        ----------
        vertex : int
            The vertex for which to find the indegree.
        weighted : bool, optional
            If True, returns the weighted indegree; otherwise, returns simple counts (default is False).

        Returns
        -------
        Union[float, int]
            The indegree of the vertex.
        """
        parent_weights = self.weights[self.edge_array[:, 1] == vertex]
        return np.sum(parent_weights) if weighted else len(parent_weights)
    
    def get_indegrees_dict(self, weighted: bool = False) -> Dict[int, Union[int, float]]:
        """
        Get a mapping from vertex to its indegree for all vertices in the graph.

        Parameters
        ----------
        weighted : bool, optional
            If True, returns weighted indegrees; otherwise, returns simple counts (default is False).

        Returns
        -------
        np.array
            An array of indegrees for all vertices.
        """
        return {v: self.get_indegree(v, weighted=weighted) for v in self.get_vertices()}

    def get_all_indegrees(self, weighted: bool = False) -> np.array:
        """
        Get indegrees for all vertices in the graph.

        Parameters
        ----------
        weighted : bool, optional
            If True, returns weighted indegrees; otherwise, returns simple counts (default is False).

        Returns
        -------
        np.array
            An array of indegrees for all vertices.
        """
        indegrees_dict = self.get_indegrees_dict(weighted=weighted)
        return np.array(list(indegrees_dict.values()))

    def get_max_indegree(self, weighted: bool = False) -> Union[int, float]:
        """
        Get the maximum indegree among all vertices.

        Parameters
        ----------
        weighted : bool, optional
            If True, considers weighted indegrees; otherwise, uses simple counts (default is False).

        Returns
        -------
        Union[int, float]
            The maximum indegree.
        """
        return np.max(self.get_all_indegrees(weighted))

    def get_avg_indegree(self, weighted: bool = False) -> float:
        """
        Calculate the average indegree of all vertices.

        Parameters
        ----------
        weighted : bool, optional
            If True, uses weighted indegrees; otherwise, uses simple counts (default is False).

        Returns
        -------
        float
            The average indegree of the vertices.
        """
        return np.mean(self.get_all_indegrees(weighted))

    def get_edge_index(self, edge: Tuple[int, int]) -> int:
        """
        Get the index of a specified edge.

        Parameters
        ----------
        edge : Tuple[int, int]
            The edge for which to find the index.

        Returns
        -------
        int
            The index of the edge in the graph.
        """
        return self.edge_2_index[tuple(edge)]

    def get_edge_weight(self, edge: Tuple[int, int]) -> float:
        """
        Get the weight of a specified edge.

        Parameters
        ----------
        edge : Tuple[int, int]
            The edge for which to retrieve the weight.

        Returns
        -------
        float
            The weight of the edge.
        """
        return self.weights[self.get_edge_index(edge)]

    def get_edges_mask_from_set_to_vertex(self, vertex_set: set, vertex: int) -> np.array:
        """
        Create a mask for edges leading from a set of vertices to a specified vertex.

        Parameters
        ----------
        vertex_set : set
            A set of source vertices.
        vertex : int
            The target vertex.

        Returns
        -------
        np.array
            A boolean array mask indicating which edges lead from vertices in vertex_set to the specified vertex.
        """
        parent_edges_mask = self.get_parents_mask(vertex)
        parent_vertices = self.edge_array[:, 0][parent_edges_mask]
        parent_edges_mask[parent_edges_mask] = [(parent in vertex_set) for parent in parent_vertices]
        return parent_edges_mask

    def get_adj_matrix(self) -> np.array:
        """
        Generate the adjacency matrix of the graph.

        Returns
        -------
        np.array
            The adjacency matrix as a numpy array.
        """
        vertices = list(self.get_vertices())
        n_vertex = len(vertices)
        adj_matrix = pd.DataFrame(np.zeros((n_vertex, n_vertex)), columns=vertices, index=vertices)
        for (v1, v2), weight in zip(self.edge_array, self.weights):
            adj_matrix.at[v1, v2] = weight

        return adj_matrix

    def is_edge_in_graph(self, edge: Tuple[int, int]) -> bool:
        """
        Check if a specified edge exists in the graph.

        Parameters
        ----------
        edge : Tuple[int, int]
            The edge to check.

        Returns
        -------
        bool
            True if the edge exists in the graph, False otherwise.
        """
        source, sink = edge
        return sink in self.get_children(source)
