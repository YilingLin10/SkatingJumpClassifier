import numpy as np
import sys

from .tools import get_spatial_graph


# Edge format: (origin, neighbor)
num_node = 17
self_link = [(i, i) for i in range(num_node)]
inward = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6),
        (5, 7), (6, 8), (7, 9), (8, 10), (5, 11), (6, 12),
        (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, num_node, self_link, inward, outward, neighbor, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    A = Graph('spatial').get_adjacency_matrix()
    print('')