"""
A four-coloring of the Einstein tiling.

Ref:
    "An Aperiodic Monotile" by Smith, et al.
    https://cs.uwaterloo.ca/~csk/hat/
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pysat.solvers import Glucose3

class Tile(object):
    def __init__(self, tiles, bounds):
        self.tiles = tiles
        self.bounds = bounds
        
    def add(self, dx):
        self.bounds += dx
        for tile in self.tiles:
            tile.add(dx)
        return self
            
    def mul(self, M):
        self.bounds = self.bounds @ M
        for tile in self.tiles:
            tile.mul(M)
        return self
            
    def copy(self):
        return Tile(
            [tile.copy() for tile in self.tiles],
            self.bounds.copy()
        )
            
    def flatten(self):
        tiles = []
        for tile in self.tiles:
            tiles += tile.flatten()
        return tiles
    
class Base(Tile):
    def __init__(self, x, bounds):
        self.x = x
        self.bounds = bounds
        
    def add(self, dx):
        self.x += dx
        self.bounds += dx
        return self
        
    def mul(self, M):
        self.x = self.x @ M
        self.bounds = self.bounds @ M
        return self
    
    def copy(self):
        return Base(self.x.copy(), self.bounds.copy())
        
    def flatten(self):
        return [self]

def edges(a, b):
    return np.array([
        [a, 0], [a, 2], [b, 11], [b, 1], [a, 4], [a, 2], [b, 5],
        [b, 3], [a, 6], [a, 8], [a, 8], [a, 10], [b, 7], #[b, 9]
    ], dtype=float)
    
def coords(e):
    dx = e.T[0] * np.cos(e.T[1] * np.pi / 6)
    dy = e.T[0] * np.sin(e.T[1] * np.pi / 6)
    
    x = np.r_[[0], np.cumsum(dx)]
    y = np.r_[[0], np.cumsum(dy)]
    
    return np.c_[x, y]

def bases(a, b):
    # Edges of a single tile in (length, angle) format:
    e = np.array([
        [a, 0], [a, 2], [b, 11], [b, 1], [a, 4], [a, 2], [b, 5],
        [b, 3], [a, 6], [a, 8], [a, 8], [a, 10], [b, 7], #[b, 9]
    ])
    
    # Get the tile coordinates
    dx = e.T[0] * np.cos(e.T[1] * np.pi / 6)
    dy = e.T[0] * np.sin(e.T[1] * np.pi / 6)
    
    x = np.r_[[0], np.cumsum(dx)]
    y = np.r_[[0], np.cumsum(dy)]
    
    x = np.c_[x, y]
    
    # The quasi-boundary comes from four points
    b = x[(1, 3, 9, 13), :]
    
    # Get the flipped tile
    y = x[::-1].copy()
    y[:, 1] *= -1
    
    # And move it to be flush with the original tile
    y += x[0] - y[5]
    
    # Return two base tiles: the single (H8) and the compound (H7)
    return Base(x, b), Tile([Base(x, b), Base(y, b)], b)

def supers(subtiles):
    # The rules are based on Figure 2.11
    rules = [
		[1, 2, 0, 0],
		[2, 2, 0, 0],
		[0, 1, 1, 1],
		[4, 2, 2, 0],
		[5, 2, 0, 0], 
		[0, 2, 0, 0],
    ]
    
    tiles = [subtiles[0]]
    for angle, pivot, anchor, subtile in rules:
        # Rotation matrix
        c = np.cos(angle * np.pi / 3)
        s = np.sin(angle * np.pi / 3)
        M = np.array([[c, s], [-s, c]])
        
        # Rotate and translate subtile
        T = subtiles[subtile].copy().mul(M)
        T.add(tiles[-1].bounds[anchor] - T.bounds[pivot])
        
        tiles.append(T)
    
    bounds = [tiles[1].bounds[3], tiles[2].bounds[0],
              tiles[4].bounds[3], tiles[6].bounds[0]]
    
    return Tile(tiles, bounds), Tile(tiles[:-1], bounds)

def neighbors(tiles):
    e = []
    for i, t1 in enumerate(tiles):
        for j, t2 in enumerate(tiles):
            if i <= j: continue
            if np.sum(np.all(np.isclose(t1.x[:, None, :], t2.x), axis=-1)) >= 2:
                e.append((i, j))
    return e

def four_color_sat(edges):
    """
    A valid four-coloring has the following constraints:
    
    (1) Vertex v has color 1, 2, 3, or 4:
        (4v + 1) | (4v + 2) | (4v + 3) | (4v + 4)
        
    (2) Vertex v has at most one of these colors:
        ~(4v + i) | ~(4v + j)
        
    (3) Adjacent vertices u, v have different colors:
        ~(4u + i) | ~(4v + i)
    """
    
    solver = Glucose3()
    
    # Number of vertices
    n = max(max(edge) for edge in edges) + 1
    
    for v in range(n):
        colors = [(4*v + i + 1) for i in range(4)]

        # Constraint (1)
        solver.add_clause(colors)
        
        # Constraint (2)
        for i in range(4):
            for j in range(i+1, 4):
                solver.add_clause([-colors[i], -colors[j]])
    
    # Constraint (3)
    for u, v in edges:
        for color in range(1, 5):
            solver.add_clause([-(4*u + color), -(4*v + color)])
    
    if solver.solve():
        model = solver.get_model()
        colors = {}
        for v in range(n):
            for c in range(4):
                if model[4*v + c] > 0:
                    colors[v] = c
                    break
        return colors
    return None

def circle(tiles, radius=25):
    # Center of mass
    G = sum(t.x for t in tiles).mean(axis=0) / len(tiles)
    def within_radius(t):
        return np.linalg.norm(t.x.mean(axis=0) - G) < radius
    return list(filter(within_radius, tiles))

if __name__ == '__main__':
    H = bases(1, 3**0.5)
    for i in range(3):
        H = supers(H)

    tiles = H[0].flatten()
    # tiles = circle(tiles, radius=25)
    e = neighbors(tiles)
    c = four_color_sat(e)

    fig, ax = plt.subplots()
    colors = [(1, 0, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1)]
    for i, tile in enumerate(tiles):
        poly = Polygon(tile.x, edgecolor='black', facecolor=colors[c[i]], lw=1)
        ax.add_patch(poly)
    ax.set_aspect('equal')
    ax.autoscale()
    ax.set_axis_off()

    plt.savefig('einstein.png', pad_inches=0, transparent=True, bbox_inches='tight', dpi=600)
    plt.show()