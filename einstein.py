"""
A four-coloring of the Einstein tiling.

Ref:
    "An Aperiodic Monotile" by Smith, et al.
    https://cs.uwaterloo.ca/~csk/hat/
"""

import numpy as np
from fractions import Fraction as F
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pysat.solvers import Glucose3

class Qr3(object):
    def __init__(self, a: F, b: F):
        self.a = a
        self.b = b
    
    @property
    def real(self):
        return self.a + 1.73205080757 * self.b
    
    def __repr__(self):
        if self.a == 0 and self.b == 0:
            return "0"
        
        if self.b == 0:
            return str(self.a)
        
        b = "√3" if abs(self.b) == 1 else f"{abs(self.b)}√3"
        if self.a == 0:
            return b
        
        return f"{self.a} + {b}" if self.b > 0 else f"{self.a} - {b}"
    
    def __eq__(self, other):
        return self.a == other.a and self.b == other.b
    
    def __add__(self, other):
        return Qr3(self.a + other.a, self.b + other.b)
    
    def __sub__(self, other):
        return Qr3(self.a - other.a, self.b - other.b)
    
    def __mul__(self, other):
        return Qr3(self.a * other.a + 3 * self.b * other.b, self.a * other.b + self.b * other.a)
    
    def __div__(self, other):
        a = self.a * other.a - 3 * self.b * other.b
        b = self.b * other.a - self.a * other.b
        d = other.a**2 - 3 * other.b**2
        return Qr3(a / d, b / d)

ONE = np.array([Qr3(1, 0), Qr3(0, 0)])
ROT30 = np.array([[Qr3(0, F(1,2)), Qr3(F(1,2), 0)],
                  [Qr3(F(-1,2), 0), Qr3(0, F(1,2))]])

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

def bases(a, b):
    # Edges of a single tile in (length, angle) format:
    e = [
        [a, 12], [a, 10], [b, 1], [b, 11], [a, 8], [a, 10], [b, 7],
        [b, 9], [a, 6], [a, 4], [a, 4], [a, 2], [b, 5],
    ]
    
    # Get the tile coordinates
    dx = np.array([np.linalg.matrix_power(ROT30, angle) @ r for r, angle in e])
    x = np.r_[[[Qr3(0,0), Qr3(0,0)]], np.cumsum(dx, axis=0)]
    
    # The quasi-boundary comes from four points
    b = x[(1, 3, 9, 13), :]
    
    # Get the flipped tile
    y = x[::-1].copy()
    y[:, 1] *= Qr3(-1, 0)
    
    # And move it to be flush with the original tile
    y += x[0] - y[5]
    
    # Return two base tiles: the single (H8) and the compound (H7)
    return Base(x, b), Tile([Base(x, b), Base(y, b)], b)

def supers(subtiles):
    # The rules are based on Figure 2.11
    rules = [
		[2, 2, 0, 0],
		[4, 2, 0, 0],
		[12, 1, 1, 1],
		[8, 2, 2, 0],
		[10, 2, 0, 0], 
		[12, 2, 0, 0],
    ]
    
    tiles = [subtiles[0]]
    for angle, pivot, anchor, subtile in rules:
        # Rotation matrix
        M = np.linalg.matrix_power(ROT30, angle)
        
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
            if np.sum(np.all(t1.x[:, None, :] == t2.x, axis=-1)) >= 2:
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
    H = bases(ONE, ONE)
    for i in range(3):
        H = supers(H)

    tiles = H[0].flatten()
    # tiles = circle(tiles, radius=25)
    e = neighbors(tiles)
    c = four_color_sat(e)

    fig, ax = plt.subplots()
    colors = [(1, 0, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1)]
    for i, tile in enumerate(tiles):
        points = [[x.real for x in point] for point in tile.x]
        poly = Polygon(points, edgecolor='black', facecolor=colors[c[i]], lw=1)
        ax.add_patch(poly)
    ax.set_aspect('equal')
    ax.autoscale()
    ax.set_axis_off()

    plt.savefig('einstein.png', pad_inches=0, transparent=True, bbox_inches='tight', dpi=600)
    plt.show()