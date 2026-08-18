"""Classic frontier detection and goal selection (Yamauchi FBE).

Ported 2026-08-17 from the slam-frontier probe worktree
(slam_frontier/frontier.py verbatim, imports package-relative).

A frontier cell is a FREE cell with at least one UNKNOWN 4-neighbour.
Clusters are 8-connected components of the frontier mask; the goal is the
representative cell (closest-to-centroid member) of the cluster whose
representative is nearest by BFS path distance over traversable cells.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ._mapping import FREE, UNKNOWN


def frontier_mask(grid: np.ndarray) -> np.ndarray:
    free = grid == FREE
    unknown = grid == UNKNOWN
    near_unknown = np.zeros_like(free)
    near_unknown[1:, :] |= unknown[:-1, :]
    near_unknown[:-1, :] |= unknown[1:, :]
    near_unknown[:, 1:] |= unknown[:, :-1]
    near_unknown[:, :-1] |= unknown[:, 1:]
    return free & near_unknown


def cluster_frontiers(mask: np.ndarray, min_size: int = 6) -> list:
    """8-connected components; each returned as an (K, 2) array of (zi, xi)."""
    visited = np.zeros_like(mask, dtype=bool)
    clusters = []
    n, m = mask.shape
    for zi, xi in zip(*np.nonzero(mask)):  # noqa: B905 — py3.9 env
        if visited[zi, xi]:
            continue
        comp = []
        q = deque([(zi, xi)])
        visited[zi, xi] = True
        while q:
            cz, cx = q.popleft()
            comp.append((cz, cx))
            for dz in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nz, nx = cz + dz, cx + dx
                    if 0 <= nz < n and 0 <= nx < m and mask[nz, nx] and not visited[nz, nx]:
                        visited[nz, nx] = True
                        q.append((nz, nx))
        if len(comp) >= min_size:
            clusters.append(np.asarray(comp))
    return clusters


def cluster_representative(comp: np.ndarray) -> tuple:
    centroid = comp.mean(axis=0)
    d = np.abs(comp - centroid).sum(axis=1)
    zi, xi = comp[int(np.argmin(d))]
    return int(zi), int(xi)
