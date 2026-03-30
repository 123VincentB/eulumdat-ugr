# -*- coding: utf-8 -*-
"""
geometry.py
-----------
UgrGrid — luminaire grid geometry for UGR catalogue calculation.

Observer is at the midpoint of the short wall (y_R = 0, x_T = 0), looking
in the +Y direction.  All luminaires are placed at y_i > 0 (in front of the
observer), symmetric about the X axis.

CIE 117:1995 Equations
-----------------------
4.5  crosswise  C = arctan(T / R)
4.6  endwise    C = 90° − arctan(T / R)
4.7  gamma      γ = arctan(√(R² + T²) / H)
4.8  r²         r² = R² + T² + H²

Filters (CIE 117 §4.5)
-----------------------
- T/R > 3.00  →  luminaire ignored in UGR sum
- γ  > 85°    →  luminaire ignored in UGR sum
"""

import numpy as np

# Fixed geometry (CIE 190:2010 §4.2)
H_MOUNT: float = 2.0     # m — luminaire height above observer eye level
GAMMA_MAX: float = 85.0  # degrees — elevation cut-off angle
TR_MAX: float = 3.0      # T/R cut-off ratio


class UgrGrid:
    """
    Grid of luminaire positions for one room configuration.

    Grid spacing S = shr × H.  Positions on the T axis (perpendicular to
    sight) are symmetric: ±S/2, ±3S/2, …, ±(Nx/2 − 1/2)·S.
    Positions on the R axis (parallel to sight): S/2, 3S/2, …, Ny·S − S/2.

    Parameters
    ----------
    x_dim_H : float
        Room X dimension as a multiple of H (perpendicular to line of sight).
    y_dim_H : float
        Room Y dimension as a multiple of H (parallel to line of sight).
    shr : float, optional
        Spacing-to-Height Ratio.  Default 0.25 (catalogue values per CIE 117).
        Use 1.0 to reproduce CIE 190:2010 numerical validation examples.

    Attributes
    ----------
    H : float
        Luminaire height above observer eye level [m] (fixed = 2.0).
    s : float
        Grid spacing [m] = shr × H.
    n_total : int
        Total number of luminaires in the grid (before filtering).
    """

    H: float = H_MOUNT

    def __init__(self, x_dim_H: float, y_dim_H: float, shr: float = 0.25) -> None:
        self._x_dim_H = float(x_dim_H)
        self._y_dim_H = float(y_dim_H)
        self._shr = float(shr)

        H = self.H
        s = shr * H
        x_room = x_dim_H * H   # room width [m], perpendicular to line of sight
        y_room = y_dim_H * H   # room depth [m], parallel to line of sight

        # T axis (X, perpendicular): symmetric positions ±S/2, ±3S/2, …
        n_half = round(x_room / s / 2)        # positions per side
        t_half = (np.arange(n_half) + 0.5) * s   # S/2, 3S/2, …
        t_vals = np.concatenate([-t_half[::-1], t_half])   # negative then positive

        # R axis (Y, parallel): S/2, 3S/2, …, up to y_room
        n_r = round(y_room / s)
        r_vals = (np.arange(n_r) + 0.5) * s       # S/2, 3S/2, …

        # Full Cartesian grid (every combination of T and R)
        T_grid, R_grid = np.meshgrid(t_vals, r_vals)
        self._t_all: np.ndarray = T_grid.ravel()
        self._r_all: np.ndarray = R_grid.ravel()

        self.s: float = s
        self.n_total: int = len(self._r_all)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def angles(
        self, orientation: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Geometry of luminaires that pass the T/R and γ filters.

        Parameters
        ----------
        orientation : {'crosswise', 'endwise'}
            'crosswise' — luminaire long axis ⊥ line of sight (CIE 117 eq. 4.5).
            'endwise'   — luminaire long axis ∥ line of sight  (CIE 117 eq. 4.6).

        Returns
        -------
        R, T, C_deg, gamma_deg, r_m : np.ndarray
            Arrays of shape (n_valid,) for luminaires that pass both filters.

            R         — depth [m], parallel to line of sight (always > 0).
            T         — absolute transverse distance [m] (|x_i|).
            C_deg     — photometric azimuth C [°], 0 ≤ C ≤ 90.
            gamma_deg — elevation angle γ from nadir [°], 0 ≤ γ < 90.
            r_m       — slant distance r [m] from luminaire to observer.
        """
        R = self._r_all
        T = np.abs(self._t_all)
        H = self.H

        hor_dist = np.sqrt(R**2 + T**2)
        r_m = np.sqrt(hor_dist**2 + H**2)
        gamma_deg = np.degrees(np.arctan2(hor_dist, H))

        if orientation == "crosswise":
            # CIE 117 eq. 4.5
            C_deg = np.degrees(np.arctan2(T, R))
        elif orientation == "endwise":
            # CIE 117 eq. 4.6
            C_deg = 90.0 - np.degrees(np.arctan2(T, R))
        else:
            raise ValueError(
                f"orientation must be 'crosswise' or 'endwise', got {orientation!r}"
            )

        # Apply filters (R > 0 always, so T/R is well-defined)
        tr = T / R
        mask = (tr <= TR_MAX) & (gamma_deg <= GAMMA_MAX)

        return R[mask], T[mask], C_deg[mask], gamma_deg[mask], r_m[mask]

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_x(self) -> int:
        """Total number of luminaire columns (X axis, perpendicular to sight)."""
        return len(np.unique(self._t_all))

    @property
    def n_y(self) -> int:
        """Total number of luminaire rows (Y axis, parallel to sight)."""
        return len(np.unique(self._r_all))
