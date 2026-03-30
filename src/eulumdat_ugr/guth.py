# -*- coding: utf-8 -*-
"""
guth.py
-------
Guth position index table (CIE 117:1995, Table 4.1).

The Guth position index p weights each luminaire's glare contribution
in the UGR formula:

    UGR = 8 · log10((0.25 / Lb) · Σ (L²_i · ω_i / p²_i))

A larger p means the luminaire is in a less critical position (further
from the line of sight), reducing its UGR contribution.

Table axes
----------
- Rows  : T/R — transverse distance / horizontal viewing distance [0.00, 0.10, …, 3.00]
- Cols  : H/R — luminaire height / horizontal viewing distance    [0.00, 0.10, …, 1.90]

Both inputs are absolute values (the table is symmetric in T/R sign).

Missing cells (marked '-' in the standard) are stored as np.nan.
Interpolation at or adjacent to a NaN cell returns np.nan → the
luminaire is ignored in the UGR sum (same as T/R > 3 or γ > 85°).
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class GuthTable:
    """
    Guth position index table from CIE 117:1995, Table 4.1.

    Usage
    -----
    ::

        p = GuthTable.p(h_r=0.5, t_r=1.0)   # float or np.nan

    The interpolator is built once on first call and cached.
    """

    # T/R axis — rows, 0.00 to 3.00 in steps of 0.10 (31 values)
    _TR_AXIS: np.ndarray = np.round(np.arange(0.0, 3.01, 0.10), 2)

    # H/R axis — columns, 0.00 to 1.90 in steps of 0.10 (20 values)
    _HR_AXIS: np.ndarray = np.round(np.arange(0.0, 1.91, 0.10), 2)

    # Table 4.1, shape (31, 20).  np.nan = missing cell ('-' in standard).
    # Source: CIE 117:1995, Table 4.1 — transcribed from the official document.
    _TABLE: np.ndarray = np.array([
        # H/R→ 0.00   0.10   0.20   0.30   0.40   0.50   0.60   0.70   0.80   0.90   1.00   1.10   1.20   1.30   1.40   1.50   1.60   1.70   1.80   1.90
        # T/R=0.00
        [ 1.00,  1.26,  1.53,  1.90,  2.35,  2.86,  3.50,  4.20,  5.00,  6.00,  7.00,  8.10,  9.25, 10.35, 11.70, 13.15, 14.70, 16.20,   np.nan, np.nan],
        # T/R=0.10
        [ 1.05,  1.22,  1.46,  1.80,  2.20,  2.75,  3.40,  4.10,  4.80,  5.80,  6.80,  8.00,  9.10, 10.30, 11.60, 13.00, 14.60, 16.10,   np.nan, np.nan],
        # T/R=0.20
        [ 1.12,  1.30,  1.50,  1.80,  2.20,  2.66,  3.18,  3.88,  4.60,  5.50,  6.50,  7.60,  8.75,  9.85, 11.20, 12.70, 14.00, 15.70,   np.nan, np.nan],
        # T/R=0.30
        [ 1.22,  1.38,  1.60,  1.97,  2.25,  2.70,  3.25,  3.90,  4.60,  5.45,  6.45,  7.40,  8.40,  9.50, 10.85, 12.10, 13.70, 15.00,   np.nan, np.nan],
        # T/R=0.40
        [ 1.32,  1.47,  1.70,  1.96,  2.35,  2.80,  3.30,  3.90,  4.60,  5.40,  6.40,  7.30,  8.30,  9.40, 10.60, 11.90, 13.20, 14.60,   16.00,  np.nan],
        # T/R=0.50
        [ 1.43,  1.60,  1.82,  2.10,  2.48,  2.91,  3.40,  3.98,  4.70,  5.50,  6.40,  7.30,  8.30,  9.40, 10.50, 11.75, 13.00, 14.40,   15.70,  np.nan],
        # T/R=0.60
        [ 1.55,  1.72,  1.98,  2.30,  2.65,  3.10,  3.60,  4.10,  4.80,  5.50,  6.40,  7.35,  8.40,  9.40, 10.50, 11.70, 13.00, 14.10,   15.40,  np.nan],
        # T/R=0.70
        [ 1.70,  1.88,  2.12,  2.48,  2.87,  3.30,  3.78,  4.30,  4.88,  5.60,  6.50,  7.40,  8.50,  9.50, 10.50, 11.70, 12.85, 14.00,   15.20,  np.nan],
        # T/R=0.80
        [ 1.82,  2.00,  2.32,  2.70,  3.08,  3.50,  3.92,  4.50,  5.10,  5.75,  6.60,  7.50,  8.60,  9.50, 10.60, 11.75, 12.80, 14.00,   15.10,  np.nan],
        # T/R=0.90
        [ 1.96,  2.20,  2.54,  2.90,  3.30,  3.70,  4.20,  4.75,  5.30,  6.00,  6.75,  7.70,  8.70,  9.65, 10.75, 11.80, 12.90, 14.00,   15.00,  16.00],
        # T/R=1.00
        [ 2.11,  2.40,  2.75,  3.10,  3.50,  3.91,  4.40,  5.00,  5.60,  6.20,  7.00,  7.90,  8.80,  9.75, 10.80, 11.90, 12.95, 14.00,   15.00,  16.00],
        # T/R=1.10
        [ 2.30,  2.55,  2.92,  3.30,  3.72,  4.20,  4.70,  5.25,  5.80,  6.55,  7.20,  8.15,  9.00,  9.90, 10.95, 12.00, 13.00, 14.00,   15.00,  16.00],
        # T/R=1.20
        [ 2.40,  2.75,  3.12,  3.50,  3.90,  4.35,  4.85,  5.50,  6.05,  6.70,  7.50,  8.30,  9.20, 10.00, 11.02, 12.10, 13.10, 14.00,   15.00,  16.00],
        # T/R=1.30
        [ 2.55,  2.90,  3.30,  3.70,  4.20,  4.65,  5.20,  5.70,  6.30,  7.00,  7.70,  8.55,  9.35, 10.20, 11.20, 12.25, 13.20, 14.00,   15.00,  16.00],
        # T/R=1.40
        [ 2.70,  3.10,  3.50,  3.90,  4.35,  4.85,  5.35,  5.85,  6.50,  7.25,  8.00,  8.70,  9.50, 10.40, 11.40, 12.40, 13.25, 14.05,   15.00,  16.00],
        # T/R=1.50
        [ 2.85,  3.15,  3.65,  4.10,  4.55,  5.00,  5.50,  6.20,  6.80,  7.50,  8.20,  8.85,  9.70, 10.55, 11.50, 12.50, 13.30, 14.05,   15.02,  16.00],
        # T/R=1.60
        [ 2.95,  3.40,  3.80,  4.25,  4.75,  5.20,  5.75,  6.30,  7.00,  7.65,  8.40,  9.00,  9.80, 10.80, 11.75, 12.60, 13.40, 14.20,   15.10,  16.00],
        # T/R=1.70
        [ 3.10,  3.55,  4.00,  4.50,  4.90,  5.40,  5.95,  6.50,  7.20,  7.80,  8.50,  9.20, 10.00, 10.85, 11.85, 12.75, 13.45, 14.20,   15.10,  16.00],
        # T/R=1.80
        [ 3.25,  3.70,  4.20,  4.65,  5.10,  5.60,  6.10,  6.75,  7.40,  8.00,  8.65,  9.35, 10.10, 11.00, 11.90, 12.80, 13.50, 14.20,   15.10,  16.00],
        # T/R=1.90
        [ 3.43,  3.86,  4.30,  4.75,  5.20,  5.70,  6.30,  6.90,  7.50,  8.17,  8.80,  9.50, 10.20, 11.00, 12.00, 12.82, 13.55, 14.20,   15.10,  16.00],
        # T/R=2.00
        [ 3.50,  4.00,  4.50,  4.90,  5.35,  5.80,  6.40,  7.10,  7.70,  8.30,  8.90,  9.60, 10.40, 11.10, 12.00, 12.85, 13.60, 14.30,   15.10,  16.00],
        # T/R=2.10
        [ 3.60,  4.17,  4.65,  5.05,  5.50,  6.00,  6.60,  7.20,  7.82,  8.45,  9.00,  9.75, 10.50, 11.20, 12.10, 12.90, 13.70, 14.35,   15.10,  16.00],
        # T/R=2.20
        [ 3.75,  4.25,  4.72,  5.20,  5.60,  6.10,  6.70,  7.35,  8.00,  8.55,  9.15,  9.85, 10.60, 11.30, 12.10, 12.90, 13.70, 14.40,   15.15,  16.00],
        # T/R=2.30
        [ 3.85,  4.35,  4.80,  5.25,  5.70,  6.22,  6.80,  7.40,  8.10,  8.65,  9.30,  9.90, 10.70, 11.40, 12.20, 12.95, 13.70, 14.40,   15.20,  16.00],
        # T/R=2.40
        [ 3.95,  4.40,  4.90,  5.35,  5.80,  6.30,  6.90,  7.50,  8.20,  8.80,  9.40, 10.00, 10.80, 11.50, 12.25, 13.00, 13.75, 14.45,   15.20,  16.00],
        # T/R=2.50
        [ 4.00,  4.50,  4.96,  5.40,  5.85,  6.40,  6.95,  7.55,  8.25,  8.85,  9.50, 10.05, 10.85, 11.55, 12.30, 13.00, 13.80, 14.50,   15.25,  16.00],
        # T/R=2.60
        [ 4.07,  4.55,  5.05,  5.47,  5.95,  6.46,  7.00,  7.65,  8.35,  8.95,  9.55, 10.10, 10.90, 11.60, 12.32, 13.00, 13.80, 14.50,   15.25,  16.00],
        # T/R=2.70  (note: H/R=0.60 = 7.50 as published in CIE 117:1995)
        [ 4.10,  4.60,  5.10,  5.53,  6.00,  6.50,  7.50,  7.70,  8.40,  9.00,  9.60, 10.16, 10.92, 11.63, 12.35, 13.00, 13.80, 14.50,   15.25,  16.00],
        # T/R=2.80
        [ 4.15,  4.62,  5.15,  5.56,  6.05,  6.55,  7.08,  7.73,  8.45,  9.05,  9.65, 10.20, 10.95, 11.65, 12.35, 13.00, 13.80, 14.50,   15.25,  16.00],
        # T/R=2.90
        [ 4.20,  4.65,  5.17,  5.60,  6.07,  6.57,  7.12,  7.75,  8.50,  9.10,  9.70, 10.23, 10.95, 11.65, 12.35, 13.00, 13.80, 14.50,   15.25,  16.00],
        # T/R=3.00
        [ 4.22,  4.67,  5.20,  5.65,  6.12,  6.60,  7.15,  7.80,  8.55,  9.12,  9.70, 10.23, 10.95, 11.65, 12.35, 13.00, 13.80, 14.50,   15.25,  16.00],
    ], dtype=np.float64)

    _interpolator: "RegularGridInterpolator | None" = None

    @classmethod
    def _build_interpolator(cls) -> RegularGridInterpolator:
        return RegularGridInterpolator(
            (cls._TR_AXIS, cls._HR_AXIS),
            cls._TABLE,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

    @classmethod
    def p(cls, h_r: float, t_r: float) -> float:
        """
        Guth position index for a luminaire at (|H/R|, |T/R|).

        Bilinear interpolation on CIE 117:1995 Table 4.1.

        Parameters
        ----------
        h_r : float
            H/R — ratio of luminaire height H to horizontal viewing distance R.
        t_r : float
            T/R — ratio of transverse distance T to horizontal viewing distance R.
            Absolute value is used (table is symmetric in T).

        Returns
        -------
        float
            Position index p ≥ 1.0.
            np.nan if (H/R, T/R) is outside the table range or adjacent to a
            missing cell — the luminaire must be excluded from the UGR sum.
        """
        if cls._interpolator is None:
            cls._interpolator = cls._build_interpolator()

        h_r = abs(float(h_r))
        t_r = abs(float(t_r))

        result = cls._interpolator([[t_r, h_r]])
        return float(result[0])
