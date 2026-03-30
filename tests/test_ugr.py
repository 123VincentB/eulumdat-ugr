# -*- coding: utf-8 -*-
"""
tests/test_ugr.py
-----------------
Tests for eulumdat-ugr.

Test organisation
-----------------
TestGuthTable       — step 2: GuthTable (guth.py)

Running
-------
    pytest                    # all tests
    pytest -v                 # verbose
    pytest tests/test_ugr.py  # this file only
"""

import math
from pathlib import Path

import numpy as np
import pytest

from eulumdat_ugr.guth import GuthTable

DATA_DIR = Path(__file__).parent.parent / "data" / "input"


# ---------------------------------------------------------------------------
# TestGuthTable — CIE 117:1995 Table 4.1
# ---------------------------------------------------------------------------


class TestGuthTable:
    """
    Tests for GuthTable.p(h_r, t_r).

    Spot-checks use values read directly from CIE 117:1995 Table 4.1.
    Interpolation tests use midpoints whose expected values are
    computable by hand from the four surrounding grid cells.
    """

    # ------------------------------------------------------------------
    # Exact grid-point spot-checks
    # ------------------------------------------------------------------

    def test_origin(self):
        """p(H/R=0, T/R=0) = 1.00 — minimum index (luminaire directly ahead)."""
        assert GuthTable.p(0.0, 0.0) == pytest.approx(1.00)

    def test_hr0_tr050(self):
        """p(H/R=0.5, T/R=0) = 2.86."""
        assert GuthTable.p(0.50, 0.0) == pytest.approx(2.86)

    def test_hr050_tr050(self):
        """p(H/R=0.5, T/R=0.5) = 2.91."""
        assert GuthTable.p(0.50, 0.50) == pytest.approx(2.91)

    def test_hr100_tr100(self):
        """p(H/R=1.0, T/R=1.0) = 7.00."""
        assert GuthTable.p(1.00, 1.00) == pytest.approx(7.00)

    def test_hr190_tr300(self):
        """p(H/R=1.9, T/R=3.0) = 16.00 — corner of table."""
        assert GuthTable.p(1.90, 3.00) == pytest.approx(16.00)

    def test_hr000_tr100(self):
        """p(H/R=0, T/R=1.0) = 2.11."""
        assert GuthTable.p(0.00, 1.00) == pytest.approx(2.11)

    def test_hr030_tr130(self):
        """p(H/R=0.3, T/R=1.3) = 3.70."""
        assert GuthTable.p(0.30, 1.30) == pytest.approx(3.70)

    def test_hr150_tr200(self):
        """p(H/R=1.5, T/R=2.0) = 12.85."""
        assert GuthTable.p(1.50, 2.00) == pytest.approx(12.85)

    def test_hr060_tr270(self):
        """p(H/R=0.6, T/R=2.7) = 7.50 — published outlier, kept as-is."""
        assert GuthTable.p(0.60, 2.70) == pytest.approx(7.50)

    # ------------------------------------------------------------------
    # Bilinear interpolation
    # ------------------------------------------------------------------

    def test_interpolation_midpoint(self):
        """
        At the midpoint of a 2×2 cell the result equals the arithmetic mean
        of the four corners (bilinear interpolation property).

        Cell: H/R ∈ [0.5, 0.6], T/R ∈ [0.5, 0.6]
        Corners: (0.5,0.5)=2.91, (0.5,0.6)=3.10, (0.6,0.5)=3.40, (0.6,0.6)=3.60
        Expected midpoint: (2.91 + 3.10 + 3.40 + 3.60) / 4 = 3.2525
        """
        expected = (2.91 + 3.10 + 3.40 + 3.60) / 4.0
        assert GuthTable.p(0.55, 0.55) == pytest.approx(expected, rel=1e-5)

    def test_interpolation_along_hr_axis(self):
        """
        At T/R=0.0, H/R=0.05 — midpoint between H/R=0 (1.00) and H/R=0.1 (1.26).
        Expected: (1.00 + 1.26) / 2 = 1.13
        """
        assert GuthTable.p(0.05, 0.0) == pytest.approx(1.13, rel=1e-4)

    def test_interpolation_along_tr_axis(self):
        """
        At H/R=0.0, T/R=0.05 — midpoint between T/R=0 (1.00) and T/R=0.1 (1.05).
        Expected: (1.00 + 1.05) / 2 = 1.025
        """
        assert GuthTable.p(0.0, 0.05) == pytest.approx(1.025, rel=1e-4)

    def test_interpolated_value_between_bounds(self):
        """Interpolated value must lie between the four surrounding grid values."""
        h_r, t_r = 0.35, 1.15
        result = GuthTable.p(h_r, t_r)
        lo = min(
            GuthTable.p(0.30, 1.10), GuthTable.p(0.30, 1.20),
            GuthTable.p(0.40, 1.10), GuthTable.p(0.40, 1.20),
        )
        hi = max(
            GuthTable.p(0.30, 1.10), GuthTable.p(0.30, 1.20),
            GuthTable.p(0.40, 1.10), GuthTable.p(0.40, 1.20),
        )
        assert lo <= result <= hi

    # ------------------------------------------------------------------
    # Symmetry in T/R
    # ------------------------------------------------------------------

    def test_symmetry_negative_tr(self):
        """p(h_r, -t_r) == p(h_r, t_r) — absolute value applied."""
        assert GuthTable.p(0.50, -1.00) == pytest.approx(GuthTable.p(0.50, 1.00))

    def test_symmetry_negative_hr(self):
        """p(-h_r, t_r) == p(h_r, t_r) — absolute value applied."""
        assert GuthTable.p(-0.80, 1.50) == pytest.approx(GuthTable.p(0.80, 1.50))

    # ------------------------------------------------------------------
    # Out-of-bounds and NaN cells → must return NaN
    # ------------------------------------------------------------------

    def test_nan_cell_missing_in_table(self):
        """
        p(H/R=1.9, T/R=0.0) — the table has '-' at this cell → NaN.
        (The four surrounding cells include NaN values.)
        """
        result = GuthTable.p(1.90, 0.0)
        assert math.isnan(result)

    def test_nan_hr_exceeds_table(self):
        """H/R > 1.90 is outside the table → NaN."""
        assert math.isnan(GuthTable.p(2.00, 1.00))

    def test_nan_tr_exceeds_table(self):
        """T/R > 3.00 is outside the table → NaN."""
        assert math.isnan(GuthTable.p(1.00, 3.10))

    def test_nan_both_exceed_table(self):
        """Both H/R and T/R outside range → NaN."""
        assert math.isnan(GuthTable.p(5.00, 5.00))

    def test_nan_adjacent_to_missing_cell(self):
        """
        Interpolation touching a NaN boundary cell returns NaN.
        At T/R=0.05, H/R=1.75: one of the four surrounding cells is NaN
        (T/R=0.00, H/R=1.80).
        """
        result = GuthTable.p(1.75, 0.05)
        assert math.isnan(result)

    # ------------------------------------------------------------------
    # Table shape / metadata
    # ------------------------------------------------------------------

    def test_table_shape(self):
        """Table must be (31 T/R rows) × (20 H/R columns)."""
        assert GuthTable._TABLE.shape == (31, 20)

    def test_tr_axis_length(self):
        assert len(GuthTable._TR_AXIS) == 31

    def test_hr_axis_length(self):
        assert len(GuthTable._HR_AXIS) == 20

    def test_all_valid_cells_positive(self):
        """Every non-NaN cell must be ≥ 1.0."""
        valid = GuthTable._TABLE[~np.isnan(GuthTable._TABLE)]
        assert np.all(valid >= 1.0)

    def test_interpolator_cached(self):
        """The interpolator is built once and reused."""
        GuthTable._interpolator = None  # reset
        GuthTable.p(1.0, 1.0)
        first = GuthTable._interpolator
        assert first is not None
        GuthTable.p(0.5, 0.5)
        assert GuthTable._interpolator is first
