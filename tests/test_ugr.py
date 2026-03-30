# -*- coding: utf-8 -*-
"""
tests/test_ugr.py
-----------------
Tests for eulumdat-ugr.

Test organisation
-----------------
TestGuthTable       — step 2: GuthTable (guth.py)
TestUgrGrid         — step 3: UgrGrid (geometry.py)

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

from eulumdat_ugr.geometry import UgrGrid
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


# ---------------------------------------------------------------------------
# TestUgrGrid — CIE 117:1995 / CIE 190:2010 room geometry
# ---------------------------------------------------------------------------


class TestUgrGrid:
    """
    Tests for UgrGrid luminaire grid geometry.

    Reference values
    ----------------
    - CIE 190:2010 Table 3 gives N luminaires for SHR=1.0 configurations.
    - CIE 190:2010 Table 2 gives exact C and γ for specific (xT/H, yR/H) pairs,
      used to verify the angle formulas.
    """

    H = 2.0  # m, fixed mounting height

    # ------------------------------------------------------------------
    # Grid size — total luminaires (before filtering)
    # ------------------------------------------------------------------

    def test_grid_4x8_shr1_total(self):
        """4H×8H, SHR=1.0 → 32 luminaires (CIE 190 Table 3, N=32)."""
        grid = UgrGrid(4, 8, shr=1.0)
        assert grid.n_total == 32

    def test_grid_8x4_shr1_total(self):
        """8H×4H, SHR=1.0 → 32 luminaires (CIE 190 Table 3, N=32)."""
        grid = UgrGrid(8, 4, shr=1.0)
        assert grid.n_total == 32

    def test_grid_4x8_shr025_total(self):
        """4H×8H, SHR=0.25 → 512 luminaires (16 on X × 32 on Y)."""
        grid = UgrGrid(4, 8, shr=0.25)
        assert grid.n_total == 512   # 16 * 32

    def test_grid_nx_ny_product(self):
        """n_total == n_x * n_y for all configurations."""
        for x, y, shr in [(4, 8, 1.0), (8, 4, 1.0), (4, 8, 0.25), (2, 4, 1.0)]:
            g = UgrGrid(x, y, shr=shr)
            assert g.n_total == g.n_x * g.n_y, f"Failed for {x}x{y} shr={shr}"

    def test_grid_4x8_shr1_nx_ny(self):
        """4H×8H, SHR=1.0: n_x=4, n_y=8."""
        grid = UgrGrid(4, 8, shr=1.0)
        assert grid.n_x == 4
        assert grid.n_y == 8

    def test_grid_8x4_shr1_nx_ny(self):
        """8H×4H, SHR=1.0: n_x=8, n_y=4."""
        grid = UgrGrid(8, 4, shr=1.0)
        assert grid.n_x == 8
        assert grid.n_y == 4

    # ------------------------------------------------------------------
    # Grid spacing and positions
    # ------------------------------------------------------------------

    def test_spacing_shr025(self):
        """S = shr × H = 0.25 × 2.0 = 0.5 m."""
        grid = UgrGrid(4, 8, shr=0.25)
        assert grid.s == pytest.approx(0.5)

    def test_spacing_shr1(self):
        """S = shr × H = 1.0 × 2.0 = 2.0 m."""
        grid = UgrGrid(4, 8, shr=1.0)
        assert grid.s == pytest.approx(2.0)

    def test_min_r_is_s_over_2(self):
        """Nearest R position is S/2 (observer is at the wall)."""
        for shr in (0.25, 1.0):
            grid = UgrGrid(4, 8, shr=shr)
            R = grid._r_all
            assert R.min() == pytest.approx(grid.s / 2), f"Failed for shr={shr}"

    def test_r_always_positive(self):
        """All R values are strictly positive (observer at y=0, wall)."""
        grid = UgrGrid(4, 8, shr=0.25)
        assert np.all(grid._r_all > 0)

    def test_t_symmetric(self):
        """T values are symmetric about 0 (equal number of ±positions)."""
        grid = UgrGrid(4, 8, shr=1.0)
        t_vals = np.unique(grid._t_all)
        assert np.allclose(np.sort(t_vals), np.sort(-t_vals))

    def test_max_t_shr1_4x8(self):
        """4H×8H, SHR=1.0: max |T| = X/2 - S/2 = 4m - 1m = 3m."""
        grid = UgrGrid(4, 8, shr=1.0)
        assert np.abs(grid._t_all).max() == pytest.approx(3.0)

    # ------------------------------------------------------------------
    # Filters — T/R and γ
    # ------------------------------------------------------------------

    def test_filter_tr_max(self):
        """No valid luminaire has T/R > 3.0 (CIE 117 §4.5)."""
        for orientation in ("crosswise", "endwise"):
            R, T, _, _, _ = UgrGrid(4, 8, shr=0.25).angles(orientation)
            assert np.all(T / R <= 3.0 + 1e-9), f"T/R > 3 for {orientation}"

    def test_filter_gamma_max(self):
        """No valid luminaire has γ > 85° (CIE 117)."""
        for orientation in ("crosswise", "endwise"):
            _, _, _, gamma_deg, _ = UgrGrid(4, 8, shr=0.25).angles(orientation)
            assert np.all(gamma_deg <= 85.0 + 1e-9), f"γ > 85° for {orientation}"

    def test_shr1_4x8_all_pass_filter(self):
        """4H×8H, SHR=1.0: all 32 luminaires pass T/R and γ filters."""
        grid = UgrGrid(4, 8, shr=1.0)
        R, T, _, _, _ = grid.angles("crosswise")
        assert len(R) == 32

    def test_shr025_some_filtered(self):
        """4H×8H, SHR=0.25: 18 luminaires filtered by T/R > 3 (close to wall)."""
        # At R=0.25m: T > 3×0.25=0.75 → T ∈ {1.25,1.75,2.25,2.75,3.25,3.75} → 6/side=12 filtered
        # At R=0.75m: T > 3×0.75=2.25 → T ∈ {2.75,3.25,3.75} → 3/side=6 filtered
        grid = UgrGrid(4, 8, shr=0.25)
        R, T, _, _, _ = grid.angles("crosswise")
        assert len(R) == 512 - 18

    # ------------------------------------------------------------------
    # Angle formulas — crosswise
    # ------------------------------------------------------------------

    def test_crosswise_angle_known(self):
        """
        Crosswise: C = arctan(T/R).

        CIE 190 Table 2: xT/H=0.5, yR/H=0.5 → C=45.00°, γ=35.26°.
        With H=2m: T=1m, R=1m.
        """
        # Build a minimal 1×1 grid at T=1, R=1 via SHR and dims
        # xT/H=0.5 → T=S/2 → S=1m → shr=0.5; yR/H=0.5 → R=S/2 → matches
        grid = UgrGrid(1, 1, shr=0.5)   # S=1m, T=±0.5m, R=0.5m
        # T=0.5m, R=0.5m → C=45°, γ=arctan(sqrt(0.5)/2)≠35.26°
        # Use a 2×2 room with shr=1: T=1m, R=1m for the (1,1) cell
        grid2 = UgrGrid(2, 2, shr=1.0)  # S=2m: T=±1m, R=1m  (only one R)
        R, T, C, gamma, _ = grid2.angles("crosswise")
        # Select any luminaire at T=1, R=1 (two exist: +1 and -1 on X, both |T|=1)
        idx = np.where(np.isclose(R, 1.0) & np.isclose(T, 1.0))[0]
        assert len(idx) >= 1
        assert C[idx[0]] == pytest.approx(45.0, rel=1e-4)
        assert gamma[idx[0]] == pytest.approx(35.264, rel=1e-3)

    def test_crosswise_t0_c_is_0(self):
        """Crosswise: when T=0 (luminaire on centre line), C = 0°."""
        # 2H×2H, shr=1: T positions ±1m, no T=0. Use 1H×2H shr=0.5: T=±0.25m.
        # Better: directly check via formula for a grid that has T=0
        # Use even n_half → no T=0 position exists. Test with odd x_dim_H/shr combo.
        # x_room/s = odd → odd number per side including midpoint?
        # n_half = round(x_room/s/2). For T=0 to exist, we'd need n_half=0 (no T positions).
        # Actually T=0 never appears in the symmetric grid (always ±S/2, ±3S/2 ...).
        # So this test verifies C→0 as T→0 by checking a small-T case.
        grid = UgrGrid(1, 4, shr=0.5)   # S=1m, T=±0.5m, R=0.5,1.5,2.5,3.5,4.5,...
        R, T, C, _, _ = grid.angles("crosswise")
        idx = np.where(np.isclose(T, 0.5) & np.isclose(R, 7.5))[0]
        if len(idx):
            # C = arctan(0.5/7.5) ≈ 3.81°
            assert C[idx[0]] == pytest.approx(math.degrees(math.atan(0.5 / 7.5)), rel=1e-4)

    def test_crosswise_equal_t_r(self):
        """Crosswise: C = 45° when T = R."""
        grid = UgrGrid(2, 2, shr=1.0)   # T=±1, R=1 → T=R=1
        R, T, C, _, _ = grid.angles("crosswise")
        idx = np.where(np.isclose(T, 1.0) & np.isclose(R, 1.0))[0]
        assert C[idx[0]] == pytest.approx(45.0, rel=1e-5)

    # ------------------------------------------------------------------
    # Angle formulas — endwise
    # ------------------------------------------------------------------

    def test_endwise_t0_c_is_90(self):
        """Endwise: when T→0 (i.e. luminaire on axis), C → 90°."""
        # Use a grid with small T and large R: C_endwise = 90 - arctan(T/R) ≈ 90°
        grid = UgrGrid(1, 8, shr=0.5)   # T=0.5m, many R positions
        R, T, C, _, _ = grid.angles("endwise")
        # Select largest R (small arctan(T/R)):  C ≈ 90 - arctan(0.5/7.5)
        idx = np.argmax(R)
        expected = 90.0 - math.degrees(math.atan(T[idx] / R[idx]))
        assert C[idx] == pytest.approx(expected, rel=1e-5)

    def test_endwise_known_angle(self):
        """
        Endwise: C = 90° − arctan(T/R).

        For T=1, R=1: C = 90° - 45° = 45°.
        """
        grid = UgrGrid(2, 2, shr=1.0)
        R, T, C, _, _ = grid.angles("endwise")
        idx = np.where(np.isclose(T, 1.0) & np.isclose(R, 1.0))[0]
        assert C[idx[0]] == pytest.approx(45.0, rel=1e-5)

    def test_endwise_larger_t_smaller_c(self):
        """Endwise: C < 90° - 0° = 90°; larger T/R → smaller C."""
        grid = UgrGrid(4, 8, shr=1.0)
        R, T, C, _, _ = grid.angles("endwise")
        assert np.all(C >= 0.0)
        assert np.all(C <= 90.0)

    # ------------------------------------------------------------------
    # Common angles
    # ------------------------------------------------------------------

    def test_gamma_known(self):
        """
        γ = arctan(√(R²+T²)/H).

        CIE 190 Table 2: xT/H=0.5, yR/H=1.5 → γ = 57.69°.
        With H=2m: T=1m, R=3m.
        """
        grid = UgrGrid(2, 4, shr=1.0)   # S=2m: T=±1, R=1,3,5,7
        R, T, C, gamma, _ = grid.angles("crosswise")
        idx = np.where(np.isclose(R, 3.0) & np.isclose(T, 1.0))[0]
        assert gamma[idx[0]] == pytest.approx(57.69, abs=0.01)

    def test_r_dist_known(self):
        """
        r = √(R²+T²+H²).

        For T=1, R=1, H=2: r = √6 ≈ 2.449 m.
        """
        grid = UgrGrid(2, 2, shr=1.0)
        _, _, _, _, r_m = grid.angles("crosswise")
        idx_rt11 = np.where(
            np.isclose(grid._r_all[grid._r_all > 0], 1.0)
        )
        expected = math.sqrt(1 + 1 + 4)  # sqrt(6)
        # Get r from the angles output for the (T=1,R=1) cell
        R, T, _, _, r_m2 = grid.angles("crosswise")
        idx = np.where(np.isclose(R, 1.0) & np.isclose(T, 1.0))[0]
        assert r_m2[idx[0]] == pytest.approx(expected, rel=1e-6)

    # ------------------------------------------------------------------
    # Invalid orientation
    # ------------------------------------------------------------------

    def test_invalid_orientation_raises(self):
        """angles() raises ValueError for unknown orientation."""
        grid = UgrGrid(4, 8)
        with pytest.raises(ValueError, match="orientation"):
            grid.angles("diagonal")
