# -*- coding: utf-8 -*-
"""
background.py
-------------
BackgroundLuminance — calculates Lb (background luminance) for UGR.

Algorithm
---------
CIE 190:2010 §4.2 and equations (7)–(12).

Overview
--------
1. Interpolate I(C, γ) at 10° zone midpoints (5°, 15°, …, 175°) from LDT data.
2. Zonal fluxes G(γ_mid) = I_avg(γ_mid) × ZF(γ_mid) [lm per 1000 lm nominal].
3. Normalize by LORL: G_H = G × (LORL/100) / (Σ G / 1000).
4. R_DLO, R_ULO — flux output ratios from G_H.
5. Cumulative flux sums Φ_zL1…4 from G_H_down → F_DF, F_DW, F_DC.
6. F_UWID per reflectance combination (eq. 8b).
7. B = Φ_real × N / A_w  (corrected mode, eq. 7).
8. Lb = B · F_UWID / π.

Zone factor formula (10° midpoint approximation, CIE 190 §4.2)
--------------------------------------------------------------
ZF(γ_mid) = 2π · sin(γ_mid_rad) · Δγ_rad    where Δγ = 10°

Midpoints: 5°, 15°, 25°, …, 175° (18 zones covering the full sphere).
Data at 0°, 10°, 20°, … is informational only and not used in the flux sum.

Cumulative flux helper
----------------------
Downward midpoints: idx 0=5°, 1=15°, 2=25°, 3=35°, 4=45°, 5=55°, 6=65°, 7=75°, 8=85°

Φ_zL1 = Σ G_H(γ ≤ 40°) + 0.130 × G_H(45°)   = G_H[:4].sum() + 0.130 × G_H[4]
Φ_zL2 = Σ G_H(γ ≤ 60°)                         = G_H[:6].sum()
Φ_zL3 = Σ G_H(γ ≤ 70°) + 0.547 × G_H(75°)   = G_H[:7].sum() + 0.547 × G_H[7]
Φ_zL4 = Σ G_H(γ ≤ 90°)                         = G_H.sum()

Scope (v1.0.0)
--------------
Hardcoded for k = 2.67 (4H×8H and 8H×4H rooms — same proportions, k identical).
Covers all 5 CIE 190 reflectance combinations from Table 5.
"""

import numpy as np

# ------------------------------------------------------------------
# 10° zone midpoints (CIE 190:2010 §4.2)
# ------------------------------------------------------------------
_DELTA_DEG: float = 10.0
_MID_ALL: np.ndarray = np.arange(5.0, 176.0, 10.0)   # 5, 15, …, 175 — 18 zones
_N_DOWN: int = 9   # first 9 midpoints cover the downward hemisphere: 5°…85°

# ------------------------------------------------------------------
# Table 5 (CIE 190:2010) — F_T transfer factors for k = 2.67
# ------------------------------------------------------------------
# Format: {reflectance: (F_T,FW, F_T,WW-1, F_T,CW)}
_FT_K267: dict[str, tuple[float, float, float]] = {
    "70/50/20": (0.115, 0.211, 0.307),
    "70/30/20": (0.106, 0.117, 0.283),
    "50/50/20": (0.100, 0.187, 0.211),
    "50/30/20": (0.093, 0.104, 0.196),
    "30/30/20": (0.081, 0.092, 0.114),
}

# Hardcoded F_GL zone-flux factors for k = 2.67 (CIE 190:2010 Table 6 worksheet).
# These weight Φ_zL1…4 to compute the direct-to-floor flux fraction F_DF.
_FGL_K267: tuple[float, float, float, float] = (0.280, 0.165, 0.499, 0.006)


class BackgroundLuminance:
    """
    Background luminance Lb for UGR (CIE 190:2010).

    Usage
    -----
    ::

        result = BackgroundLuminance.compute(ldt, n_luminaires=32, a_w=96.0)
        lb_70_50_20 = result["70/50/20"]   # cd/m²

    Parameters are room-dependent (N and A_w depend on room dimensions and SHR).
    The module is hardcoded for k = 2.67 (v1.0.0).
    """

    @classmethod
    def compute(
        cls,
        ldt,
        n_luminaires: int,
        a_w: float,
    ) -> dict[str, float]:
        """
        Lb (cd/m²) for each reflectance combination at the luminaire's real flux.

        Parameters
        ----------
        ldt : pyldt.model.Ldt
            EULUMDAT file — intensities in cd/klm, first lamp set used.
        n_luminaires : int
            Number of luminaires in the room (N).  Depends on SHR.
        a_w : float
            Total wall area between observer eye level and luminaire plane [m²].
            A_w = 2 · H · (X + Y).

        Returns
        -------
        dict[str, float]
            Mapping ``"C/W/R"`` reflectance string → Lb [cd/m²].
        """
        h = ldt.header

        # --- Build intensity matrix (n_gamma, n_C), values in cd/klm ---
        I_matrix = np.array(ldt.intensities).T   # (n_C, n_gamma) → (n_gamma, n_C)
        gamma = np.asarray(h.g_angles, dtype=np.float64)

        # --- Zonal fluxes at 10° midpoints (unnormalized, per 1000 lm) ---
        G = cls._zonal_fluxes(I_matrix, gamma)   # shape (18,)

        # --- Normalize by LORL ---
        lorl = float(h.lorl) / 100.0
        scale = lorl / (G.sum() / 1000.0)
        G_H = G * scale   # actual flux distribution per 1000 lm reference

        # --- Flux output ratios ---
        G_H_down = G_H[:_N_DOWN]   # midpoints 5°…85°
        G_H_up   = G_H[_N_DOWN:]   # midpoints 95°…175°
        R_DLO = float(G_H_down.sum() / 1000.0)
        R_ULO = float(G_H_up.sum()   / 1000.0)

        # --- Cumulative flux sums (downward hemisphere) ---
        Phi1, Phi2, Phi3, Phi4 = cls._phi_zl(G_H_down)

        # --- F_DF, F_DW, F_DC (CIE 190 eq. 10–12) ---
        fgl1, fgl2, fgl3, fgl4 = _FGL_K267
        Phi_zL = Phi1 * fgl1 + Phi2 * fgl2 + Phi3 * fgl3 + Phi4 * fgl4
        F_DF = Phi_zL / 1000.0
        F_DW = R_DLO - F_DF
        F_DC = R_ULO

        # --- Real luminaire flux (first lamp set, eq. 7) ---
        phi_real = float(h.num_lamps[0]) * float(h.lamp_flux[0])
        B = phi_real * n_luminaires / a_w

        # --- Lb per reflectance (eq. 8b → eq. 7) ---
        result: dict[str, float] = {}
        for refl, (ft_fw, ft_ww1, ft_cw) in _FT_K267.items():
            F_UWID = F_DF * ft_fw + F_DW * ft_ww1 + F_DC * ft_cw
            E_WID = B * F_UWID
            result[refl] = E_WID / np.pi

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _zonal_fluxes(
        I_matrix: np.ndarray,
        gamma: np.ndarray,
    ) -> np.ndarray:
        """
        Zonal fluxes G(γ_mid) [lm per 1000 lm] at the 18 fixed 10° midpoints.

        G(γ_mid) = I_avg(γ_mid) × ZF(γ_mid)

        ZF(γ_mid) = 2π · sin(γ_mid_rad) · Δγ_rad    (Δγ = 10°)

        I at each midpoint is obtained by linear interpolation along the γ axis;
        then averaged over all C planes.

        Parameters
        ----------
        I_matrix : np.ndarray, shape (n_gamma, n_C)
            Intensities in cd/klm.
        gamma : np.ndarray, shape (n_gamma,)
            Gamma angles [degrees] as stored in the LDT file.

        Returns
        -------
        np.ndarray, shape (18,)
            G values at midpoints 5°, 15°, …, 175°.
        """
        n_C = I_matrix.shape[1]
        # Interpolate each C plane at the 18 midpoints, then average
        I_mid = np.column_stack(
            [np.interp(_MID_ALL, gamma, I_matrix[:, c]) for c in range(n_C)]
        )  # shape (18, n_C)
        I_avg_mid = I_mid.mean(axis=1)   # shape (18,)

        ZF = 2.0 * np.pi * np.sin(np.radians(_MID_ALL)) * np.radians(_DELTA_DEG)
        return I_avg_mid * ZF

    @staticmethod
    def _phi_zl(
        G_H_down: np.ndarray,
    ) -> tuple[float, float, float, float]:
        """
        Cumulative flux sums Φ_zL1…4 [lm per 1000 lm].

        G_H_down must contain exactly 9 values corresponding to the downward
        midpoints 5°, 15°, 25°, 35°, 45°, 55°, 65°, 75°, 85° (in that order).

        Φ_zL1 = G_H[:4].sum() + 0.130 × G_H[4]
        Φ_zL2 = G_H[:6].sum()
        Φ_zL3 = G_H[:7].sum() + 0.547 × G_H[7]
        Φ_zL4 = G_H.sum()

        Parameters
        ----------
        G_H_down : np.ndarray, shape (9,)
            Normalized zonal fluxes for the downward hemisphere (LORL-scaled).
        """
        Phi1 = float(G_H_down[:4].sum()) + 0.130 * float(G_H_down[4])
        Phi2 = float(G_H_down[:6].sum())
        Phi3 = float(G_H_down[:7].sum()) + 0.547 * float(G_H_down[7])
        Phi4 = float(G_H_down.sum())
        return Phi1, Phi2, Phi3, Phi4
