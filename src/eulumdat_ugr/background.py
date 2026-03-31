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

Room configurations
-------------------
19 standard rooms indexed 0–18:
(2,2),(2,3),(2,4),(2,6),(2,8),(2,12),
(4,2),(4,3),(4,4),(4,6),(4,8),(4,12),
(8,4),(8,6),(8,8),(8,12),
(12,4),(12,6),(12,8)

Each room has a k value; F_GL and F_T are looked up by room_idx.
"""

import numpy as np

# ------------------------------------------------------------------
# 10° zone midpoints (CIE 190:2010 §4.2)
# ------------------------------------------------------------------
_DELTA_DEG: float = 10.0
_MID_ALL: np.ndarray = np.arange(5.0, 176.0, 10.0)   # 5, 15, …, 175 — 18 zones
_N_DOWN: int = 9   # first 9 midpoints cover the downward hemisphere: 5°…85°

# ------------------------------------------------------------------
# Room configurations — 19 standard rooms (CIE 190:2010)
# ------------------------------------------------------------------
# Room index → (x_dim_H, y_dim_H) as multiples of H
_ROOM_CONFIGS: list[tuple[float, float]] = [
    (2, 2), (2, 3), (2, 4), (2, 6), (2, 8), (2, 12),   # idx 0–5
    (4, 2), (4, 3), (4, 4), (4, 6), (4, 8), (4, 12),   # idx 6–11
    (8, 4), (8, 6), (8, 8), (8, 12),                    # idx 12–15
    (12, 4), (12, 6), (12, 8),                           # idx 16–18
]

# k = (X × Y) / (H × (X + Y)) — computed from x_dim_H, y_dim_H (H cancels)
_ROOM_K: list[float] = [
    1.00, 1.20, 1.33, 1.50, 1.60, 1.71,   # (2,2)…(2,12)
    1.33, 1.71, 2.00, 2.40, 2.67, 3.00,   # (4,2)…(4,12)
    2.67, 3.43, 4.00, 4.80,               # (8,4)…(8,12)
    3.00, 4.00, 4.80,                     # (12,4)…(12,8)
]

# ------------------------------------------------------------------
# F_GL factors (CIE 190:2010 Table 4 worksheet)
# Format: {k: (F_GL1, F_GL2, F_GL3, F_GL4)}
# ------------------------------------------------------------------
_FGL: dict[float, tuple[float, float, float, float]] = {
    1.00: ( 0.690,  0.109,  0.085, -0.016),
    1.20: ( 0.578,  0.200,  0.127, -0.018),
    1.33: ( 0.528,  0.218,  0.170, -0.017),
    1.50: ( 0.485,  0.215,  0.222, -0.012),
    1.60: ( 0.466,  0.207,  0.249, -0.006),
    1.71: ( 0.448,  0.198,  0.272,  0.005),
    2.00: ( 0.338,  0.257,  0.351, -0.018),
    2.40: ( 0.296,  0.203,  0.449, -0.006),
    2.67: ( 0.280,  0.165,  0.499,  0.006),
    3.00: ( 0.264,  0.125,  0.541,  0.027),
    3.43: ( 0.248,  0.058,  0.628,  0.032),
    4.00: ( 0.239, -0.012,  0.690,  0.058),
    4.80: ( 0.232, -0.084,  0.740,  0.098),
}

# Convenience alias for k = 2.67 (backward compatibility)
_FGL_K267: tuple[float, float, float, float] = _FGL[2.67]

# ------------------------------------------------------------------
# F_T factors — all 19 rooms × 5 reflectances (CIE 190:2010 Table 5)
# Format: {refl: [(FT.FW, FT.WW-1, FT.CW) for room_idx in 0..18]}
# ------------------------------------------------------------------
_FT: dict[str, list[tuple[float, float, float]]] = {
    "70/50/20": [
        #  (FW,    WW-1,  CW)                  room (x_dim_H, y_dim_H)
        (0.220, 0.422, 0.646),  # 0  (2, 2)
        (0.199, 0.376, 0.571),  # 1  (2, 3)
        (0.187, 0.351, 0.531),  # 2  (2, 4)
        (0.174, 0.322, 0.488),  # 3  (2, 6)
        (0.167, 0.307, 0.466),  # 4  (2, 8)
        (0.160, 0.290, 0.443),  # 5  (2, 12)
        (0.187, 0.351, 0.531),  # 6  (4, 2)
        (0.158, 0.295, 0.439),  # 7  (4, 3)
        (0.142, 0.265, 0.389),  # 8  (4, 4)
        (0.124, 0.230, 0.335),  # 9  (4, 6)
        (0.115, 0.211, 0.307),  # 10 (4, 8)
        (0.105, 0.190, 0.279),  # 11 (4, 12)
        (0.115, 0.211, 0.307),  # 12 (8, 4)
        (0.094, 0.175, 0.247),  # 13 (8, 6)
        (0.083, 0.155, 0.215),  # 14 (8, 8)
        (0.071, 0.133, 0.183),  # 15 (8, 12)
        (0.105, 0.190, 0.279),  # 16 (12, 4)
        (0.083, 0.153, 0.216),  # 17 (12, 6)
        (0.071, 0.133, 0.183),  # 18 (12, 8)
    ],
    "70/30/20": [
        (0.188, 0.217, 0.553),  # 0  (2, 2)
        (0.173, 0.196, 0.497),  # 1  (2, 3)
        (0.164, 0.184, 0.465),  # 2  (2, 4)
        (0.154, 0.171, 0.432),  # 3  (2, 6)
        (0.149, 0.164, 0.415),  # 4  (2, 8)
        (0.143, 0.156, 0.397),  # 5  (2, 12)
        (0.164, 0.184, 0.465),  # 6  (4, 2)
        (0.142, 0.159, 0.393),  # 7  (4, 3)
        (0.129, 0.144, 0.351),  # 8  (4, 4)
        (0.114, 0.127, 0.307),  # 9  (4, 6)
        (0.106, 0.117, 0.283),  # 10 (4, 8)
        (0.098, 0.106, 0.259),  # 11 (4, 12)
        (0.106, 0.117, 0.283),  # 12 (8, 4)
        (0.088, 0.098, 0.231),  # 13 (8, 6)
        (0.078, 0.088, 0.203),  # 14 (8, 8)
        (0.067, 0.076, 0.174),  # 15 (8, 12)
        (0.098, 0.106, 0.259),  # 16 (12, 4)
        (0.078, 0.086, 0.204),  # 17 (12, 6)
        (0.067, 0.076, 0.174),  # 18 (12, 8)
    ],
    "50/50/20": [
        (0.198, 0.380, 0.445),  # 0  (2, 2)
        (0.178, 0.338, 0.393),  # 1  (2, 3)
        (0.166, 0.314, 0.365),  # 2  (2, 4)
        (0.154, 0.287, 0.335),  # 3  (2, 6)
        (0.147, 0.273, 0.320),  # 4  (2, 8)
        (0.141, 0.257, 0.304),  # 5  (2, 12)
        (0.166, 0.314, 0.364),  # 6  (4, 2)
        (0.140, 0.263, 0.301),  # 7  (4, 3)
        (0.125, 0.235, 0.267),  # 8  (4, 4)
        (0.108, 0.204, 0.230),  # 9  (4, 6)
        (0.100, 0.187, 0.211),  # 10 (4, 8)
        (0.091, 0.167, 0.191),  # 11 (4, 12)
        (0.100, 0.187, 0.211),  # 12 (8, 4)
        (0.081, 0.154, 0.169),  # 13 (8, 6)
        (0.071, 0.137, 0.147),  # 14 (8, 8)
        (0.061, 0.117, 0.125),  # 15 (8, 12)
        (0.091, 0.167, 0.191),  # 16 (12, 4)
        (0.071, 0.134, 0.148),  # 17 (12, 6)
        (0.061, 0.117, 0.125),  # 18 (12, 8)
    ],
    "50/30/20": [
        (0.172, 0.198, 0.386),  # 0  (2, 2)
        (0.157, 0.179, 0.346),  # 1  (2, 3)
        (0.148, 0.167, 0.324),  # 2  (2, 4)
        (0.138, 0.155, 0.301),  # 3  (2, 6)
        (0.133, 0.147, 0.288),  # 4  (2, 8)
        (0.128, 0.140, 0.276),  # 5  (2, 12)
        (0.148, 0.167, 0.324),  # 6  (4, 2)
        (0.126, 0.143, 0.272),  # 7  (4, 3)
        (0.114, 0.129, 0.244),  # 8  (4, 4)
        (0.100, 0.113, 0.212),  # 9  (4, 6)
        (0.093, 0.104, 0.196),  # 10 (4, 8)
        (0.085, 0.094, 0.179),  # 11 (4, 12)
        (0.093, 0.104, 0.196),  # 12 (8, 4)
        (0.076, 0.087, 0.158),  # 13 (8, 6)
        (0.067, 0.078, 0.140),  # 14 (8, 8)
        (0.058, 0.067, 0.120),  # 15 (8, 12)
        (0.085, 0.094, 0.179),  # 16 (12, 4)
        (0.068, 0.077, 0.140),  # 17 (12, 6)
        (0.058, 0.067, 0.120),  # 18 (12, 8)
    ],
    "30/30/20": [
        (0.157, 0.181, 0.227),  # 0  (2, 2)
        (0.141, 0.162, 0.203),  # 1  (2, 3)
        (0.132, 0.151, 0.190),  # 2  (2, 4)
        (0.123, 0.139, 0.176),  # 3  (2, 6)
        (0.118, 0.132, 0.169),  # 4  (2, 8)
        (0.113, 0.124, 0.161),  # 5  (2, 12)
        (0.132, 0.151, 0.190),  # 6  (4, 2)
        (0.112, 0.128, 0.159),  # 7  (4, 3)
        (0.100, 0.115, 0.142),  # 8  (4, 4)
        (0.087, 0.101, 0.124),  # 9  (4, 6)
        (0.081, 0.092, 0.114),  # 10 (4, 8)
        (0.074, 0.083, 0.104),  # 11 (4, 12)
        (0.081, 0.092, 0.114),  # 12 (8, 4)
        (0.066, 0.077, 0.092),  # 13 (8, 6)
        (0.058, 0.069, 0.081),  # 14 (8, 8)
        (0.049, 0.059, 0.069),  # 15 (8, 12)
        (0.074, 0.083, 0.104),  # 16 (12, 4)
        (0.058, 0.067, 0.081),  # 17 (12, 6)
        (0.049, 0.059, 0.069),  # 18 (12, 8)
    ],
}

# Ordered list of reflectance keys (col % 5 → index into this list)
REFLECTANCES: list[str] = ["70/50/20", "70/30/20", "50/50/20", "50/30/20", "30/30/20"]

# Backward-compat alias (used by existing tests indirectly via compute())
_FT_K267: dict[str, tuple[float, float, float]] = {
    refl: _FT[refl][10] for refl in REFLECTANCES
}


class BackgroundLuminance:
    """
    Background luminance Lb for UGR (CIE 190:2010).

    Usage
    -----
    ::

        result = BackgroundLuminance.compute(ldt, n_luminaires=32, a_w=96.0)
        lb_70_50_20 = result["70/50/20"]   # cd/m²

    Parameters are room-dependent (N and A_w depend on room dimensions and SHR).
    ``room_idx`` selects the F_GL and F_T factors from CIE 190 Tables 4 and 5.
    Default room_idx=10 corresponds to the 4H×8H room (k=2.67).
    """

    @classmethod
    def compute(
        cls,
        ldt,
        n_luminaires: int,
        a_w: float,
        room_idx: int = 10,
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
        room_idx : int, optional
            Index into the 19-room table (default 10 = 4H×8H, k=2.67).

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
        fgl1, fgl2, fgl3, fgl4 = _FGL[_ROOM_K[room_idx]]
        Phi_zL = Phi1 * fgl1 + Phi2 * fgl2 + Phi3 * fgl3 + Phi4 * fgl4
        F_DF = Phi_zL / 1000.0
        F_DW = R_DLO - F_DF
        F_DC = R_ULO

        # --- Real luminaire flux (first lamp set, eq. 7) ---
        phi_real = float(h.num_lamps[0]) * float(h.lamp_flux[0])
        B = phi_real * n_luminaires / a_w

        # --- Lb per reflectance (eq. 8b → eq. 7) ---
        result: dict[str, float] = {}
        for refl in REFLECTANCES:
            ft_fw, ft_ww1, ft_cw = _FT[refl][room_idx]
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
