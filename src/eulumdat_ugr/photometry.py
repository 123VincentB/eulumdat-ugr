# -*- coding: utf-8 -*-
"""
photometry.py
-------------
UgrPhotometry — luminance L_i and solid angle ω_i for each luminaire.

For each luminaire that passes the UgrGrid filters (T/R ≤ 3, γ ≤ 85°),
the UGR formula requires:

    L_i  — apparent luminance [cd/m²] in direction (C_i, γ_i)
    ω_i  — solid angle [sr] subtended by the luminous area from the observer

Formulae (CIE 117:1995, eq. 4.3–4.4)
--------------------------------------
    L_i  = LuminanceResult.at(C_i, γ_i)
    A_pi = LuminanceResult.projected_area(C_i, γ_i)   [m²]
    ω_i  = A_pi / r_i²                                 [sr]

where r_i = √(R_i² + T_i² + H²) is the slant distance from the observer
to luminaire i [m].

Usage
-----
::

    from pyldt import LdtReader
    from eulumdat_luminance import LuminanceCalculator
    from eulumdat_ugr.geometry import UgrGrid
    from eulumdat_ugr.photometry import UgrPhotometry

    ldt = LdtReader.read("file.ldt")
    lum = LuminanceCalculator.compute(ldt, full=True)   # full=True is required

    grid = UgrGrid(4, 8)
    R, T, C_deg, gamma_deg, r_m = grid.angles("crosswise")

    L, omega = UgrPhotometry.compute(lum, C_deg, gamma_deg, r_m)

Note
----
``full=True`` is required when calling ``LuminanceCalculator.compute``.  With
``full=False`` the g_axis covers only 65°–85°, which would cause ``at()`` to
raise ``ValueError`` for luminaires with γ < 65°.
"""

import numpy as np


class UgrPhotometry:
    """
    Luminance and solid angle for each luminaire in the UGR sum.

    All computation is vectorised: C_deg, gamma_deg and r_m must be 1-D
    numpy arrays of the same length (as returned by ``UgrGrid.angles()``).
    """

    @staticmethod
    def compute(
        lum_result,
        C_deg: np.ndarray,
        gamma_deg: np.ndarray,
        r_m: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Luminance L_i [cd/m²] and solid angle ω_i [sr] for each luminaire.

        Parameters
        ----------
        lum_result : eulumdat_luminance.LuminanceResult
            Luminance table built with ``full=True``.
        C_deg : np.ndarray, shape (n,)
            Photometric azimuth [°] for each luminaire.
        gamma_deg : np.ndarray, shape (n,)
            Elevation angle from nadir [°] for each luminaire.
        r_m : np.ndarray, shape (n,)
            Slant distance observer → luminaire [m].

        Returns
        -------
        L : np.ndarray, shape (n,)
            Apparent luminance [cd/m²].
        omega : np.ndarray, shape (n,)
            Solid angle [sr].
        """
        C_deg = np.asarray(C_deg, dtype=np.float64)
        gamma_deg = np.asarray(gamma_deg, dtype=np.float64)
        r_m = np.asarray(r_m, dtype=np.float64)

        L = np.asarray(
            lum_result.at(c_deg=C_deg, g_deg=gamma_deg), dtype=np.float64
        )
        A_p = np.asarray(
            lum_result.projected_area(c_deg=C_deg, g_deg=gamma_deg), dtype=np.float64
        )
        omega = A_p / r_m**2

        return L, omega
