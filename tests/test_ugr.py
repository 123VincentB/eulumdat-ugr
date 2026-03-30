# -*- coding: utf-8 -*-
"""
tests/test_ugr.py
-----------------
Tests for eulumdat-ugr.

Test organisation
-----------------
(à remplir au fil des étapes)

Step 2 — GuthTable (guth.py)
Step 3 — UgrGrid   (geometry.py)
Step 4 — BackgroundLuminance (background.py)
Step 5 — UgrPhotometry (photometry.py)
Step 6 — UgrCalculator (ugr.py)

Running
-------
    pytest                    # all tests
    pytest -v                 # verbose
    pytest tests/test_ugr.py  # this file only
"""

from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "input"
