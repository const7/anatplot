"""Release Matplotlib figures created by graphics checks."""

import matplotlib
import pytest

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
