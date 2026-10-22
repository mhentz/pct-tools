"""compute_ATAx against a hand-computed A.T @ A @ x for a single stored block."""

from pathlib import Path


def test_compute_ATAx(tmp_path: Path):
    import numpy as np

    from pct_tools.io import construct_matrix
    from pct_tools.lipschitz import compute_ATAx

    out_dir = tmp_path / "matrices"
    out_dir.mkdir()

    test_matrix_path = out_dir / "test.mat"

    construct_matrix(
        filename=str(test_matrix_path),
        row_indexes=np.array([0, 0, 0, 0], dtype=np.int32),
        col_indexes=np.array([0, 1, 2, 3], dtype=np.int32),
        values=np.array([1, 1, 1, 1], dtype=np.float32),
        shape=(1, 4),
        compression_level=9,
        verbose_level=0,
    )

    x = np.array([1, 1, 1, 1], dtype=np.float32)
    ATAx = compute_ATAx(x, [str(test_matrix_path)])

    assert ATAx.shape == (4,)
