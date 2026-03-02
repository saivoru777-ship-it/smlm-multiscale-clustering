"""
Data loaders for external SMLM file formats.

Supported formats:
    - ThunderSTORM CSV (ImageJ plugin)
    - SMLM Challenge CSV (smlm-challenge.net)

Both loaders return a dict with standardized keys:
    positions : ndarray (N, 3) in nm
    frames    : ndarray (N,) frame numbers
    photons   : ndarray (N,) photon counts (or ones if unavailable)
    metadata  : dict with format-specific info
"""

import csv
import numpy as np
from pathlib import Path
from typing import Union


def load_thunderstorm_csv(
    filepath: Union[str, Path],
    x_col: str = 'x [nm]',
    y_col: str = 'y [nm]',
    z_col: str = 'z [nm]',
    frame_col: str = 'frame',
    photon_col: str = 'intensity [photon]',
    z_default: float = 0.0,
) -> dict:
    """
    Load a ThunderSTORM CSV localization table.

    ThunderSTORM exports CSV with columns like:
        "id","frame","x [nm]","y [nm]","sigma [nm]","intensity [photon]",
        "offset [photon]","bkgstd [photon]","chi2","uncertainty [nm]"

    For 3D data, a "z [nm]" column is present.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.
    x_col, y_col, z_col : str
        Column names for x, y, z coordinates in nm.
    frame_col : str
        Column name for frame number.
    photon_col : str
        Column name for photon count / intensity.
    z_default : float
        Default z value (nm) if z column is absent (2D data).

    Returns
    -------
    dict with keys: positions, frames, photons, metadata
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    rows = []
    with open(filepath, 'r', newline='') as f:
        # ThunderSTORM may use quoted headers with units
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        if headers is None:
            raise ValueError("CSV file has no header row")

        # Strip whitespace from headers
        headers = [h.strip().strip('"') for h in headers]

        for row in reader:
            # Strip quotes and whitespace from keys
            cleaned = {k.strip().strip('"'): v.strip().strip('"')
                       for k, v in row.items()}
            rows.append(cleaned)

    if len(rows) == 0:
        raise ValueError("CSV file contains no data rows")

    # Extract columns
    has_z = z_col in rows[0]
    has_photons = photon_col in rows[0]
    has_frames = frame_col in rows[0]

    n = len(rows)
    x = np.array([float(r[x_col]) for r in rows])
    y = np.array([float(r[y_col]) for r in rows])
    z = np.array([float(r[z_col]) for r in rows]) if has_z else np.full(n, z_default)
    frames = np.array([int(float(r[frame_col])) for r in rows]) if has_frames else np.arange(1, n + 1)
    photons = np.array([float(r[photon_col]) for r in rows]) if has_photons else np.ones(n)

    positions = np.column_stack([x, y, z])

    return {
        'positions': positions,
        'frames': frames,
        'photons': photons,
        'metadata': {
            'format': 'thunderstorm',
            'filepath': str(filepath),
            'n_localizations': n,
            'has_z': has_z,
            'has_photons': has_photons,
            'columns': list(rows[0].keys()),
        }
    }


def load_smlm_challenge_csv(
    filepath: Union[str, Path],
    x_col: str = 'x',
    y_col: str = 'y',
    z_col: str = 'z',
    frame_col: str = 'frame',
    photon_col: str = 'photons',
    z_default: float = 0.0,
    coordinate_unit: str = 'nm',
) -> dict:
    """
    Load an SMLM Challenge format CSV file.

    The SMLM Challenge (smlm-challenge.net) uses simple column headers:
        frame, x, y, z, photons
    Coordinates may be in nm or pixels.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.
    x_col, y_col, z_col : str
        Column names for coordinates.
    frame_col : str
        Column name for frame number.
    photon_col : str
        Column name for photon count.
    z_default : float
        Default z value if z column is absent.
    coordinate_unit : str
        'nm' or 'pixel'. If 'pixel', coordinates are NOT converted
        (caller must multiply by pixel size).

    Returns
    -------
    dict with keys: positions, frames, photons, metadata
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    rows = []
    with open(filepath, 'r', newline='') as f:
        # Try comma first, then tab
        sample = f.read(4096)
        f.seek(0)
        delimiter = '\t' if '\t' in sample.split('\n')[0] else ','
        reader = csv.DictReader(f, delimiter=delimiter)
        headers = reader.fieldnames
        if headers is None:
            raise ValueError("CSV file has no header row")

        for row in reader:
            cleaned = {k.strip(): v.strip() for k, v in row.items()}
            rows.append(cleaned)

    if len(rows) == 0:
        raise ValueError("CSV file contains no data rows")

    has_z = z_col in rows[0]
    has_photons = photon_col in rows[0]
    has_frames = frame_col in rows[0]

    n = len(rows)
    x = np.array([float(r[x_col]) for r in rows])
    y = np.array([float(r[y_col]) for r in rows])
    z = np.array([float(r[z_col]) for r in rows]) if has_z else np.full(n, z_default)
    frames = np.array([int(float(r[frame_col])) for r in rows]) if has_frames else np.arange(1, n + 1)
    photons = np.array([float(r[photon_col]) for r in rows]) if has_photons else np.ones(n)

    positions = np.column_stack([x, y, z])

    return {
        'positions': positions,
        'frames': frames,
        'photons': photons,
        'metadata': {
            'format': 'smlm_challenge',
            'filepath': str(filepath),
            'n_localizations': n,
            'has_z': has_z,
            'has_photons': has_photons,
            'coordinate_unit': coordinate_unit,
            'columns': list(rows[0].keys()),
        }
    }
