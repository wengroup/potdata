def kbar_to_eV_per_A_cube() -> float:
    """Get the units convert ratio from kbar to eV/A^3."""
    ratio = 0.1 / 160.2176634  # kbar to GPa and then to eV/A^3

    return ratio
