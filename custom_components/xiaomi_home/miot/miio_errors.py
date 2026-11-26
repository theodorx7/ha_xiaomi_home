"""Error codes of miio devices, grouped by model.

Each miio device has its own code set, so messages live per model, like
the command specs in miio_specs.py. Add a model's block as its error
codes are verified on hardware.
"""

MIIO_ERRORS: dict[str, dict[int, str]] = {
    'zhimi.airfresh.va4': {
        -6011: 'Device is powered off',
    },
}
