"""Shared lookup helpers for MIIO-spec driven tests.

Tests stay model-agnostic: instead of naming a concrete device, they
pick the first MIIO_SPECS entry matching a structural criterion of the
generated table (dict/onoff/template/setter shapes).
"""
from custom_components.xiaomi_home.miot import miot_lan


def miio_spec_prop(pred):
    """First (model, siid, piid, config) in MIIO_SPECS matching pred.

    Fails loudly when no entry matches: the tests depend on the
    generated table containing the needed shape."""
    for model, spec in miot_lan.MIIO_SPECS.items():
        for key, cfg in spec.get('miio_specs', {}).items():
            if cfg and pred(cfg):
                _, siid, piid = key.split('.')
                return model, int(siid), int(piid), cfg
    raise LookupError('no MIIO_SPECS entry matches the test criterion')
