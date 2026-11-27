"""Read-side translation of the MIIO layer.

The write path is pinned by test_route_policy.py; these cover the read
path: the dict inverse mapping, the onoff cast, the read-side template
lambdas and the instant rejection of uncovered properties. One
write-path case guards the native miIO error reply of the set
transport.
"""
# pylint: disable=protected-access
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.xiaomi_home.miot.miot_lan import MIoTLan, MIIO_SPECS
from custom_components.xiaomi_home.miot.miot_error import (
    MIoTError, MIoTErrorCode)
from tests.miio_spec_helpers import miio_spec_prop

DID = 'dev1'
MIIO_MODEL = next(iter(MIIO_SPECS))


def _lan(model: str) -> MIoTLan:
    """A MIoTLan whose _lan_devices holds one MIIO-spec device."""
    lan = object.__new__(MIoTLan)
    lan._lan_devices = {DID: SimpleNamespace(did=DID, model=model)}
    return lan


def _template_raises_on_bad_input(cfg) -> bool:
    """Whether the entry's read template raises on a non-numeric value
    (the shape the failure test needs)."""
    tmpl = cfg.get('template')
    if not callable(tmpl):
        return False
    try:
        tmpl('oops', {}, 100)
    except Exception:  # pylint: disable=broad-exception-caught
        return True
    return False


async def _read(lan: MIoTLan, siid: int, piid: int, raw) -> object:
    """Read one property with the MIIO transport stubbed to `raw`."""
    with patch.object(
            MIoTLan, '_MIoTLan__call_api_async',
            AsyncMock(return_value={'result': [raw]})):
        return await lan._MIoTLan__get_prop_miio_async(
            lan._lan_devices[DID], siid=siid, piid=piid, timeout_ms=100)


@pytest.mark.asyncio
async def test_read_translates_dict_value_back_to_key() -> None:
    """A dict entry reads back value -> key (a non-digit key passes
    through unchanged)."""
    model, siid, piid, cfg = miio_spec_prop(
        lambda c: 'dict' in c and any(
            not k.isdigit() for k in c['dict']))
    key = next(k for k in cfg['dict'] if not k.isdigit())
    assert await _read(_lan(model), siid, piid, cfg['dict'][key]) == key


@pytest.mark.asyncio
async def test_read_casts_onoff_to_bool() -> None:
    """An onoff-format entry casts 'on'/'off' to booleans."""
    model, siid, piid, _ = miio_spec_prop(
        lambda c: c.get('format') == 'onoff')
    assert await _read(_lan(model), siid, piid, 'on') is True
    assert await _read(_lan(model), siid, piid, 'off') is False


@pytest.mark.asyncio
async def test_read_applies_template_lambda() -> None:
    """A template entry applies its lambda to the read: the result is
    the lambda's own value for the raw input."""
    model, siid, piid, cfg = miio_spec_prop(
        lambda c: callable(c.get('template')))
    for raw in (50, 0, 100, 25):
        try:
            expected = cfg['template'](raw, {}, 100)
            break
        except Exception:  # pylint: disable=broad-exception-caught
            continue
    else:
        pytest.fail('no numeric value passes the template lambda')
    assert await _read(_lan(model), siid, piid, raw) == expected


@pytest.mark.asyncio
async def test_read_template_failure_returns_raw_value(caplog) -> None:
    """A broken lambda must not break the read: log and return the raw
    value, the same way set_template failures are handled."""
    model, siid, piid, _ = miio_spec_prop(_template_raises_on_bad_input)
    with caplog.at_level(logging.ERROR):
        assert await _read(_lan(model), siid, piid, 'oops') == 'oops'
    assert 'Failed to execute template lambda' in caplog.text


@pytest.mark.asyncio
async def test_read_uncovered_property_raises_clean_miot_error() -> None:
    """The router needs the same instant 'skip LAN' rejection on reads
    as the write path raises."""
    with pytest.raises(MIoTError) as err:
        await _read(_lan(MIIO_MODEL), 99, 99, 1)
    assert err.value.code is MIoTErrorCode.CODE_INVALID_PROPERTY


@pytest.mark.asyncio
async def test_write_surfaces_native_device_error() -> None:
    """A MIIO error reply nests the code inside 'error'. The write must
    return that code like the gateway path does, not
    raise the internal 'Invalid result'. The parsing is generic: any
    code any MIIO device answers passes through unchanged."""
    model, siid, piid, _ = miio_spec_prop(lambda c: c.get('setter'))
    lan = _lan(model)
    cases = (
        ({'error': {'code': -6011, 'message': 'device_poweroff'}},
         {'code': -6011, 'message': 'device_poweroff'}),
        ({'error': {'code': -9999, 'message': 'method not found'}},
         {'code': -9999, 'message': 'method not found'}),
        # the transport's own timeout answers carry 'error' as a string:
        # they must keep flowing through the top-level 'code' branch
        ({'code': MIoTErrorCode.CODE_TIMEOUT.value, 'error': 'timeout'},
         {'code': MIoTErrorCode.CODE_TIMEOUT.value, 'error': 'timeout'}),
    )
    for reply, expected in cases:
        with patch.object(
                MIoTLan, '_MIoTLan__call_api_async',
                AsyncMock(return_value=reply)):
            result = await lan._MIoTLan__set_prop_miio_async(
                lan._lan_devices[DID], siid=siid, piid=piid, value=True,
                timeout_ms=100)
        assert result == expected


def _sub_lan(**overrides) -> MIoTLan:
    """A MIoTLan stub for the LAN-subscribe gate."""
    lan = object.__new__(MIoTLan)
    lan._ignore_mips_service = overrides.get('ignore', False)
    lan._enable_subscribe = overrides.get('enable', True)
    services = overrides.get('services', [{'group_id': 'gw1'}])
    lan._mips_service = SimpleNamespace(get_services=lambda: services)
    lan._init_done = True
    lan._internal_loop = SimpleNamespace(
        call_soon_threadsafe=MagicMock())
    return lan


def test_lan_subscribe_blocked_when_gateway_present() -> None:
    """With a central gateway present, LAN subscription is blocked at
    the code level regardless of the enable_subscribe option: the
    gateway delivers the pushes itself."""
    lan = _sub_lan(enable=True, services=[{'group_id': 'gw1'}])
    assert not lan._lan_subscribe_allowed()
    assert not lan.sub_prop('d1', lambda m, c: None)
    assert not lan._internal_loop.call_soon_threadsafe.called


def test_lan_subscribe_allowed_without_gateway() -> None:
    """Without a gateway the enable_subscribe option rules."""
    lan = _sub_lan(enable=True, services=[])
    assert lan._lan_subscribe_allowed()
    assert lan.sub_prop('d1', lambda m, c: None)
    assert lan._internal_loop.call_soon_threadsafe.called


def test_lan_subscribe_option_still_gates() -> None:
    """The option stays a gate: off means no LAN subscription, gateway
    or not."""
    assert not _sub_lan(enable=False, services=[])._lan_subscribe_allowed()
    assert _sub_lan(enable=False, ignore=True)._lan_subscribe_allowed() is False


def test_lan_subscribe_ignores_gateway_for_non_cn() -> None:
    """_ignore_mips_service (non-CN regions): the gateway is not part
    of the rule, the option rules."""
    assert _sub_lan(enable=True, ignore=True,
                    services=[{'group_id': 'gw1'}])._lan_subscribe_allowed()
