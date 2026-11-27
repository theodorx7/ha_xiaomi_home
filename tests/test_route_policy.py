"""Write, action and poll policy of the MIoT client.

The first block pins the base chain for devices without a MIIO spec
(unchanged by the migration). The second covers the covered/uncovered
rule for MIIO-spec devices: the write, the action, the control path
label and the error rendering. The third pins the background poll
delivery per mode (matrices 9.1-9.4 of the migration plan): the LAN
pre-pass, the reader skips and the freeze of covered props when the
LAN is down. The layer lifecycle and the client vote/composition
tests live in tests/test_lan_gates.py.
"""
# pylint: disable=protected-access
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.xiaomi_home.miot import miot_lan
from custom_components.xiaomi_home.miot.miot_client import CtrlMode, MIoTClient
from custom_components.xiaomi_home.miot.miot_error import (
    MIoTClientError, MIoTError, MIoTErrorCode, MIoTLanError)
from tests.miio_spec_helpers import miio_spec_prop

DID = 'dev1'
MIIO_MODEL = next(iter(miot_lan.MIIO_SPECS))
# The first table entry covers service 2 property 1, the fixed
# coordinates the poll tests queue under miio1.
MIIO_SIID, MIIO_PIID = 2, 1


def _route_client(*, ctrl_mode, priority='cloud_first',
                  lan_online=True, cloud_online=True, gw_online=False,
                  model=None):
    """A MIoTClient whose cloud/LAN/gateway doubles record every call.

    `lan_online=False` makes the LAN route unavailable at the routing
    layer (device not in the online list), the case the current code
    skips to the cloud (AUTO) or fails on (LOCAL). `model` names a
    MIIO-spec model, which turns the device into a LAN-first one.
    """
    calls: list[str] = []

    def route(kind, value=None):
        async def _call(*_args, **_kwargs):
            calls.append(kind)
            return value if value is not None else (
                {'code': 0} if 'set' in kind else {'code': 0, 'out': []})
        return AsyncMock(side_effect=_call)

    with patch.object(MIoTClient, '__init__', lambda self, *a, **k: None):
        client = MIoTClient()  # pylint: disable=no-value-for-parameter
    client._ctrl_mode = ctrl_mode
    client._entry_data = {'poll_priority': priority}
    client._network = SimpleNamespace(network_status=True)
    client._i18n = SimpleNamespace(translate_str=lambda key: key)
    client._device_list_cache = {DID: {'model': model} if model else {}}
    client._device_list_cloud = ({DID: {'online': cloud_online}}
                                  if cloud_online else {})
    client._device_list_lan = ({DID: {'online': lan_online}}
                                if lan_online else {})
    client._http = SimpleNamespace(
        set_prop_async=route('cloud.set', [{'code': 0}]),
        get_prop_async=route('cloud.get', 5),
        action_async=route('cloud.action'))
    client._miot_lan = SimpleNamespace(
        set_prop_async=route('lan.set'),
        get_prop_async=route('lan.get', 7),
        action_async=route('lan.action'))
    client._mips_local = ({DID + '.gw': SimpleNamespace(
        set_prop_async=route('gw.set'),
        get_prop_async=route('gw.get', 9),
        action_async=route('gw.action'))} if gw_online else {})
    client._device_list_gateway = (
        {DID: {'online': True, 'specv2_access': True,
               'group_id': DID + '.gw'}} if gw_online else {})
    client.calls = calls
    return client


def _miio_lan(model: str):
    """A MIoTLan whose _lan_devices contains one MIIO-spec device."""
    lan = object.__new__(miot_lan.MIoTLan)
    lan._lan_devices = {DID: SimpleNamespace(did=DID, model=model)}
    return lan


def _covered_prop() -> tuple[str, int, int]:
    """A model and its covered, writable property coordinates."""
    model, siid, piid, _ = miio_spec_prop(lambda cfg: cfg.get('setter'))
    return model, siid, piid


# --- The unchanged base chain --------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize('ctrl_mode', [CtrlMode.AUTO, CtrlMode.LOCAL])
async def test_set_prefers_lan_over_cloud(ctrl_mode) -> None:
    """AUTO and LOCAL both try the gateway, then LAN, before the cloud."""
    client = _route_client(ctrl_mode=ctrl_mode)
    assert await client.set_prop_async(DID, 2, 1, 1) is True
    assert client.calls == ['lan.set']


@pytest.mark.asyncio
async def test_set_prefers_gateway_when_present() -> None:
    client = _route_client(ctrl_mode=CtrlMode.AUTO, gw_online=True)
    assert await client.set_prop_async(DID, 2, 1, 1) is True
    assert client.calls == ['gw.set']


@pytest.mark.asyncio
async def test_set_falls_back_to_cloud_when_lan_not_online() -> None:
    """Routing-level LAN miss (device not online) goes to the cloud."""
    client = _route_client(ctrl_mode=CtrlMode.AUTO, lan_online=False)
    assert await client.set_prop_async(DID, 2, 1, 1) is True
    assert client.calls == ['cloud.set']


@pytest.mark.asyncio
async def test_set_local_mode_does_not_use_cloud() -> None:
    """LOCAL never falls back to the cloud: with the LAN route missing it
    raises the user-facing execution error instead."""
    client = _route_client(ctrl_mode=CtrlMode.LOCAL, lan_online=False)
    with pytest.raises(MIoTClientError):
        await client.set_prop_async(DID, 2, 1, 1)
    assert not client.calls


@pytest.mark.asyncio
async def test_set_cloud_mode_skips_local_routes() -> None:
    client = _route_client(ctrl_mode=CtrlMode.CLOUD)
    assert await client.set_prop_async(DID, 2, 1, 1) is True
    assert client.calls == ['cloud.set']


@pytest.mark.asyncio
async def test_get_cloud_first_order() -> None:
    """Default priority cloud_first: the cloud value wins."""
    client = _route_client(ctrl_mode=CtrlMode.AUTO)
    assert await client.get_prop_async(DID, 2, 1) == 5
    assert client.calls == ['cloud.get']


@pytest.mark.asyncio
async def test_get_local_first_order() -> None:
    """local_first: the LAN value wins without a cloud call."""
    client = _route_client(ctrl_mode=CtrlMode.AUTO, priority='local_first')
    assert await client.get_prop_async(DID, 2, 1) == 7
    assert client.calls == ['lan.get']


@pytest.mark.asyncio
async def test_get_local_first_falls_back_to_cloud() -> None:
    client = _route_client(ctrl_mode=CtrlMode.AUTO, priority='local_first',
                           lan_online=False)
    assert await client.get_prop_async(DID, 2, 1) == 5
    assert client.calls == ['cloud.get']


@pytest.mark.asyncio
async def test_get_local_first_uses_gateway_before_lan() -> None:
    client = _route_client(ctrl_mode=CtrlMode.AUTO, priority='local_first',
                           gw_online=True)
    assert await client.get_prop_async(DID, 2, 1) == 9
    assert client.calls == ['gw.get']


@pytest.mark.asyncio
async def test_get_local_mode_uses_only_local_readers() -> None:
    """In LOCAL mode the cloud reader short-circuits: the LAN value wins."""
    client = _route_client(ctrl_mode=CtrlMode.LOCAL)
    assert await client.get_prop_async(DID, 2, 1) == 7
    assert client.calls == ['lan.get']


@pytest.mark.asyncio
async def test_get_local_mode_returns_none_when_no_local_route() -> None:
    """LOCAL with no LAN route: the cloud is still not called."""
    client = _route_client(ctrl_mode=CtrlMode.LOCAL, lan_online=False)
    assert await client.get_prop_async(DID, 2, 1) is None
    assert not client.calls


@pytest.mark.asyncio
async def test_action_prefers_lan_over_cloud() -> None:
    client = _route_client(ctrl_mode=CtrlMode.AUTO)
    assert await client.action_async(DID, 5, 1, []) == []
    assert client.calls == ['lan.action']


@pytest.mark.asyncio
async def test_action_falls_back_to_cloud_when_lan_not_online() -> None:
    client = _route_client(ctrl_mode=CtrlMode.AUTO, lan_online=False)
    assert await client.action_async(DID, 5, 1, []) == []
    assert client.calls == ['cloud.action']


@pytest.mark.asyncio
async def test_action_local_mode_does_not_use_cloud() -> None:
    """LOCAL with the LAN route missing: MIoTClientError, no cloud call."""
    client = _route_client(ctrl_mode=CtrlMode.LOCAL, lan_online=False)
    with pytest.raises(MIoTClientError):
        await client.action_async(DID, 5, 1, [])
    assert not client.calls


# --- Translator rejections -----------------------------


@pytest.mark.asyncio
async def test_set_unsupported_prop_raises_clean_miot_error() -> None:
    """Regression for the enum bug: the unsupported-property branch
    raises MIoTError. Before the fix it raised AttributeError, because
    MIoTErrorCode had no member for this branch to name."""
    lan = _miio_lan(next(iter(miot_lan.MIIO_SPECS)))
    with pytest.raises(MIoTError) as err:
        await lan._MIoTLan__set_prop_miio_async(
            lan._lan_devices[DID], siid=99, piid=99, value=1,
            timeout_ms=1000)
    assert err.value.code is MIoTErrorCode.CODE_INVALID_PROPERTY


@pytest.mark.asyncio
async def test_set_read_only_prop_raises_clean_miot_error() -> None:
    """The no-setter branch (Property not writable) also stays MIoTError:
    any spec entry without a setter is read-only."""
    model, siid, piid, _ = miio_spec_prop(lambda cfg: not cfg.get('setter'))
    lan = _miio_lan(model)
    with pytest.raises(MIoTError) as err:
        await lan._MIoTLan__set_prop_miio_async(
            lan._lan_devices[DID], siid=siid, piid=piid, value=1,
            timeout_ms=1000)
    assert err.value.code is MIoTErrorCode.CODE_PROPERTY_NOT_WRITABLE


# --- The covered/uncovered rule for MIIO-spec devices --------


@pytest.mark.asyncio
async def test_set_miio_covered_prefers_lan_even_with_gateway() -> None:
    """A covered prop of a MIIO-spec device is served by the LAN even
    while the gateway is visible: the gateway skips it (matrix 9.2)."""
    model, siid, piid = _covered_prop()
    client = _route_client(ctrl_mode=CtrlMode.AUTO, gw_online=True,
                           model=model)
    assert await client.set_prop_async(DID, siid, piid, 1) is True
    assert client.calls == ['lan.set']


@pytest.mark.asyncio
async def test_set_miio_covered_uses_lan_without_gateway() -> None:
    """Matrix 9.1: without a gateway a covered MIIO prop takes the LAN
    like any other device's prop."""
    model, siid, piid = _covered_prop()
    client = _route_client(ctrl_mode=CtrlMode.AUTO, model=model)
    assert await client.set_prop_async(DID, siid, piid, 1) is True
    assert client.calls == ['lan.set']


@pytest.mark.asyncio
async def test_set_miio_covered_local_mode_uses_lan() -> None:
    """Matrix 9.2: LOCAL keeps the LAN choice for a covered MIIO prop."""
    model, siid, piid = _covered_prop()
    client = _route_client(ctrl_mode=CtrlMode.LOCAL, gw_online=True,
                           model=model)
    assert await client.set_prop_async(DID, siid, piid, 1) is True
    assert client.calls == ['lan.set']


@pytest.mark.asyncio
async def test_set_miio_covered_cloud_mode_uses_cloud() -> None:
    """Matrix 9.2: CLOUD serves a covered MIIO prop through the cloud
    like any other prop, gateway or not."""
    model, siid, piid = _covered_prop()
    client = _route_client(ctrl_mode=CtrlMode.CLOUD, gw_online=True,
                           model=model)
    assert await client.set_prop_async(DID, siid, piid, 1) is True
    assert client.calls == ['cloud.set']


@pytest.mark.asyncio
async def test_set_miio_uncovered_prop_goes_to_gateway() -> None:
    """An uncovered prop has no MIIO command: the gateway serves it in
    its own scale, as in the base (matrix 9.2)."""
    client = _route_client(ctrl_mode=CtrlMode.AUTO, gw_online=True,
                           model=MIIO_MODEL)
    assert await client.set_prop_async(DID, 99, 99, 1) is True
    assert client.calls == ['gw.set']


@pytest.mark.asyncio
async def test_set_miio_covered_lan_down_falls_to_cloud() -> None:
    """Matrix 9.2 (O2): with the LAN route missing, a covered MIIO prop
    falls through to the cloud like any other prop would."""
    model, siid, piid = _covered_prop()
    client = _route_client(ctrl_mode=CtrlMode.AUTO, gw_online=True,
                           lan_online=False, model=model)
    assert await client.set_prop_async(DID, siid, piid, 1) is True
    assert client.calls == ['cloud.set']


@pytest.mark.asyncio
async def test_set_miio_covered_lan_refusal_is_final() -> None:
    """Matrix 9.2: a device refusal over the LAN ends the write with the
    device's own code - no gateway/cloud retry."""
    model, siid, piid = _covered_prop()
    client = _route_client(ctrl_mode=CtrlMode.AUTO, gw_online=True,
                           model=model)

    async def _lan_refuse(*_args, **_kwargs):
        client.calls.append('lan.set')
        return {'code': -6011, 'message': 'device_poweroff'}

    client._miot_lan = SimpleNamespace(
        set_prop_async=AsyncMock(side_effect=_lan_refuse))
    with pytest.raises(MIoTClientError) as err:
        await client.set_prop_async(DID, siid, piid, 1)
    assert client.calls == ['lan.set']
    assert '-6011' in str(err.value)


@pytest.mark.asyncio
async def test_set_non_miio_error_stays_final() -> None:
    """Without a MIIO spec the first path error still ends the
    operation - no silent cloud retry."""
    client = _route_client(ctrl_mode=CtrlMode.AUTO, gw_online=True)

    async def _gw_fail(*_args, **_kwargs):
        client.calls.append('gw.set')
        return {'code': -704042011}

    client._mips_local = {DID + '.gw': SimpleNamespace(
        set_prop_async=AsyncMock(side_effect=_gw_fail),
        get_prop_async=AsyncMock(),
        action_async=AsyncMock())}
    with pytest.raises(MIoTClientError):
        await client.set_prop_async(DID, 2, 1, 1)
    assert client.calls == ['gw.set']


@pytest.mark.asyncio
@pytest.mark.parametrize('error,message', [
    (MIoTError('Unsupported property',
               MIoTErrorCode.CODE_INVALID_PROPERTY),
     'Unsupported property'),
    (MIoTError('Property not writable',
               MIoTErrorCode.CODE_PROPERTY_NOT_WRITABLE),
     'Property not writable'),
    (MIoTLanError('miot lan is not ready'), 'miot lan is not ready'),
])
async def test_set_wraps_the_translators_miot_error(
        error, message) -> None:
    """The LAN branch wraps the translator's MIoTError into
    MIoTClientError, so the entity layer receives an error it handles
    instead of an unhandled exception (V22)."""
    model, siid, piid = _covered_prop()
    client = _route_client(ctrl_mode=CtrlMode.AUTO, model=model)
    client._miot_lan = SimpleNamespace(
        set_prop_async=AsyncMock(side_effect=error))
    with pytest.raises(MIoTClientError) as err:
        await client.set_prop_async(DID, siid, piid, 1)
    assert message in str(err.value)


@pytest.mark.asyncio
async def test_action_miio_never_tries_lan() -> None:
    """No MIIO action translator exists: a LAN attempt would only burn a
    timeout, so a MIIO-spec device's action takes the gateway."""
    client = _route_client(ctrl_mode=CtrlMode.AUTO, gw_online=True,
                           model=MIIO_MODEL)
    assert await client.action_async(DID, 5, 1, []) == []
    assert client.calls == ['gw.action']


@pytest.mark.asyncio
async def test_action_miio_without_gateway_skips_lan() -> None:
    """Even without a gateway a MIIO-spec action skips the LAN (no
    translator) and goes to the cloud."""
    client = _route_client(ctrl_mode=CtrlMode.AUTO, model=MIIO_MODEL)
    assert await client.action_async(DID, 5, 1, []) == []
    assert client.calls == ['cloud.action']


def test_control_path_reports_the_miio_translator() -> None:
    client = _route_client(ctrl_mode=CtrlMode.AUTO, gw_online=True,
                           model=MIIO_MODEL)
    assert client.get_device_control_path(DID) == 'miIO'
    # LAN gone: the honest answer is the gateway, as for any device
    client._device_list_lan = {}
    assert client.get_device_control_path(DID) == 'Gateway'
    # a non-MIIO device never shows the MIIO label
    client = _route_client(ctrl_mode=CtrlMode.AUTO, gw_online=True)
    assert client.get_device_control_path(DID) == 'Gateway'


def test_exec_error_prefers_model_block_then_common() -> None:
    """The write error text looks the code up in the device's own miio
    error block first, then in the common block; unknown codes fall to
    'Unknown error, code='."""
    i18n = {
        'miot.client.device_exec_error': 'Execution error',
        'error.common.-10000': 'Unknown error',
    }
    with patch.object(MIoTClient, '__init__', lambda self, *a, **k: None):
        client = MIoTClient()  # pylint: disable=no-value-for-parameter

    def _lookup(key):
        return i18n.get(key)

    client._i18n = SimpleNamespace(translate_str=_lookup)
    with patch.dict(
            'custom_components.xiaomi_home.miot.miot_client.MIIO_ERRORS',
            {'some.model': {-6011: 'Device is powered off'}}):
        client._device_list_cache = {DID: {'model': 'some.model'}}
        msg = client._MIoTClient__get_exec_error_with_rc(DID, -6011)
        assert msg == 'Execution error, Device is powered off'
        # an unknown code in the model's block falls to common
        msg = client._MIoTClient__get_exec_error_with_rc(DID, -9999)
        assert msg == 'Execution error, Unknown error, code=-9999'
    # a model without a block falls straight to common
    client._device_list_cache = {DID: {'model': 'other.model'}}
    msg = client._MIoTClient__get_exec_error_with_rc(DID, -6011)
    assert msg == 'Execution error, Unknown error, code=-6011'


def test_sub_from_skips_prop_subscription_for_miio_devices() -> None:
    """MIIO-spec devices get no property pushes in modes with local
    paths: gateway/cloud pushes speak the publication scale while the
    LAN poll speaks the firmware scale, so their values come from the
    LAN poll and the write echo. Events stay subscribed (no scale
    conflict); non-MIIO devices keep both subscriptions."""
    with patch.object(MIoTClient, '__init__', lambda self, *a, **k: None):
        client = MIoTClient()  # pylint: disable=no-value-for-parameter
    client._ctrl_mode = CtrlMode.AUTO
    client._device_is_miio = lambda did: did == 'miio1'
    mips = SimpleNamespace(sub_prop=MagicMock(), sub_event=MagicMock())
    client._mips_local = {'gw1': mips}

    client._MIoTClient__sub_from('gw1', 'miio1')
    client._MIoTClient__sub_from('gw1', 'norm1')

    assert mips.sub_prop.call_count == 1  # norm1 only
    assert mips.sub_event.call_count == 2  # both devices
    miio_prop_call = [c for c in mips.sub_prop.call_args_list
                      if c.kwargs['did'] == 'miio1']
    assert miio_prop_call == []


def test_sub_from_subscribes_miio_prop_in_cloud_mode() -> None:
    """In CLOUD mode the device is served entirely on its MIoT spec:
    writes and pushes speak the same scale there, so MIIO-spec devices
    get the cloud property subscription like any other device."""
    with patch.object(MIoTClient, '__init__', lambda self, *a, **k: None):
        client = MIoTClient()  # pylint: disable=no-value-for-parameter
    client._ctrl_mode = CtrlMode.CLOUD
    client._device_is_miio = lambda did: did == 'miio1'
    mips = SimpleNamespace(sub_prop=MagicMock(), sub_event=MagicMock())
    client._mips_cloud = mips

    client._MIoTClient__sub_from('cloud', 'miio1')

    mips.sub_prop.assert_called_once_with(
        did='miio1', handler=client.on_prop_msg)
    mips.sub_event.assert_called_once()


# --- The background poll --------------------------------------


def _poll_client(*, ctrl_mode, priority='cloud_first', gw=True):
    """A client with one MIIO-spec and one plain device in the refresh
    queue, whose manager doubles record every pass with its did group."""
    with patch.object(MIoTClient, '__init__', lambda self, *a, **k: None):
        client = MIoTClient()  # pylint: disable=no-value-for-parameter
    client._ctrl_mode = ctrl_mode
    client._entry_data = {'poll_priority': priority}
    client._device_list_cache = {
        'miio1': {'model': MIIO_MODEL},
        'norm1': {'model': 'other.model'}}
    # Behind a gateway only MIIO-spec devices are LAN-registered;
    # without one, all LAN-capable devices are, as always.
    client._device_list_lan = ({'miio1': {'online': True}} if gw else {
        'miio1': {'online': True}, 'norm1': {'online': True}})
    client._device_list_cloud = {
        'miio1': {'online': True}, 'norm1': {'online': True}}
    client._device_list_gateway = (
        {'miio1': {'online': True, 'specv2_access': True, 'group_id': 'g'},
         'norm1': {'online': True, 'specv2_access': True, 'group_id': 'g'}}
        if gw else {})
    client._mips_local = {'g': object()} if gw else {}
    client._network = SimpleNamespace(network_status=True)
    client._refresh_props_list = {
        'miio1|2|1': {'did': 'miio1', 'siid': MIIO_SIID, 'piid': MIIO_PIID},
        'norm1|2|1': {'did': 'norm1', 'siid': 2, 'piid': 1}}
    client._refresh_props_attempts = {}
    client._refresh_props_cloud_missed = set()
    client._refresh_props_retry_count = 0
    client._refresh_props_attempted = False
    client._refresh_props_timer = None
    client._main_loop = MagicMock()
    calls: list[tuple[str, frozenset]] = []

    def rec(name, result=True):
        async def _handler(only_dids=None):
            calls.append((name, only_dids))
            return result
        return AsyncMock(side_effect=_handler)

    client._lan_manager = SimpleNamespace(
        refresh_props_from_lan=rec('lan'),
        refresh_props_from_gw=rec('gw'),
        can_refresh=lambda did: True)
    client._cloud_manager = SimpleNamespace(refresh_props=rec('cloud'))
    client.calls = calls
    return client


def _real_managers(client):
    """Swap the recorder stubs for the real managers."""
    from custom_components.xiaomi_home.miot.miot_cloud_manager import (  # pylint: disable=import-outside-toplevel
        MIoTCloudManager)
    from custom_components.xiaomi_home.miot.miot_lan_manager import (  # pylint: disable=import-outside-toplevel
        MIoTLanManager)

    client._cloud_manager = MIoTCloudManager(client)
    client._lan_manager = MIoTLanManager(client)


def _deliver(client):
    """Record delivered poll values as (did, value) pairs."""
    client.delivered = []
    client.on_prop_msg = lambda params, ctx: client.delivered.append(
        (params['did'], params['value']))


@pytest.mark.asyncio
async def test_poll_miio_property_goes_to_lan_under_real_managers() -> None:
    """Matrix 9.3, AUTO cloud_first without a gateway: the LAN pre-pass
    reads the MIIO device, the cloud batch carries the plain device
    only - the MIIO property never leaves for the cloud."""
    client = _poll_client(ctrl_mode=CtrlMode.AUTO, gw=False)
    cloud_calls: list[list[str]] = []

    async def _cloud(params):
        cloud_calls.append([p['did'] for p in params])
        return [{'did': p['did'], 'siid': p['siid'], 'piid': p['piid'],
                 'value': 5} for p in params if p['did'] == 'norm1']

    client._http = SimpleNamespace(get_props_async=AsyncMock(
        side_effect=_cloud))
    client._miot_lan = SimpleNamespace(
        init_done=True, get_prop_async=AsyncMock(return_value=42))
    _deliver(client)
    _real_managers(client)

    await client._MIoTClient__refresh_props_handler()

    assert sorted(client.delivered) == [('miio1', 42), ('norm1', 5)]
    assert cloud_calls == [['norm1']]
    client._miot_lan.get_prop_async.assert_awaited_once_with(
        did='miio1', siid=MIIO_SIID, piid=MIIO_PIID, timeout_ms=6000)
    assert not client._refresh_props_list


@pytest.mark.asyncio
async def test_poll_miio_covered_prop_lan_miss_stays_queued() -> None:
    """A MIIO-covered property the LAN cannot read stays queued for the
    LAN retry: the gateway/cloud passes speak the publication scale and
    must not serve covered props (publication vs firmware scale
    conflict) - under the real managers."""
    client = _poll_client(ctrl_mode=CtrlMode.AUTO, gw=False)
    cloud_calls: list[list[str]] = []

    async def _cloud(params):
        cloud_calls.append([p['did'] for p in params])
        # the cloud has no value for the MIIO device's property
        return [{'did': p['did'], 'siid': p['siid'], 'piid': p['piid'],
                 'value': 5} for p in params if p['did'] == 'norm1']

    client._http = SimpleNamespace(get_props_async=AsyncMock(
        side_effect=_cloud))
    client._miot_lan = SimpleNamespace(
        init_done=True, get_prop_async=AsyncMock(return_value=None))
    _deliver(client)
    _real_managers(client)

    await client._MIoTClient__refresh_props_handler()

    assert client.delivered == [('norm1', 5)]
    # the covered MIIO prop is not cloud-served (publication scale)
    assert cloud_calls == [['norm1']]
    # the LAN miss left the MIIO property queued for the LAN retry
    assert f'miio1|{MIIO_SIID}|{MIIO_PIID}' in client._refresh_props_list
    assert client._refresh_props_attempts[
        f'miio1|{MIIO_SIID}|{MIIO_PIID}'] == 1


@pytest.mark.asyncio
async def test_poll_covered_freezes_when_lan_offline() -> None:
    """Matrix 9.3, the D1 regression: with the LAN gone the cloud must
    not read a covered MIIO prop in the publication scale - the value
    freezes (stays queued) instead of being corrupted."""
    client = _poll_client(ctrl_mode=CtrlMode.AUTO)
    client._device_list_lan = {}  # the LAN is gone
    cloud_calls: list[list[str]] = []

    async def _cloud(params):
        cloud_calls.append([p['did'] for p in params])
        return [{'did': p['did'], 'siid': p['siid'], 'piid': p['piid'],
                 'value': 5} for p in params if p['did'] == 'norm1']

    client._http = SimpleNamespace(get_props_async=AsyncMock(
        side_effect=_cloud))
    client._miot_lan = SimpleNamespace(
        init_done=True, get_prop_async=AsyncMock(return_value=None))
    _deliver(client)
    _real_managers(client)

    await client._MIoTClient__refresh_props_handler()

    assert client.delivered == [('norm1', 5)]
    # the covered MIIO prop never reached the cloud batch
    assert cloud_calls == [['norm1']]
    assert f'miio1|{MIIO_SIID}|{MIIO_PIID}' in client._refresh_props_list


@pytest.mark.asyncio
async def test_poll_local_first_gw_serves_the_rest() -> None:
    """Matrix 9.3, AUTO local_first behind a gateway: the LAN pre-pass
    reads the covered prop, the gateway pass serves the uncovered one
    and the plain device; the cloud is never asked."""
    client = _poll_client(ctrl_mode=CtrlMode.AUTO, priority='local_first')
    client._refresh_props_list = {
        f'miio1|{MIIO_SIID}|{MIIO_PIID}': {
            'did': 'miio1', 'siid': MIIO_SIID, 'piid': MIIO_PIID},
        'miio1|99|99': {'did': 'miio1', 'siid': 99, 'piid': 99},
        'norm1|2|1': {'did': 'norm1', 'siid': 2, 'piid': 1}}
    gw_calls: list[tuple] = []

    async def _gw(did, siid, piid, **_kwargs):
        gw_calls.append((did, siid, piid))
        return 100

    client._mips_local = {'g': SimpleNamespace(
        get_prop_async=AsyncMock(side_effect=_gw))}
    cloud_calls: list[list[str]] = []

    async def _cloud(params):
        cloud_calls.append([p['did'] for p in params])
        return []

    client._http = SimpleNamespace(get_props_async=AsyncMock(
        side_effect=_cloud))
    client._miot_lan = SimpleNamespace(
        init_done=True, get_prop_async=AsyncMock(return_value=42))
    _deliver(client)
    _real_managers(client)

    await client._MIoTClient__refresh_props_handler()

    assert sorted(client.delivered) == [
        ('miio1', 42), ('miio1', 100), ('norm1', 100)]
    # the gateway served the uncovered MIIO prop and the plain device
    assert sorted(gw_calls) == [('miio1', 99, 99), ('norm1', 2, 1)]
    assert not cloud_calls
    assert not client._refresh_props_list


@pytest.mark.asyncio
async def test_poll_lan_pass_skips_miio_uncovered_props() -> None:
    """The LAN pre-pass skips a MIIO-uncovered prop (the translator has
    no command for it) instead of burning its one-prop slot on a doomed
    attempt: the base chain then serves it through the gateway in its
    own scale within the same cycle."""
    client = _poll_client(ctrl_mode=CtrlMode.AUTO)
    client._refresh_props_list = {
        'miio1|99|99': {'did': 'miio1', 'siid': 99, 'piid': 99},
        f'miio1|{MIIO_SIID}|{MIIO_PIID}': {
            'did': 'miio1', 'siid': MIIO_SIID, 'piid': MIIO_PIID}}
    client._miot_lan = SimpleNamespace(
        init_done=True, get_prop_async=AsyncMock(return_value=42))
    gw_calls: list[tuple] = []

    async def _gw(did, siid, piid, **_kwargs):
        gw_calls.append((did, siid, piid))
        return 100

    client._mips_local = {'g': SimpleNamespace(
        get_prop_async=AsyncMock(side_effect=_gw))}
    client._cloud_manager = SimpleNamespace(
        refresh_props=AsyncMock(return_value=False))
    client.delivered = []
    client.on_prop_msg = lambda params, ctx: client.delivered.append(
        (params['siid'], params['piid'], params['value']))
    from custom_components.xiaomi_home.miot.miot_lan_manager import (  # pylint: disable=import-outside-toplevel
        MIoTLanManager)
    client._lan_manager = MIoTLanManager(client)

    await client._MIoTClient__refresh_props_handler()

    # the LAN pre-pass read the covered prop only
    client._miot_lan.get_prop_async.assert_awaited_once_with(
        did='miio1', siid=MIIO_SIID, piid=MIIO_PIID, timeout_ms=6000)
    # the gateway pass then served the uncovered prop in its own scale
    assert gw_calls == [('miio1', 99, 99)]
    assert sorted(client.delivered) == [
        (MIIO_SIID, MIIO_PIID, 42), (99, 99, 100)]
    assert not client._refresh_props_list


@pytest.mark.asyncio
async def test_poll_cloud_pass_skips_miio_covered_props() -> None:
    """A MIIO-covered property is not cloud-serviceable: the cloud
    speaks the publication scale while the entity display is the
    firmware scale, so the cloud pass leaves it queued for the LAN
    pass; the non-MIIO device's property is served as usual."""
    client = _poll_client(ctrl_mode=CtrlMode.AUTO, gw=False)
    cloud_calls: list[list[str]] = []

    async def _cloud(params):
        cloud_calls.append([p['did'] for p in params])
        return [{'did': p['did'], 'siid': p['siid'], 'piid': p['piid'],
                 'value': 5} for p in params if p['did'] == 'norm1']

    client._http = SimpleNamespace(get_props_async=AsyncMock(
        side_effect=_cloud))
    client._miot_lan = SimpleNamespace(
        init_done=True, get_prop_async=AsyncMock(return_value=None))
    _deliver(client)
    _real_managers(client)

    await client._MIoTClient__refresh_props_handler()

    # the cloud was asked about norm1 only; the covered MIIO prop stayed
    # queued for the LAN pass
    assert cloud_calls == [['norm1']]
    assert client.delivered == [('norm1', 5)]
    assert f'miio1|{MIIO_SIID}|{MIIO_PIID}' in client._refresh_props_list


@pytest.mark.asyncio
async def test_poll_cloud_pass_serves_miio_props_in_cloud_mode() -> None:
    """In CLOUD mode the device is served entirely on its MIoT spec:
    there is no LAN pass behind the explicit cloud-only choice, and
    writes and pushes speak the same MIoT scale - so the cloud pass
    reads MIIO-covered props like any other property. Nothing is left
    queued for a LAN reader that does not exist."""
    client = _poll_client(ctrl_mode=CtrlMode.CLOUD)
    cloud_calls: list[list[str]] = []

    async def _cloud(params):
        cloud_calls.append([p['did'] for p in params])
        return [{'did': p['did'], 'siid': p['siid'], 'piid': p['piid'],
                 'value': 5} for p in params]

    client._http = SimpleNamespace(get_props_async=AsyncMock(
        side_effect=_cloud))
    client._miot_lan = SimpleNamespace(
        init_done=True, get_prop_async=AsyncMock(return_value=None))
    _deliver(client)
    _real_managers(client)

    await client._MIoTClient__refresh_props_handler()

    # CLOUD mode: the cloud was asked about both devices in one pass,
    # the covered MIIO prop is delivered like any other
    assert len(cloud_calls) == 1
    assert sorted(cloud_calls[0]) == ['miio1', 'norm1']
    assert sorted(client.delivered) == [('miio1', 5), ('norm1', 5)]


def _matrix_queue() -> dict:
    """The plan's matrix queue: a covered and an uncovered prop of one
    miio device plus a prop of one plain device."""
    return {
        f'miio1|{MIIO_SIID}|{MIIO_PIID}': {
            'did': 'miio1', 'siid': MIIO_SIID, 'piid': MIIO_PIID},
        'miio1|99|99': {'did': 'miio1', 'siid': 99, 'piid': 99},
        'norm1|2|1': {'did': 'norm1', 'siid': 2, 'piid': 1},
    }


@pytest.mark.asyncio
async def test_poll_cloud_first_gw_all_available() -> None:
    """Matrix 9.3, AUTO cloud_first behind a gateway with everything
    up: the LAN pre-pass reads the covered prop, the cloud takes the
    uncovered one and the plain device in one request."""
    client = _poll_client(ctrl_mode=CtrlMode.AUTO)
    client._refresh_props_list = _matrix_queue()
    cloud_calls: list[list[str]] = []

    async def _cloud(params):
        cloud_calls.append([p['did'] for p in params])
        return [{'did': p['did'], 'siid': p['siid'], 'piid': p['piid'],
                 'value': 5} for p in params]

    client._http = SimpleNamespace(get_props_async=AsyncMock(
        side_effect=_cloud))
    client._miot_lan = SimpleNamespace(
        init_done=True, get_prop_async=AsyncMock(return_value=42))
    _deliver(client)
    _real_managers(client)

    await client._MIoTClient__refresh_props_handler()

    assert sorted(cloud_calls[0]) == ['miio1', 'norm1']
    assert sorted(client.delivered) == [
        ('miio1', 5), ('miio1', 42), ('norm1', 5)]
    assert not client._refresh_props_list


@pytest.mark.asyncio
async def test_poll_cloud_first_no_gw_lan_offline_freezes_covered() -> None:
    """Matrix 9.3, AUTO cloud_first without a gateway and with the LAN
    gone: the covered prop freezes (the cloud skips it), the rest is
    served by the cloud."""
    client = _poll_client(ctrl_mode=CtrlMode.AUTO, gw=False)
    client._device_list_lan = {}
    client._refresh_props_list = _matrix_queue()
    cloud_calls: list[list[str]] = []

    async def _cloud(params):
        cloud_calls.append([p['did'] for p in params])
        return [{'did': p['did'], 'siid': p['siid'], 'piid': p['piid'],
                 'value': 5} for p in params]

    client._http = SimpleNamespace(get_props_async=AsyncMock(
        side_effect=_cloud))
    client._miot_lan = SimpleNamespace(
        init_done=True, get_prop_async=AsyncMock(return_value=42))
    _deliver(client)
    _real_managers(client)

    await client._MIoTClient__refresh_props_handler()

    assert sorted(cloud_calls[0]) == ['miio1', 'norm1']
    assert sorted(client.delivered) == [('miio1', 5), ('norm1', 5)]
    assert f'miio1|{MIIO_SIID}|{MIIO_PIID}' in client._refresh_props_list


@pytest.mark.asyncio
async def test_poll_local_gw_all_available() -> None:
    """Matrix 9.3, LOCAL behind a gateway with the LAN up: the covered
    prop goes LAN, the uncovered one and the plain device go gateway."""
    client = _poll_client(ctrl_mode=CtrlMode.LOCAL)
    client._refresh_props_list = _matrix_queue()
    gw_calls: list[tuple] = []

    async def _gw(did, siid, piid, **_kwargs):
        gw_calls.append((did, siid, piid))
        return 100

    client._mips_local = {'g': SimpleNamespace(
        get_prop_async=AsyncMock(side_effect=_gw))}
    client._miot_lan = SimpleNamespace(
        init_done=True, get_prop_async=AsyncMock(return_value=42))
    _deliver(client)
    _real_managers(client)

    await client._MIoTClient__refresh_props_handler()

    assert sorted(gw_calls) == [('miio1', 99, 99), ('norm1', 2, 1)]
    assert sorted(client.delivered) == [
        ('miio1', 42), ('miio1', 100), ('norm1', 100)]
    assert not client._refresh_props_list


@pytest.mark.asyncio
async def test_poll_local_gw_lan_offline_freezes_covered() -> None:
    """Matrix 9.3, LOCAL behind a gateway with the LAN gone: the
    covered prop freezes, the uncovered one and the plain device go
    gateway."""
    client = _poll_client(ctrl_mode=CtrlMode.LOCAL)
    client._device_list_lan = {}
    client._refresh_props_list = _matrix_queue()
    gw_calls: list[tuple] = []

    async def _gw(did, siid, piid, **_kwargs):
        gw_calls.append((did, siid, piid))
        return 100

    client._mips_local = {'g': SimpleNamespace(
        get_prop_async=AsyncMock(side_effect=_gw))}
    client._miot_lan = SimpleNamespace(
        init_done=True, get_prop_async=AsyncMock(return_value=42))
    _deliver(client)
    _real_managers(client)

    await client._MIoTClient__refresh_props_handler()

    assert sorted(gw_calls) == [('miio1', 99, 99), ('norm1', 2, 1)]
    assert sorted(client.delivered) == [('miio1', 100), ('norm1', 100)]
    assert f'miio1|{MIIO_SIID}|{MIIO_PIID}' in client._refresh_props_list


@pytest.mark.asyncio
async def test_poll_local_no_gw_lan_works() -> None:
    """Matrix 9.3, LOCAL without a gateway and with the LAN up: the
    covered prop and the plain device go LAN; the uncovered prop has
    no reader (no translation, no cloud in LOCAL) and stays queued
    (O1b)."""
    client = _poll_client(ctrl_mode=CtrlMode.LOCAL, gw=False)
    client._refresh_props_list = _matrix_queue()
    client._miot_lan = SimpleNamespace(
        init_done=True, get_prop_async=AsyncMock(return_value=42))
    _deliver(client)
    _real_managers(client)

    await client._MIoTClient__refresh_props_handler()

    assert sorted(client.delivered) == [('miio1', 42), ('norm1', 42)]
    assert 'miio1|99|99' in client._refresh_props_list


@pytest.mark.asyncio
async def test_poll_nothing_available_keeps_the_queue() -> None:
    """Matrix 9.3, nothing available: nobody reads anything, the whole
    queue stays for the fetch protection to clear."""
    client = _poll_client(ctrl_mode=CtrlMode.AUTO, gw=False)
    client._device_list_lan = {}
    client._device_list_cloud = {}
    # pylint: disable-next=protected-access
    client._network = SimpleNamespace(network_status=False)
    client._refresh_props_list = _matrix_queue()
    client._miot_lan = SimpleNamespace(
        init_done=True, get_prop_async=AsyncMock(return_value=42))
    _deliver(client)
    _real_managers(client)

    await client._MIoTClient__refresh_props_handler()

    assert not client.delivered
    assert client._refresh_props_list == _matrix_queue()
