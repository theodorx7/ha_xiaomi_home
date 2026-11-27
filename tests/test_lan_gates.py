"""LAN transport gates and the LAN-vote predicate.

Covers the gate changes on live code (no network):
- gate 1: init is skipped behind a central hub gateway only when there
  is no MIIO-spec device to serve;
- gate 2: a gateway appearing no longer deinits the LAN layer while a
  MIIO device needs it, and stands the layer down when none does;
- gate 3: cloud local_ip is passed into the LAN layer for CN too;
- the client vote predicate: mode in {AUTO, LOCAL} AND (no gateway OR
  MIIO devices present OR gateway ignored);
- the MIIO membership of the device cache (models in MIIO_SPECS),
  read on the fly by the vote.
"""
# pylint: disable=protected-access
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from custom_components.xiaomi_home.miot import miot_lan as miot_lan_module
from custom_components.xiaomi_home.miot.miot_client import CtrlMode, MIoTClient
from custom_components.xiaomi_home.miot.miot_lan import MIoTLan

DID_MIIO = '101'
DID_MIIO_2 = '102'
DID_SPEC = '103'
MIIO_MODEL = next(iter(miot_lan_module.MIIO_SPECS))
SPEC_MODEL = 'xiaomi.gateway.hub1'
KEY = 'uid-cn'


def _lan(*, gateway_seen: bool, ignore_mips_service: bool = False) -> MIoTLan:
    """A MIoTLan around a fake network/mDNS, with the real init/deinit
    logic but no sockets or threads: tests patch init_async/deinit_async
    on the class."""
    lan = object.__new__(MIoTLan)
    lan._main_loop = None  # unused while init_async is short-circuited
    lan._net_ifs = {'eth0'}
    lan._network = SimpleNamespace(network_info={'eth0': {}})
    lan._mips_service = SimpleNamespace(
        get_services=lambda group_id=None: (
            {'g1': {}} if gateway_seen else {}))
    lan._enable_subscribe = False
    lan._virtual_did = '123'
    lan._lan_devices = {}
    lan._available_net_ifs = {'eth0'}
    lan._broadcast_socks = {}
    lan._local_port = None
    lan._scan_timer = None
    lan._last_scan_interval = None
    lan._msg_id_counter = 1
    lan._pending_requests = {}
    lan._device_msg_matcher = None
    lan._device_state_sub_map = {}
    lan._reply_msg_buffer = {}
    lan._lan_state_sub_map = {}
    lan._lan_ctrl_vote_map = {}
    lan._miio_required_map = {}
    lan._miio_required = False
    lan._profile_models = {}
    lan._init_lock = None  # not used: init path is intercepted by tests
    lan._init_done = False
    lan._ignore_mips_service = ignore_mips_service
    lan._internal_loop = None
    lan._thread = None
    return lan


async def _deinit_noop(self) -> None:
    self._init_done = False


def _client(*, ctrl_mode=CtrlMode.AUTO, cloud_server='cn',
            miio_dids=None, gateway_seen=False,
            ignore_mips_service=False) -> MIoTClient:
    """A MIoTClient wired to a recording MIoTLan double."""
    with patch.object(MIoTClient, '__init__', lambda self, *a, **k: None):
        client = MIoTClient()  # pylint: disable=no-value-for-parameter
    client._ctrl_mode = ctrl_mode
    client._uid = 'uid'
    client._cloud_server = cloud_server
    client._device_list_cache = {
        did: {'model': MIIO_MODEL} for did in (miio_dids or [])}
    client._mips_service = SimpleNamespace(
        get_services=lambda group_id=None: (
            {'g1': {}} if gateway_seen else {}))
    client._miot_lan = _RecordingLan(ignore_mips_service)
    return client


class _RecordingLan:
    """MIoTLan double: records votes and miio-required reports."""

    def __init__(self, ignore_mips_service: bool = False) -> None:
        self._ignore_mips_service = ignore_mips_service
        self.votes: list[tuple[str, bool]] = []
        self.miio_required: list[tuple[str, bool]] = []

    async def vote_for_lan_ctrl_async(self, key: str, vote: bool) -> None:
        self.votes.append((key, vote))

    async def set_miio_required_async(self, key: str, required: bool) -> None:
        self.miio_required.append((key, required))


# ---------------------------------------------------------------------------
# Gate 1 + 2: MIoTLan init/deinit behind a central hub gateway
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_init_skips_behind_gateway_without_miio() -> None:
    """Gate 1: no MIIO devices -> the layer stays down (today's CN
    behaviour, zero regression for gateway-only installs)."""
    lan = _lan(gateway_seen=True)
    lan._lan_ctrl_vote_map = {KEY: True}
    async def fake_init(self) -> None:
        self._init_done = True
    with patch.object(MIoTLan, 'init_async', fake_init):
        await lan._MIoTLan__on_mips_service_change(
            'g1', None, {'addresses': ['10.0.0.1'], 'port': 8883})
    assert lan._init_done is False


@pytest.mark.asyncio
async def test_init_async_gate_itself_skips_without_miio() -> None:
    """Gate 1, the init_async entry itself: with a vote but no MIIO
    devices the real init_async must stop before starting the thread;
    with MIIO devices it must get past the gate and start it."""

    # Without MIIO: the gate stops the init
    lan = _lan(gateway_seen=True)
    lan._lan_ctrl_vote_map = {KEY: True}
    lan._miio_required = False
    lan._init_lock = asyncio.Lock()
    started = []

    class _FakeThread:
        def __init__(self, target=None):
            self.target = target
        name = ''
        daemon = False
        def start(self):
            started.append(self.target)

    with patch.object(threading, 'Thread', _FakeThread):
        await MIoTLan.init_async(lan)  # type: ignore[arg-type]
    assert lan._init_done is False
    assert not started

    # With MIIO devices: the gate passes, the init runs through
    lan2 = _lan(gateway_seen=True)
    lan2._lan_ctrl_vote_map = {KEY: True}
    lan2._miio_required = True
    lan2._miio_required_map = {KEY: True}
    lan2._init_lock = asyncio.Lock()

    class _FakeLoop:
        def call_later(self, *_args, **_kwargs):
            return None
        def run_forever(self):
            return None
        def close(self):
            return None
        def call_soon_threadsafe(self, *_args, **_kwargs):
            return None
        def add_reader(self, *_args, **_kwargs):
            return None

    with patch.object(threading, 'Thread', _FakeThread), \
            patch.object(asyncio, 'new_event_loop', _FakeLoop):
        # The executor call for profile models: main_loop is None in the
        # fixture, so run through it cannot happen; give a fake main loop
        class _FakeMainLoop:
            def run_in_executor(self, *_args, **_kwargs):
                async def _inner(*_args, **_kwargs):
                    return {}
                return _inner()
        lan2._main_loop = _FakeMainLoop()
        await MIoTLan.init_async(lan2)  # type: ignore[arg-type]
    assert lan2._init_done is True
    assert started


@pytest.mark.asyncio
async def test_init_runs_behind_gateway_with_miio() -> None:
    """Gate 1 lifted: a MIIO device present -> the layer starts even
    though the gateway is visible."""
    lan = _lan(gateway_seen=True)
    lan._lan_ctrl_vote_map = {KEY: True}
    lan._miio_required = True
    lan._miio_required_map = {KEY: True}
    inited = []

    async def fake_init(self) -> None:
        inited.append(True)
        self._init_done = True

    with patch.object(MIoTLan, 'init_async', fake_init):
        await lan._MIoTLan__on_mips_service_change(
            'g1', None, {'addresses': ['10.0.0.1'], 'port': 8883})
    assert inited == [True]
    assert lan._init_done is True


@pytest.mark.asyncio
async def test_gateway_appearing_keeps_layer_with_miio() -> None:
    """Gate 2 removed: the gateway appearing does NOT deinit the layer
    while a MIIO device needs it."""
    lan = _lan(gateway_seen=False)
    lan._lan_ctrl_vote_map = {KEY: True}
    lan._miio_required = True
    lan._miio_required_map = {KEY: True}
    lan._init_done = True  # layer already up (no gateway yet)
    lan._internal_loop = None
    lan._thread = None
    deinited = []

    # pylint: disable-next=unused-argument
    async def fake_deinit(self) -> None:
        deinited.append(True)
    async def fake_init(self) -> None:
        self._init_done = True

    with patch.object(MIoTLan, 'deinit_async', fake_deinit), \
            patch.object(MIoTLan, 'init_async', fake_init):
        # The gateway appears (its mDNS service is now visible)
        lan._mips_service = SimpleNamespace(
            get_services=lambda group_id=None: {'g1': {}})
        await lan._MIoTLan__on_mips_service_change(
            'g1', None, {'addresses': ['10.0.0.1'], 'port': 8883})
    assert not deinited
    assert lan._init_done is True


@pytest.mark.asyncio
async def test_gateway_appearing_stops_layer_without_miio() -> None:
    """Without MIIO devices the layer stands down when the gateway
    appears (the documented 'gateway only' behaviour)."""
    lan = _lan(gateway_seen=True)
    lan._lan_ctrl_vote_map = {KEY: True}
    lan._miio_required = False
    lan._miio_required_map = {}
    lan._init_done = True  # came up before the gateway was seen
    deinited = []

    async def fake_deinit(self) -> None:
        deinited.append(True)
        self._init_done = False

    with patch.object(MIoTLan, 'deinit_async', fake_deinit):
        await lan._MIoTLan__on_mips_service_change(
            'g1', None, {'addresses': ['10.0.0.1'], 'port': 8883})
    assert deinited == [True]


@pytest.mark.asyncio
async def test_set_miio_required_or_across_clients() -> None:
    """Two accounts: the effective flag is the OR of their reports; the
    last one standing releases it."""
    lan = _lan(gateway_seen=True)
    lan._lan_ctrl_vote_map = {'a': True, 'b': True}
    lan._miio_required = True  # first report already flipped it
    lan._miio_required_map = {'a': True}
    lan._init_done = True
    # client b reports MIIO devices too: no transition
    deinited = []

    # pylint: disable-next=unused-argument
    async def fake_deinit(self) -> None:
        deinited.append(True)
    with patch.object(MIoTLan, 'deinit_async', fake_deinit):
        await lan.set_miio_required_async('b', True)
        assert lan._miio_required is True
        assert not deinited
        # a releases, b still needs it: stays up
        await lan.set_miio_required_async('a', False)
        assert lan._miio_required is True
        assert not deinited
        # b releases too: nothing to serve behind the gateway
        await lan.set_miio_required_async('b', False)
        assert lan._miio_required is False
        assert deinited == [True]


@pytest.mark.asyncio
async def test_set_miio_required_inits_when_down() -> None:
    """A MIIO device appearing behind the gateway raises the layer."""
    lan = _lan(gateway_seen=True)
    lan._lan_ctrl_vote_map = {KEY: True}
    lan._init_done = False
    inited = []

    async def fake_init(self) -> None:
        inited.append(True)
        self._init_done = True

    with patch.object(MIoTLan, 'init_async', fake_init):
        await lan.set_miio_required_async(KEY, True)
    assert inited == [True]
    assert lan._miio_required is True


# ---------------------------------------------------------------------------
# The client vote predicate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize('ctrl_mode', [CtrlMode.AUTO, CtrlMode.LOCAL])
async def test_vote_true_without_gateway(ctrl_mode) -> None:
    """No gateway -> the layer is voted up, as before."""
    client = _client(ctrl_mode=ctrl_mode, gateway_seen=False)
    await client._MIoTClient__vote_for_lan_ctrl_async()
    assert client._miot_lan.votes == [(KEY, True)]
    assert client._miot_lan.miio_required == [(KEY, False)]


@pytest.mark.asyncio
@pytest.mark.parametrize('ctrl_mode', [CtrlMode.AUTO, CtrlMode.LOCAL])
async def test_vote_true_behind_gateway_with_miio(ctrl_mode) -> None:
    """Gateway visible + MIIO device -> voted up and required."""
    client = _client(
        ctrl_mode=ctrl_mode, gateway_seen=True, miio_dids=[DID_MIIO])
    await client._MIoTClient__vote_for_lan_ctrl_async()
    assert client._miot_lan.votes == [(KEY, True)]
    assert client._miot_lan.miio_required == [(KEY, True)]


@pytest.mark.asyncio
async def test_vote_false_behind_gateway_without_miio() -> None:
    """Gateway visible, no MIIO device -> voted down (today's CN
    behaviour); miio-required report is False."""
    client = _client(ctrl_mode=CtrlMode.AUTO, gateway_seen=True)
    await client._MIoTClient__vote_for_lan_ctrl_async()
    assert client._miot_lan.votes == [(KEY, False)]
    assert client._miot_lan.miio_required == [(KEY, False)]


@pytest.mark.asyncio
async def test_vote_cloud_mode_never_votes() -> None:
    """CLOUD mode: no vote at all, and any miio-required report is
    retracted."""
    client = _client(ctrl_mode=CtrlMode.CLOUD, gateway_seen=False,
                     miio_dids=[DID_MIIO])
    await client._MIoTClient__vote_for_lan_ctrl_async()
    assert client._miot_lan.votes == [(KEY, False)]
    assert client._miot_lan.miio_required == [(KEY, False)]


@pytest.mark.asyncio
async def test_vote_true_when_gateway_ignored() -> None:
    """Non-CN region (ignore_mips_service): always voted up in
    AUTO/LOCAL."""
    client = _client(ctrl_mode=CtrlMode.AUTO, gateway_seen=True,
                     ignore_mips_service=True)
    await client._MIoTClient__vote_for_lan_ctrl_async()
    assert client._miot_lan.votes == [(KEY, True)]
    # gateway ignored -> no miio-required report needed
    assert client._miot_lan.miio_required == [(KEY, False)]


@pytest.mark.asyncio
async def test_vote_non_cn_cloud_server() -> None:
    """A non-CN cloud server: gateway_seen must not gate the vote even
    if some mDNS service is around (the ignore flag owns that)."""
    client = _client(ctrl_mode=CtrlMode.AUTO, cloud_server='de',
                     gateway_seen=True, ignore_mips_service=True)
    await client._MIoTClient__vote_for_lan_ctrl_async()
    assert client._miot_lan.votes == [('uid-de', True)]


# ---------------------------------------------------------------------------
# MIIO membership of the device cache
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_miio_membership_reads_the_device_cache() -> None:
    """The vote classifies the device cache directly: models in
    MIIO_SPECS count, spec devices and unknown models do not. No stored
    set - the cache is the single source of truth."""
    client = _client(gateway_seen=True, ctrl_mode=CtrlMode.AUTO)
    client._device_list_cache = {
        DID_MIIO: {'model': MIIO_MODEL},
        DID_MIIO_2: {'model': MIIO_MODEL},
        DID_SPEC: {'model': SPEC_MODEL},
        '104': {'model': None},
    }
    await client._MIoTClient__vote_for_lan_ctrl_async()
    assert client._miot_lan.votes == [(KEY, True)]
    assert client._miot_lan.miio_required == [(KEY, True)]
    # only spec and unknown models left: nothing to serve behind the hub
    client._miot_lan.votes.clear()
    client._miot_lan.miio_required.clear()
    client._device_list_cache = {
        DID_SPEC: {'model': SPEC_MODEL}, '104': {'model': None}}
    await client._MIoTClient__vote_for_lan_ctrl_async()
    assert client._miot_lan.votes == [(KEY, False)]
    assert client._miot_lan.miio_required == [(KEY, False)]


# ---------------------------------------------------------------------------
# Gate 3: cloud local_ip reaches the LAN layer for CN
# ---------------------------------------------------------------------------

def _cloud_refresh_client() -> MIoTClient:
    """A client shaped for __refresh_cloud_devices_async's update block."""
    with patch.object(MIoTClient, '__init__', lambda self, *a, **k: None):
        client = MIoTClient()  # pylint: disable=no-value-for-parameter
    client._uid = 'uid'
    client._cloud_server = 'cn'  # the region that used to null the IP
    client._ctrl_mode = CtrlMode.AUTO
    client._entry_data = {'home_selected': {}}
    client._miot_lan = _RecordingLan()
    client._miot_lan.update_devices = lambda devices: None
    client._miot_lan.init_done = True
    client._mips_service = SimpleNamespace(
        get_services=lambda group_id=None: {'g1': {}})
    client._device_list_cache = {
        DID_MIIO: {
            'token': 'a' * 32, 'model': MIIO_MODEL,
            'connect_type': 0, 'local_ip': '192.168.1.10'},
        DID_SPEC: {
            'token': 'b' * 32, 'model': SPEC_MODEL,
            'connect_type': 0, 'local_ip': '192.168.1.11'},
        '104': {  # no token: not LAN-capable, skipped
            'model': SPEC_MODEL, 'connect_type': 0},
    }
    client._device_list_cloud = {}
    client._refresh_cloud_devices_timer = None
    client._http = SimpleNamespace(get_devices_async=AsyncMock(
        return_value={'devices': {DID_MIIO: {'online': True}}}))
    client._storage = SimpleNamespace(save_async=AsyncMock(
        return_value=True))
    client._i18n = SimpleNamespace(translate_str=lambda key: key)
    client._persistence_notify = lambda *args: None
    client._show_devices_changed_notify_timer = None
    client._network = SimpleNamespace(network_status=True)
    client._display_devs_notify = []
    client._sub_source_list = {}
    client._sub_tree = None
    client._sub_device_state = {}
    client._device_list_gateway = {}
    client._device_list_lan = {}
    client._refresh_props_list = {}
    return client


@pytest.mark.asyncio
async def test_refresh_cloud_devices_passes_ip_for_cn() -> None:
    """Gate 3: the CN region no longer nulls local_ip; the LAN layer
    receives real IPs and can unicast-probe them. The cache-merging
    part of the refresh is patched out: this test pins the update
    block's payload and the re-run vote."""
    client = _cloud_refresh_client()
    passed = {}
    deleted: list[str] = []
    # pylint: disable-next=unnecessary-lambda
    client._miot_lan.update_devices = lambda devices: passed.update(devices)
    # pylint: disable-next=unnecessary-lambda
    client._miot_lan.delete_devices = lambda devices: deleted.extend(
        devices)
    cloud_devices = {DID_MIIO: {'online': True}}

    async def fake_update(self, cloud_list, filter_dids=None):  # pylint: disable=unused-argument
        client._device_list_cloud.update(cloud_list)
        return None
    with patch.object(
            MIoTClient, '_MIoTClient__update_devices_from_cloud_async',
            fake_update):
        client._http.get_devices_async = AsyncMock(
            return_value={'devices': cloud_devices})
        refresh = client._MIoTClient__refresh_cloud_devices_async
        await refresh()
    assert passed[DID_MIIO]['ip'] == '192.168.1.10'
    # specv2 devices stay fully owned by the gateway behind it
    assert DID_SPEC not in passed
    assert deleted == [DID_SPEC]
    assert '104' not in passed  # no token: not LAN-capable
    # the vote re-ran after the device list update
    assert client._miot_lan.votes == [(KEY, True)]
    assert client._miot_lan.miio_required == [(KEY, True)]


@pytest.mark.asyncio
async def test_sync_clears_the_stale_lan_online_entry() -> None:
    """V14: a device found over the LAN before the gateway appeared is
    deregistered from the layer, and its stale 'online' entry in the
    client's LAN list is cleared through the usual offline path."""
    client = _cloud_refresh_client()
    client._device_list_lan = {
        DID_SPEC: {'online': True, 'push_available': True}}
    # pylint: disable-next=unnecessary-lambda
    client._miot_lan.delete_devices = lambda devices: None
    with patch.object(
            MIoTClient, '_MIoTClient__update_devices_from_cloud_async',
            AsyncMock(return_value=None)):
        await client._MIoTClient__refresh_cloud_devices_async()
    assert client._device_list_lan[DID_SPEC]['online'] is False
    assert client._device_list_lan[DID_SPEC]['push_available'] is False


@pytest.mark.asyncio
async def test_lan_state_change_passes_ip_for_cn() -> None:
    """Gate 3, second call site: __on_miot_lan_state_change(True) feeds
    the LAN layer with IPs for CN as well."""
    with patch.object(MIoTClient, '__init__', lambda self, *a, **k: None):
        client = MIoTClient()  # pylint: disable=no-value-for-parameter
    client._uid = 'uid'
    client._cloud_server = 'cn'
    client._ctrl_mode = CtrlMode.AUTO
    client._mips_service = SimpleNamespace(
        get_services=lambda group_id=None: {})
    client._miot_lan = _RecordingLan()
    client._miot_lan.sub_device_state = lambda **kwargs: None
    client._miot_lan.get_dev_list_async = AsyncMock(return_value={})
    client._device_list_lan = {}
    client._sub_source_list = {}
    client._sub_tree = None
    client._sub_device_state = {}
    client._device_list_cloud = {}
    client._device_list_gateway = {}
    client._device_list_cache = {
        DID_MIIO: {
            'token': 'a' * 32, 'model': MIIO_MODEL,
            'connect_type': 0, 'local_ip': '192.168.1.10'}}
    passed = {}
    # pylint: disable-next=unnecessary-lambda
    client._miot_lan.update_devices = lambda devices: passed.update(devices)
    await client._MIoTClient__on_miot_lan_state_change(True)
    assert passed[DID_MIIO]['ip'] == '192.168.1.10'
