"""
AstraCore Neo — Connectivity testbench.

Coverage:
  - CAN-FD: frame creation, TX/RX queues, priority ordering, error counters,
             bus state transitions, BUS_OFF recovery
  - Ethernet: frame validation, link state, TX/RX, MAC filtering, broadcast,
               buffer overflow
  - PCIe: link training, BAR management, MMIO read/write, TLP processing,
           L1 power state
  - V2X: channel management, broadcast TX, RX filtering (channel + RSSI),
          queue overflow
  - ConnectivityManager: init, double-init guard, link_status, any_link_up
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from connectivity import (
    CANFDController, CANFrame, CANIDFormat, CANBusState,
    EthernetController, EthernetFrame, EthernetLinkState, BROADCAST_MAC,
    PCIeController, PCIeBAR, PCIeLinkState, TLP, TLPType,
    V2XController, V2XMessage, V2XMessageType, CCH_CHANNEL, DSRC_CHANNELS,
    ConnectivityManager,
    ConnectivityBaseError, CANError, EthernetError, PCIeError, V2XError,
)


# ===========================================================================
# CAN-FD
# ===========================================================================

def _std_frame(can_id: int, data: bytes = b"\x00") -> CANFrame:
    return CANFrame(can_id=can_id, id_format=CANIDFormat.STANDARD, data=data)

def _ext_frame(can_id: int, data: bytes = b"\x00") -> CANFrame:
    return CANFrame(can_id=can_id, id_format=CANIDFormat.EXTENDED, data=data)

def _fd_frame(can_id: int, data: bytes) -> CANFrame:
    return CANFrame(can_id=can_id, id_format=CANIDFormat.STANDARD, data=data, is_fd=True)


class TestCANFrame:

    def test_standard_frame_dlc(self):
        f = _std_frame(0x100, b"hello!!!")
        assert f.dlc == 8

    def test_extended_frame_id(self):
        f = _ext_frame(0x1FFFFFFF, b"\xAB")
        assert f.can_id == 0x1FFFFFFF

    def test_fd_frame_dlc_12_bytes(self):
        f = _fd_frame(0x100, bytes(12))
        assert f.dlc == 9

    def test_fd_frame_dlc_64_bytes(self):
        f = _fd_frame(0x100, bytes(64))
        assert f.dlc == 15

    def test_standard_id_overflow_raises(self):
        with pytest.raises(CANError):
            CANFrame(can_id=0x800, id_format=CANIDFormat.STANDARD, data=b"\x00")

    def test_classic_payload_overflow_raises(self):
        with pytest.raises(CANError):
            _std_frame(0x100, bytes(9))

    def test_fd_payload_overflow_raises(self):
        with pytest.raises(CANError):
            _fd_frame(0x100, bytes(65))


class TestCANFDTxRx:

    def test_send_and_transmit(self):
        ctrl = CANFDController(node_id=1)
        ctrl.send(_std_frame(0x100, b"data"))
        f = ctrl.transmit_next()
        assert f is not None
        assert f.can_id == 0x100

    def test_tx_count_increments(self):
        ctrl = CANFDController(node_id=1)
        ctrl.send(_std_frame(0x100))
        assert ctrl.tx_count == 1

    def test_priority_ordering(self):
        ctrl = CANFDController(node_id=1)
        ctrl.send(_std_frame(0x300))
        ctrl.send(_std_frame(0x100))
        ctrl.send(_std_frame(0x200))
        ids = [ctrl.transmit_next().can_id for _ in range(3)]
        assert ids == [0x100, 0x200, 0x300]

    def test_transmit_next_empty_returns_none(self):
        ctrl = CANFDController(node_id=1)
        assert ctrl.transmit_next() is None

    def test_receive_and_read(self):
        ctrl = CANFDController(node_id=1)
        ctrl.receive(_std_frame(0x200, b"hello"))
        f = ctrl.read()
        assert f is not None
        assert f.can_id == 0x200

    def test_read_empty_returns_none(self):
        ctrl = CANFDController(node_id=1)
        assert ctrl.read() is None

    def test_rx_fifo_order(self):
        ctrl = CANFDController(node_id=1)
        for i in range(3):
            ctrl.receive(_std_frame(i + 1))
        for i in range(3):
            assert ctrl.read().can_id == i + 1

    def test_rx_available(self):
        ctrl = CANFDController(node_id=1)
        ctrl.receive(_std_frame(0x100))
        ctrl.receive(_std_frame(0x200))
        assert ctrl.rx_available() == 2


class TestCANFDBusState:

    def test_initial_state_error_active(self):
        ctrl = CANFDController(node_id=1)
        assert ctrl.bus_state == CANBusState.ERROR_ACTIVE

    def test_tx_errors_escalate_to_passive(self):
        ctrl = CANFDController(node_id=1)
        # Each tx error = +8 TEC; need 128 → 16 errors
        for _ in range(16):
            ctrl.inject_tx_error()
        assert ctrl.bus_state == CANBusState.ERROR_PASSIVE
        assert ctrl.tec >= 128

    def test_tx_errors_escalate_to_bus_off(self):
        ctrl = CANFDController(node_id=1)
        # TEC > 255 = 32 × 8 = 256
        for _ in range(33):
            ctrl.inject_tx_error()
        assert ctrl.bus_state == CANBusState.BUS_OFF

    def test_bus_off_blocks_send(self):
        ctrl = CANFDController(node_id=1)
        for _ in range(33):
            ctrl.inject_tx_error()
        with pytest.raises(CANError):
            ctrl.send(_std_frame(0x100))

    def test_bus_off_blocks_transmit(self):
        ctrl = CANFDController(node_id=1)
        ctrl.send(_std_frame(0x100))
        for _ in range(33):
            ctrl.inject_tx_error()
        with pytest.raises(CANError):
            ctrl.transmit_next()

    def test_bus_off_recovery(self):
        ctrl = CANFDController(node_id=1)
        for _ in range(33):
            ctrl.inject_tx_error()
        assert ctrl.bus_state == CANBusState.BUS_OFF
        ctrl.bus_off_recovery()
        assert ctrl.bus_state == CANBusState.ERROR_ACTIVE
        assert ctrl.tec == 0
        assert ctrl.rec == 0

    def test_recovery_from_non_bus_off_raises(self):
        ctrl = CANFDController(node_id=1)
        with pytest.raises(CANError):
            ctrl.bus_off_recovery()

    def test_rx_error_increments_rec(self):
        ctrl = CANFDController(node_id=1)
        for _ in range(5):
            ctrl.inject_rx_error()
        assert ctrl.rec == 5


# ===========================================================================
# Ethernet
# ===========================================================================

def _mac(hex_str: str) -> bytes:
    return bytes.fromhex(hex_str.replace(":", ""))

MY_MAC  = _mac("aa:bb:cc:dd:ee:ff")
OTHER_MAC = _mac("22:22:33:44:55:66")  # first byte even = unicast, not multicast


def _frame(dest: bytes, payload: bytes = b"\x00" * 46) -> EthernetFrame:
    return EthernetFrame(dest_mac=dest, src_mac=MY_MAC, ethertype=0x0800, payload=payload)


class TestEthernetFrame:

    def test_valid_frame_created(self):
        f = _frame(MY_MAC)
        assert f.ethertype == 0x0800

    def test_invalid_dest_mac_length(self):
        with pytest.raises(EthernetError):
            EthernetFrame(dest_mac=b"\x00" * 5, src_mac=MY_MAC, ethertype=0x0800, payload=b"\x00")

    def test_payload_too_large(self):
        with pytest.raises(EthernetError):
            _frame(MY_MAC, payload=b"\x00" * 1501)

    def test_invalid_ethertype(self):
        with pytest.raises(EthernetError):
            EthernetFrame(dest_mac=MY_MAC, src_mac=MY_MAC, ethertype=0x1FFFF, payload=b"\x00")


class TestEthernetLink:

    def test_initial_state_down(self):
        ctrl = EthernetController(MY_MAC)
        assert ctrl.link_state == EthernetLinkState.DOWN

    def test_link_up(self):
        ctrl = EthernetController(MY_MAC)
        ctrl.link_up()
        assert ctrl.link_state == EthernetLinkState.UP

    def test_send_fails_link_down(self):
        ctrl = EthernetController(MY_MAC)
        with pytest.raises(EthernetError):
            ctrl.send(_frame(OTHER_MAC))

    def test_send_succeeds_link_up(self):
        ctrl = EthernetController(MY_MAC)
        ctrl.link_up()
        ctrl.send(_frame(OTHER_MAC))
        assert ctrl.tx_count == 1


class TestEthernetRx:

    def test_receive_unicast(self):
        ctrl = EthernetController(MY_MAC)
        assert ctrl.receive(_frame(MY_MAC)) is True
        assert ctrl.rx_available() == 1

    def test_receive_broadcast(self):
        ctrl = EthernetController(MY_MAC)
        assert ctrl.receive(_frame(BROADCAST_MAC)) is True

    def test_receive_wrong_mac_filtered(self):
        ctrl = EthernetController(MY_MAC)
        assert ctrl.receive(_frame(OTHER_MAC)) is False
        assert ctrl.rx_available() == 0

    def test_read_returns_frame(self):
        ctrl = EthernetController(MY_MAC)
        ctrl.receive(_frame(MY_MAC))
        f = ctrl.read()
        assert f is not None
        assert f.dest_mac == MY_MAC

    def test_read_empty_returns_none(self):
        ctrl = EthernetController(MY_MAC)
        assert ctrl.read() is None

    def test_rx_buffer_overflow_increments_dropped(self):
        ctrl = EthernetController(MY_MAC, rx_buffer_size=2)
        for _ in range(3):
            ctrl.receive(_frame(MY_MAC))
        assert ctrl.dropped == 1

    def test_multicast_accepted(self):
        # LSB of first byte set = multicast
        mcast_mac = bytes([0x01, 0x00, 0x5E, 0x00, 0x00, 0x01])
        ctrl = EthernetController(MY_MAC)
        assert ctrl.receive(_frame(mcast_mac)) is True


# ===========================================================================
# PCIe
# ===========================================================================

class TestPCIeLinkTraining:

    def test_initial_state_detect(self):
        ep = PCIeController(device_id=0x1000)
        assert ep.link_state == PCIeLinkState.DETECT

    def test_train_link_reaches_l0(self):
        ep = PCIeController(device_id=0x1000)
        ep.train_link()
        assert ep.link_state == PCIeLinkState.L0

    def test_l1_power_state(self):
        ep = PCIeController(device_id=0x1000)
        ep.train_link()
        ep.enter_l1()
        assert ep.link_state == PCIeLinkState.L1
        ep.exit_l1()
        assert ep.link_state == PCIeLinkState.L0

    def test_enter_l1_from_detect_raises(self):
        ep = PCIeController(device_id=0x1000)
        with pytest.raises(PCIeError):
            ep.enter_l1()

    def test_invalid_lane_count(self):
        with pytest.raises(PCIeError):
            PCIeController(device_id=0x1000, num_lanes=3)


class TestPCIeBAR:

    def _ep(self):
        ep = PCIeController(device_id=0x1000, num_lanes=4)
        ep.train_link()
        return ep

    def test_add_bar(self):
        ep = self._ep()
        bar = ep.add_bar(bar_index=0, base_addr=0x10000000, size=4096)
        assert bar.size == 4096

    def test_mmio_write_and_read(self):
        ep = self._ep()
        ep.add_bar(bar_index=0, base_addr=0x10000000, size=4096)
        ep.mmio_write(0x10000000, b"\xDE\xAD\xBE\xEF")
        data = ep.mmio_read(0x10000000, 4)
        assert data == b"\xDE\xAD\xBE\xEF"

    def test_mmio_read_at_offset(self):
        ep = self._ep()
        ep.add_bar(bar_index=0, base_addr=0x20000000, size=256)
        ep.mmio_write(0x20000010, b"\xAA\xBB")
        assert ep.mmio_read(0x20000010, 2) == b"\xAA\xBB"

    def test_mmio_out_of_bar_range_raises(self):
        ep = self._ep()
        ep.add_bar(bar_index=0, base_addr=0x10000000, size=256)
        with pytest.raises(PCIeError):
            ep.mmio_read(0x10000100, 1)

    def test_mmio_requires_l0(self):
        ep = PCIeController(device_id=0x1000)
        ep.add_bar(bar_index=0, base_addr=0x10000000, size=256)
        with pytest.raises(PCIeError):
            ep.mmio_read(0x10000000, 1)

    def test_duplicate_bar_raises(self):
        ep = self._ep()
        ep.add_bar(bar_index=0, base_addr=0x10000000, size=256)
        with pytest.raises(PCIeError):
            ep.add_bar(bar_index=0, base_addr=0x20000000, size=256)

    def test_bar_size_must_be_power_of_two(self):
        with pytest.raises(PCIeError):
            PCIeBAR(bar_index=0, base_addr=0x10000000, size=300)


class TestPCIeTLP:

    def _ep(self):
        ep = PCIeController(device_id=0x1000, num_lanes=4)
        ep.train_link()
        ep.add_bar(bar_index=0, base_addr=0x10000000, size=4096)
        return ep

    def test_send_tlp(self):
        ep = self._ep()
        tlp = TLP(
            tlp_type=TLPType.MEM_WRITE, requester_id=0x0100,
            tag=1, address=0x10000000, data=b"\x01\x02\x03\x04", length_dw=1,
        )
        ep.send_tlp(tlp)
        assert ep.tx_count == 1

    def test_receive_mem_write_tlp(self):
        ep = self._ep()
        tlp = TLP(
            tlp_type=TLPType.MEM_WRITE, requester_id=0x0100,
            tag=1, address=0x10000004, data=b"\xCA\xFE", length_dw=1,
        )
        ep.receive_tlp(tlp)
        assert ep.mmio_read(0x10000004, 2) == b"\xCA\xFE"

    def test_receive_mem_read_returns_completion(self):
        ep = self._ep()
        ep.mmio_write(0x10000000, b"\xDE\xAD\xBE\xEF")
        tlp = TLP(
            tlp_type=TLPType.MEM_READ, requester_id=0x0100,
            tag=5, address=0x10000000, data=b"", length_dw=1,
        )
        cpl = ep.receive_tlp(tlp)
        assert cpl is not None
        assert cpl.tlp_type == TLPType.COMPLETION_DATA
        assert cpl.data == b"\xDE\xAD\xBE\xEF"
        assert cpl.tag == 5


# ===========================================================================
# V2X
# ===========================================================================

def _bsm(sender: int = 1, channel: int = CCH_CHANNEL, rssi: float = -70.0) -> V2XMessage:
    return V2XMessage(
        msg_type=V2XMessageType.BSM,
        sender_id=sender,
        channel=channel,
        payload=b"\x00" * 38,
        rssi_dbm=rssi,
    )


class TestV2XController:

    def test_broadcast_tx(self):
        ctrl = V2XController(node_id=1, channel=CCH_CHANNEL)
        ctrl.broadcast(_bsm())
        assert ctrl.tx_count == 1

    def test_broadcast_wrong_channel_raises(self):
        ctrl = V2XController(node_id=1, channel=CCH_CHANNEL)
        with pytest.raises(V2XError):
            ctrl.broadcast(_bsm(channel=174))

    def test_receive_same_channel(self):
        ctrl = V2XController(node_id=2, channel=CCH_CHANNEL)
        assert ctrl.receive(_bsm()) is True
        assert ctrl.rx_available() == 1

    def test_receive_wrong_channel_filtered(self):
        ctrl = V2XController(node_id=2, channel=CCH_CHANNEL)
        assert ctrl.receive(_bsm(channel=174)) is False

    def test_receive_weak_rssi_filtered(self):
        ctrl = V2XController(node_id=2, channel=CCH_CHANNEL, rssi_threshold_dbm=-80.0)
        assert ctrl.receive(_bsm(rssi=-95.0)) is False
        assert ctrl.filtered_rssi == 1

    def test_read_fifo_order(self):
        ctrl = V2XController(node_id=2, channel=CCH_CHANNEL)
        for i in range(3):
            ctrl.receive(_bsm(sender=i + 1))
        for i in range(3):
            assert ctrl.read().sender_id == i + 1

    def test_rx_queue_overflow_dropped(self):
        ctrl = V2XController(node_id=2, channel=CCH_CHANNEL, rx_queue_size=2)
        for _ in range(3):
            ctrl.receive(_bsm())
        assert ctrl.dropped == 1

    def test_set_channel(self):
        ctrl = V2XController(node_id=1, channel=CCH_CHANNEL)
        ctrl.set_channel(174)
        assert ctrl.channel == 174

    def test_set_invalid_channel_raises(self):
        ctrl = V2XController(node_id=1, channel=CCH_CHANNEL)
        with pytest.raises(V2XError):
            ctrl.set_channel(100)

    def test_invalid_channel_in_constructor(self):
        with pytest.raises(V2XError):
            V2XController(node_id=1, channel=100)

    def test_empty_payload_raises(self):
        with pytest.raises(V2XError):
            V2XMessage(
                msg_type=V2XMessageType.BSM,
                sender_id=1,
                channel=CCH_CHANNEL,
                payload=b"",
            )


# ===========================================================================
# ConnectivityManager
# ===========================================================================

class TestConnectivityManager:

    def test_init_all_interfaces(self):
        mgr = ConnectivityManager()
        mgr.init_canfd(node_id=1)
        mgr.init_ethernet(mac_address=MY_MAC)
        mgr.init_pcie(device_id=0x1000, num_lanes=4)
        mgr.init_v2x(node_id=0xDEAD)
        assert mgr.canfd is not None
        assert mgr.ethernet is not None
        assert mgr.pcie is not None
        assert mgr.v2x is not None

    def test_double_init_canfd_raises(self):
        mgr = ConnectivityManager()
        mgr.init_canfd(node_id=1)
        with pytest.raises(ConnectivityBaseError):
            mgr.init_canfd(node_id=2)

    def test_double_init_ethernet_raises(self):
        mgr = ConnectivityManager()
        mgr.init_ethernet(mac_address=MY_MAC)
        with pytest.raises(ConnectivityBaseError):
            mgr.init_ethernet(mac_address=OTHER_MAC)

    def test_double_init_pcie_raises(self):
        mgr = ConnectivityManager()
        mgr.init_pcie(device_id=0x1000)
        with pytest.raises(ConnectivityBaseError):
            mgr.init_pcie(device_id=0x2000)

    def test_double_init_v2x_raises(self):
        mgr = ConnectivityManager()
        mgr.init_v2x(node_id=1)
        with pytest.raises(ConnectivityBaseError):
            mgr.init_v2x(node_id=2)

    def test_link_status_all_interfaces(self):
        mgr = ConnectivityManager()
        mgr.init_canfd(node_id=1)
        mgr.init_ethernet(mac_address=MY_MAC)
        mgr.init_pcie(device_id=0x1000)
        mgr.init_v2x(node_id=1)
        status = mgr.link_status()
        assert "canfd" in status
        assert "ethernet" in status
        assert "pcie" in status
        assert "v2x" in status

    def test_link_status_empty_without_init(self):
        mgr = ConnectivityManager()
        assert mgr.link_status() == {}

    def test_any_link_up_false_initially(self):
        # No interfaces at all → False
        mgr = ConnectivityManager()
        assert mgr.any_link_up() is False

    def test_any_link_up_true_after_ethernet_up(self):
        mgr = ConnectivityManager()
        eth = mgr.init_ethernet(mac_address=MY_MAC)
        eth.link_up()
        assert mgr.any_link_up() is True

    def test_any_link_up_true_after_pcie_l0(self):
        mgr = ConnectivityManager()
        pcie = mgr.init_pcie(device_id=0x1000)
        pcie.train_link()
        assert mgr.any_link_up() is True

    def test_accessors_none_before_init(self):
        mgr = ConnectivityManager()
        assert mgr.canfd is None
        assert mgr.ethernet is None
        assert mgr.pcie is None
        assert mgr.v2x is None
