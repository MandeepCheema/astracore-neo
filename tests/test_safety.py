"""
AstraCore Neo — Module 6: Safety testbench.

Coverage:
  - ECC: encode/decode, single-bit correction, double-bit detection, scrub, counters
  - TMR: majority vote, fault lane detection, triple disagreement, numpy arrays
  - Watchdog: start/stop, valid service, timeout, window violation, wrong token
  - ClockMonitor: frequency bounds, clock loss, glitch, domain management
  - SafetyManager: state escalation, event log, source filtering, lifecycle
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from safety import (
    ECCEngine, ECCConfig, ECCError, BitFlipType, CorrectionResult,
    TMRVoter, TMRResult, TMRError,
    WatchdogTimer, WatchdogConfig, WatchdogError,
    ClockMonitor, ClockMonitorConfig, ClockFaultType, ClockFault,
    SafetyManager, SafetyEvent, SafetySeverity, SafetyError,
    SafetyBaseError,
)
from safety.safety_manager import SafetyState
from safety.watchdog import WatchdogResponse


# ===========================================================================
# ECC tests
# ===========================================================================

class TestECCEncodeDecode:
    def setup_method(self):
        self.ecc = ECCEngine()

    def test_encode_returns_72bit(self):
        encoded = self.ecc.encode(0xDEADBEEFCAFEBABE)
        assert encoded >> 64 != 0  # parity bits present
        assert (encoded & 0xFFFFFFFFFFFFFFFF) == 0xDEADBEEFCAFEBABE

    def test_decode_clean_word_no_error(self):
        word = 0x0102030405060708
        encoded = self.ecc.encode(word)
        result = self.ecc.decode(encoded, bank=0, address=0)
        assert result.error_type == BitFlipType.NONE
        assert not result.corrected
        assert result.corrected_word == word

    def test_decode_single_bit_error_corrected(self):
        word = 0xABCDEF1234567890
        encoded = self.ecc.encode(word)
        # Flip bit 0
        corrupted = encoded ^ 1
        result = self.ecc.decode(corrupted, bank=0, address=0x10)
        assert result.corrected
        assert result.error_type == BitFlipType.SINGLE_BIT
        assert result.corrected_word == word

    def test_decode_single_bit_error_various_positions(self):
        word = 0xFFFFFFFFFFFFFFFF
        encoded = self.ecc.encode(word)
        for bit in range(0, 64, 8):  # test every 8th bit
            corrupted = encoded ^ (1 << bit)
            result = self.ecc.decode(corrupted, bank=0, address=bit)
            assert result.corrected or result.error_type == BitFlipType.NONE

    def test_decode_double_bit_error_raises(self):
        word = 0x1234567890ABCDEF
        encoded = self.ecc.encode(word)
        # Flip two data bits
        corrupted = encoded ^ 0b11
        with pytest.raises(ECCError):
            self.ecc.decode(corrupted, bank=0, address=0)

    def test_decode_invalid_bank_raises(self):
        encoded = self.ecc.encode(0x1234)
        with pytest.raises(ECCError):
            self.ecc.decode(encoded, bank=99, address=0)

    def test_single_bit_error_increments_counter(self):
        word = 0xCAFEBABEDEADBEEF
        encoded = self.ecc.encode(word)
        corrupted = encoded ^ 1
        self.ecc.decode(corrupted, bank=0, address=0)
        assert self.ecc.single_bit_error_count(0) == 1
        assert self.ecc.correction_count(0) == 1

    def test_double_bit_error_increments_counter(self):
        word = 0xCAFEBABEDEADBEEF
        encoded = self.ecc.encode(word)
        corrupted = encoded ^ 0b11
        with pytest.raises(ECCError):
            self.ecc.decode(corrupted, bank=0, address=0)
        assert self.ecc.double_bit_error_count(0) == 1

    def test_no_error_no_counter_increment(self):
        word = 0xAAAAAAAAAAAAAAAA
        encoded = self.ecc.encode(word)
        self.ecc.decode(encoded, bank=1, address=0)
        assert self.ecc.single_bit_error_count(1) == 0
        assert self.ecc.double_bit_error_count(1) == 0

    def test_reset_counters(self):
        word = 0x1234
        encoded = self.ecc.encode(word)
        self.ecc.decode(encoded ^ 1, bank=0, address=0)
        self.ecc.reset_counters()
        assert self.ecc.single_bit_error_count(0) == 0
        assert self.ecc.total_errors() == 0

    def test_total_errors_across_banks(self):
        for bank in range(4):
            encoded = self.ecc.encode(0x1234)
            self.ecc.decode(encoded ^ 1, bank=bank, address=0)
        assert self.ecc.total_errors() == 4

    def test_repr(self):
        r = repr(self.ecc)
        assert "ECCEngine" in r


class TestECCScrub:
    def setup_method(self):
        self.ecc = ECCEngine()

    def test_scrub_clean_bank_no_corrections(self):
        words = [self.ecc.encode(i * 0x1111111111111111) for i in range(8)]
        scrubbed, corrections = self.ecc.scrub_bank(0, words)
        assert corrections == 0
        assert len(scrubbed) == len(words)

    def test_scrub_corrects_single_bit_errors(self):
        words = [self.ecc.encode(0xDEADBEEF + i) for i in range(4)]
        # Inject a single-bit error into word 2
        words[2] ^= 1
        scrubbed, corrections = self.ecc.scrub_bank(0, words)
        assert corrections == 1

    def test_scrub_increments_scrub_count(self):
        self.ecc.scrub_bank(0, [])
        self.ecc.scrub_bank(1, [])
        assert self.ecc.scrub_count == 2

    def test_scrub_invalid_bank_raises(self):
        with pytest.raises(ECCError):
            self.ecc.scrub_bank(99, [])


# ===========================================================================
# TMR tests
# ===========================================================================

class TestTMRVoterScalars:
    def setup_method(self):
        self.voter = TMRVoter()

    def test_all_agree_integer(self):
        result = self.voter.vote(42, 42, 42)
        assert result.voted_value == 42
        assert result.agreement
        assert result.faulty_lane is None
        assert result.vote_count == 3

    def test_lane_c_faulty(self):
        result = self.voter.vote(10, 10, 99)
        assert result.voted_value == 10
        assert result.faulty_lane == "C"
        assert result.vote_count == 2

    def test_lane_b_faulty(self):
        result = self.voter.vote(7, 99, 7)
        assert result.voted_value == 7
        assert result.faulty_lane == "B"

    def test_lane_a_faulty(self):
        result = self.voter.vote(99, 5, 5)
        assert result.voted_value == 5
        assert result.faulty_lane == "A"

    def test_all_disagree_raises(self):
        with pytest.raises(TMRError):
            self.voter.vote(1, 2, 3)

    def test_fault_counter_increments(self):
        self.voter.vote(1, 1, 99)  # C faulty
        self.voter.vote(1, 99, 1)  # B faulty
        self.voter.vote(99, 1, 1)  # A faulty
        assert self.voter.fault_count("A") == 1
        assert self.voter.fault_count("B") == 1
        assert self.voter.fault_count("C") == 1

    def test_total_votes_counter(self):
        self.voter.vote(1, 1, 1)
        self.voter.vote(2, 2, 2)
        assert self.voter.total_votes == 2

    def test_triple_disagreement_counter(self):
        with pytest.raises(TMRError):
            self.voter.vote(1, 2, 3)
        assert self.voter.triple_disagreements == 1

    def test_reset_counters(self):
        self.voter.vote(1, 1, 2)
        self.voter.reset_counters()
        assert self.voter.total_votes == 0
        assert self.voter.fault_count("C") == 0

    def test_float_voting_within_tolerance(self):
        result = self.voter.vote(1.0, 1.0, 1.0 + 1e-9)
        assert result.agreement
        assert result.voted_value == pytest.approx(1.0)

    def test_float_voting_outside_tolerance(self):
        result = self.voter.vote(1.0, 1.0, 2.0)
        assert result.faulty_lane == "C"

    def test_invalid_lane_name_raises(self):
        with pytest.raises(ValueError):
            self.voter.fault_count("X")

    def test_repr(self):
        r = repr(self.voter)
        assert "TMRVoter" in r


class TestTMRVoterArrays:
    def setup_method(self):
        self.voter = TMRVoter()

    def test_numpy_arrays_all_agree(self):
        a = np.array([1.0, 2.0, 3.0])
        result = self.voter.vote(a, a.copy(), a.copy())
        assert result.agreement
        np.testing.assert_array_equal(result.voted_value, a)

    def test_numpy_arrays_one_faulty(self):
        a = np.array([1.0, 2.0, 3.0])
        b = a.copy()
        c = np.array([1.0, 2.0, 99.0])  # C is faulty
        result = self.voter.vote(a, b, c)
        assert result.faulty_lane == "C"
        np.testing.assert_array_equal(result.voted_value, a)

    def test_numpy_arrays_shape_mismatch_is_not_equal(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0])
        c = np.array([1.0, 2.0, 3.0])  # different shape = not equal
        result = self.voter.vote(a, b, c)
        assert result.faulty_lane == "C"


# ===========================================================================
# Watchdog tests
# ===========================================================================

class TestWatchdogLifecycle:
    def test_start_stop(self):
        wdt = WatchdogTimer()
        assert not wdt.is_running
        wdt.start()
        assert wdt.is_running
        wdt.stop()
        assert not wdt.is_running

    def test_double_start_raises(self):
        wdt = WatchdogTimer()
        wdt.start()
        with pytest.raises(WatchdogError):
            wdt.start()
        wdt.stop()

    def test_stop_without_start_raises(self):
        wdt = WatchdogTimer()
        with pytest.raises(WatchdogError):
            wdt.stop()

    def test_service_without_start_raises(self):
        wdt = WatchdogTimer()
        with pytest.raises(WatchdogError):
            wdt.service(token=0xA5A5)


class TestWatchdogService:
    def _make_wdt(self, timeout_ms=100.0, window_open_ms=10.0, token=0xA5A5):
        cfg = WatchdogConfig(timeout_ms=timeout_ms, window_open_ms=window_open_ms, token=token)
        wdt = WatchdogTimer(cfg)
        wdt.start()
        return wdt

    def test_valid_service_in_window(self):
        wdt = self._make_wdt()
        wdt._inject_elapsed(50.0)  # 50ms — within [10, 100] window
        wdt.service(token=0xA5A5)
        assert wdt.service_count == 1

    def test_wrong_token_raises(self):
        wdt = self._make_wdt()
        wdt._inject_elapsed(50.0)
        with pytest.raises(WatchdogError):
            wdt.service(token=0x1234)

    def test_service_too_early_raises(self):
        wdt = self._make_wdt(window_open_ms=50.0)
        wdt._inject_elapsed(10.0)  # before window opens at 50ms
        with pytest.raises(WatchdogError):
            wdt.service(token=0xA5A5)

    def test_early_kick_counter(self):
        wdt = self._make_wdt(window_open_ms=50.0)
        wdt._inject_elapsed(10.0)
        with pytest.raises(WatchdogError):
            wdt.service(token=0xA5A5)
        assert wdt.early_kick_count == 1

    def test_service_after_timeout_raises(self):
        wdt = self._make_wdt(timeout_ms=100.0)
        wdt._inject_elapsed(150.0)  # past timeout
        with pytest.raises(WatchdogError):
            wdt.service(token=0xA5A5)

    def test_timeout_counter_increments(self):
        wdt = self._make_wdt(timeout_ms=100.0)
        wdt._inject_elapsed(200.0)
        with pytest.raises(WatchdogError):
            wdt.service(token=0xA5A5)
        assert wdt.timeout_count == 1

    def test_check_timeout_true_when_expired(self):
        wdt = self._make_wdt(timeout_ms=100.0)
        wdt._inject_elapsed(200.0)
        assert wdt.check_timeout()

    def test_check_timeout_false_when_ok(self):
        wdt = self._make_wdt(timeout_ms=100.0)
        wdt._inject_elapsed(50.0)
        assert not wdt.check_timeout()

    def test_check_timeout_false_when_not_running(self):
        wdt = WatchdogTimer()
        assert not wdt.check_timeout()

    def test_repr(self):
        wdt = WatchdogTimer()
        r = repr(wdt)
        assert "WatchdogTimer" in r


class TestWatchdogEscalation:
    def test_escalation_no_timeouts(self):
        wdt = WatchdogTimer()
        assert wdt.escalation_level() == WatchdogResponse.INTERRUPT

    def test_escalation_after_two_timeouts(self):
        cfg = WatchdogConfig(timeout_ms=100.0, window_open_ms=10.0)
        wdt = WatchdogTimer(cfg)
        wdt.start()
        # Simulate 2 timeouts
        for _ in range(2):
            wdt._inject_elapsed(200.0)
            with pytest.raises(WatchdogError):
                wdt.service(token=cfg.token)
        assert wdt.escalation_level() == WatchdogResponse.NMI

    def test_escalation_after_three_timeouts(self):
        cfg = WatchdogConfig(timeout_ms=100.0, window_open_ms=10.0)
        wdt = WatchdogTimer(cfg)
        wdt.start()
        for _ in range(3):
            wdt._inject_elapsed(200.0)
            with pytest.raises(WatchdogError):
                wdt.service(token=cfg.token)
        assert wdt.escalation_level() == WatchdogResponse.RESET


# ===========================================================================
# Clock Monitor tests
# ===========================================================================

class TestClockMonitorDomains:
    def setup_method(self):
        self.monitor = ClockMonitor()

    def test_add_and_list_domains(self):
        self.monitor.add_domain("core", ClockMonitorConfig(expected_mhz=1000.0))
        self.monitor.add_domain("ddr", ClockMonitorConfig(expected_mhz=400.0))
        assert "core" in self.monitor.domain_names()
        assert "ddr" in self.monitor.domain_names()

    def test_remove_domain(self):
        self.monitor.add_domain("temp", ClockMonitorConfig(expected_mhz=100.0))
        self.monitor.remove_domain("temp")
        assert "temp" not in self.monitor.domain_names()

    def test_remove_nonexistent_domain_raises(self):
        with pytest.raises(KeyError):
            self.monitor.remove_domain("ghost")

    def test_check_unknown_domain_raises(self):
        with pytest.raises(KeyError):
            self.monitor.check_frequency("unknown", 100.0)

    def test_repr(self):
        r = repr(self.monitor)
        assert "ClockMonitor" in r


class TestClockMonitorFrequency:
    def setup_method(self):
        self.monitor = ClockMonitor()
        self.monitor.add_domain("core", ClockMonitorConfig(
            expected_mhz=1000.0, tolerance_pct=5.0
        ))

    def test_within_tolerance_no_fault(self):
        fault = self.monitor.check_frequency("core", measured_mhz=1000.0)
        assert fault.fault_type == ClockFaultType.NONE

    def test_upper_edge_within_tolerance(self):
        fault = self.monitor.check_frequency("core", measured_mhz=1049.0)
        assert fault.fault_type == ClockFaultType.NONE

    def test_above_tolerance_freq_high(self):
        fault = self.monitor.check_frequency("core", measured_mhz=1060.0)
        assert fault.fault_type == ClockFaultType.FREQ_HIGH

    def test_below_tolerance_freq_low(self):
        fault = self.monitor.check_frequency("core", measured_mhz=940.0)
        assert fault.fault_type == ClockFaultType.FREQ_LOW

    def test_fault_logged(self):
        self.monitor.check_frequency("core", measured_mhz=1200.0)
        log = self.monitor.fault_log()
        assert len(log) == 1
        assert log[0].fault_type == ClockFaultType.FREQ_HIGH

    def test_no_fault_not_logged(self):
        self.monitor.check_frequency("core", measured_mhz=1000.0)
        assert len(self.monitor.fault_log()) == 0

    def test_any_fault_true_when_fault_present(self):
        self.monitor.check_frequency("core", measured_mhz=2000.0)
        assert self.monitor.any_fault()

    def test_any_fault_false_when_clean(self):
        self.monitor.check_frequency("core", measured_mhz=1000.0)
        assert not self.monitor.any_fault()

    def test_domain_status_updated(self):
        self.monitor.check_frequency("core", measured_mhz=500.0)
        assert self.monitor.domain_status("core") == ClockFaultType.FREQ_LOW

    def test_clear_fault_log(self):
        self.monitor.check_frequency("core", measured_mhz=2000.0)
        self.monitor.clear_fault_log()
        assert len(self.monitor.fault_log()) == 0
        assert not self.monitor.any_fault()


class TestClockMonitorLoss:
    def setup_method(self):
        self.monitor = ClockMonitor()
        self.monitor.add_domain("pll", ClockMonitorConfig(
            expected_mhz=400.0, loss_timeout_us=10.0
        ))

    def test_clock_present_no_fault(self):
        fault = self.monitor.check_clock_loss("pll", time_since_last_edge_us=5.0)
        assert fault.fault_type == ClockFaultType.NONE

    def test_clock_loss_detected(self):
        fault = self.monitor.check_clock_loss("pll", time_since_last_edge_us=15.0)
        assert fault.fault_type == ClockFaultType.CLOCK_LOSS

    def test_clock_loss_logged(self):
        self.monitor.check_clock_loss("pll", time_since_last_edge_us=20.0)
        assert self.monitor.fault_log()[0].fault_type == ClockFaultType.CLOCK_LOSS


class TestClockMonitorGlitch:
    def setup_method(self):
        self.monitor = ClockMonitor()
        self.monitor.add_domain("io", ClockMonitorConfig(
            expected_mhz=200.0, min_pulse_width_ns=2.0
        ))

    def test_valid_pulse_no_fault(self):
        fault = self.monitor.check_glitch("io", pulse_width_ns=5.0)
        assert fault.fault_type == ClockFaultType.NONE

    def test_glitch_detected(self):
        fault = self.monitor.check_glitch("io", pulse_width_ns=0.5)
        assert fault.fault_type == ClockFaultType.GLITCH

    def test_glitch_logged(self):
        self.monitor.check_glitch("io", pulse_width_ns=0.1)
        assert self.monitor.fault_log()[0].fault_type == ClockFaultType.GLITCH


# ===========================================================================
# SafetyManager tests
# ===========================================================================

class TestSafetyManagerLifecycle:
    def test_start_stop(self):
        sm = SafetyManager()
        assert not sm.is_running
        sm.start()
        assert sm.is_running
        sm.shutdown()
        assert not sm.is_running

    def test_double_start_raises(self):
        sm = SafetyManager()
        sm.start()
        with pytest.raises(SafetyError):
            sm.start()
        sm.shutdown()

    def test_report_without_start_raises(self):
        sm = SafetyManager()
        with pytest.raises(SafetyError):
            sm.report_event("ECC", SafetySeverity.INFO, "test")

    def test_repr(self):
        sm = SafetyManager()
        r = repr(sm)
        assert "SafetyManager" in r


class TestSafetyManagerStateTransitions:
    def setup_method(self):
        self.sm = SafetyManager()
        self.sm.start()

    def teardown_method(self):
        if self.sm.is_running:
            self.sm.shutdown()

    def test_initial_state_normal(self):
        assert self.sm.state == SafetyState.NORMAL

    def test_info_keeps_normal(self):
        self.sm.report_event("ECC", SafetySeverity.INFO, "single-bit corrected")
        assert self.sm.state == SafetyState.NORMAL

    def test_warning_transitions_to_degraded(self):
        self.sm.report_event("TMR", SafetySeverity.WARNING, "lane B fault")
        assert self.sm.state == SafetyState.DEGRADED

    def test_critical_transitions_to_safe_state(self):
        self.sm.report_event("CLOCK", SafetySeverity.CRITICAL, "clock loss")
        assert self.sm.state == SafetyState.SAFE_STATE

    def test_fatal_transitions_to_shutdown(self):
        self.sm.report_event("WATCHDOG", SafetySeverity.FATAL, "timeout")
        assert self.sm.state == SafetyState.SHUTDOWN

    def test_state_only_escalates(self):
        self.sm.report_event("X", SafetySeverity.FATAL, "fatal")
        self.sm.report_event("X", SafetySeverity.INFO, "info")
        assert self.sm.state == SafetyState.SHUTDOWN  # did not downgrade

    def test_is_safe_to_operate_normal(self):
        assert self.sm.is_safe_to_operate()

    def test_is_safe_to_operate_degraded(self):
        self.sm.report_event("X", SafetySeverity.WARNING, "w")
        assert self.sm.is_safe_to_operate()

    def test_is_not_safe_in_safe_state(self):
        self.sm.report_event("X", SafetySeverity.CRITICAL, "c")
        assert not self.sm.is_safe_to_operate()

    def test_reset_state(self):
        self.sm.report_event("X", SafetySeverity.FATAL, "f")
        self.sm.reset_state()
        assert self.sm.state == SafetyState.NORMAL


class TestSafetyManagerEvents:
    def setup_method(self):
        self.sm = SafetyManager()
        self.sm.start()

    def teardown_method(self):
        if self.sm.is_running:
            self.sm.shutdown()

    def test_event_logged(self):
        self.sm.report_event("ECC", SafetySeverity.WARNING, "test event")
        assert self.sm.total_events() == 1

    def test_event_count_by_severity(self):
        self.sm.report_event("A", SafetySeverity.INFO, "i1")
        self.sm.report_event("B", SafetySeverity.INFO, "i2")
        self.sm.report_event("C", SafetySeverity.WARNING, "w")
        assert self.sm.event_count(SafetySeverity.INFO) == 2
        assert self.sm.event_count(SafetySeverity.WARNING) == 1

    def test_events_by_source(self):
        self.sm.report_event("ECC", SafetySeverity.INFO, "e1")
        self.sm.report_event("TMR", SafetySeverity.WARNING, "t1")
        self.sm.report_event("ECC", SafetySeverity.INFO, "e2")
        ecc_events = self.sm.events_by_source("ECC")
        assert len(ecc_events) == 2

    def test_highest_severity(self):
        self.sm.report_event("A", SafetySeverity.INFO, "i")
        self.sm.report_event("B", SafetySeverity.CRITICAL, "c")
        self.sm.report_event("C", SafetySeverity.WARNING, "w")
        assert self.sm.highest_severity() == SafetySeverity.CRITICAL

    def test_highest_severity_empty(self):
        assert self.sm.highest_severity() is None

    def test_event_has_timestamp(self):
        event = self.sm.report_event("SRC", SafetySeverity.INFO, "msg")
        assert event.timestamp_us > 0

    def test_event_state_after_is_correct(self):
        event = self.sm.report_event("SRC", SafetySeverity.FATAL, "fatal")
        assert event.state_after == SafetyState.SHUTDOWN

    def test_clear_log(self):
        self.sm.report_event("X", SafetySeverity.WARNING, "w")
        self.sm.clear_log()
        assert self.sm.total_events() == 0
        assert self.sm.event_count(SafetySeverity.WARNING) == 0
