"""
AstraCore Neo — Module 8: Telemetry testbench.

Coverage:
  - TelemetryLogger: levels, ring buffer, filtering, counters, min_level
  - ThermalZone: thresholds, state transitions, throttle, shutdown, slope
  - ThermalMonitor: multi-zone, hottest, summary, add/remove
  - FaultPredictor: metric tracking, risk levels, spike, trend, multi-metric
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from telemetry import (
    TelemetryLogger, LogLevel, LogEntry,
    ThermalMonitor, ThermalZone, ThermalZoneConfig, ThermalState, ThermalReading,
    FaultPredictor, MetricTracker, MetricConfig, FaultRisk, FaultPrediction,
    TelemetryBaseError, LoggerError, ThermalError, ThermalShutdownError, FaultPredictorError,
)


# ===========================================================================
# TelemetryLogger tests
# ===========================================================================

class TestLoggerBasic:
    def setup_method(self):
        self.log = TelemetryLogger(capacity=100)

    def test_info_logged(self):
        self.log.info("hal", "boot complete")
        assert self.log.current_size() == 1

    def test_entry_fields(self):
        entry = self.log.info("hal", "test message", key="value")
        assert entry.source == "hal"
        assert entry.message == "test message"
        assert entry.metadata["key"] == "value"
        assert entry.level == LogLevel.INFO
        assert entry.timestamp_us > 0
        assert entry.sequence == 1

    def test_sequence_increments(self):
        self.log.info("a", "1")
        self.log.info("b", "2")
        self.log.info("c", "3")
        entries = self.log.snapshot()
        assert [e.sequence for e in entries] == [1, 2, 3]

    def test_all_levels(self):
        self.log.debug("s", "d")
        self.log.info("s", "i")
        self.log.warning("s", "w")
        self.log.error("s", "e")
        self.log.critical("s", "c")
        assert self.log.current_size() == 5

    def test_level_counters(self):
        self.log.warning("s", "w")
        self.log.error("s", "e")
        self.log.error("s", "e2")
        assert self.log.count(LogLevel.WARNING) == 1
        assert self.log.count(LogLevel.ERROR) == 2

    def test_total_logged(self):
        for i in range(10):
            self.log.info("s", f"msg {i}")
        assert self.log.total_logged() == 10

    def test_has_errors_false_initially(self):
        self.log.info("s", "ok")
        assert not self.log.has_errors()

    def test_has_errors_true_after_error(self):
        self.log.error("s", "bad")
        assert self.log.has_errors()

    def test_has_errors_true_after_critical(self):
        self.log.critical("s", "very bad")
        assert self.log.has_errors()

    def test_repr(self):
        r = repr(self.log)
        assert "TelemetryLogger" in r

    def test_invalid_capacity_raises(self):
        with pytest.raises(LoggerError):
            TelemetryLogger(capacity=0)


class TestLoggerFiltering:
    def setup_method(self):
        self.log = TelemetryLogger(capacity=100, min_level=LogLevel.WARNING)

    def test_below_min_level_filtered(self):
        result = self.log.debug("s", "debug msg")
        assert result is None
        assert self.log.current_size() == 0

    def test_info_below_warning_filtered(self):
        result = self.log.info("s", "info msg")
        assert result is None

    def test_at_min_level_logged(self):
        result = self.log.warning("s", "warn")
        assert result is not None
        assert self.log.current_size() == 1

    def test_above_min_level_logged(self):
        self.log.critical("s", "crit")
        assert self.log.current_size() == 1

    def test_set_min_level(self):
        self.log.min_level = LogLevel.DEBUG
        self.log.debug("s", "now visible")
        assert self.log.current_size() == 1


class TestLoggerRingBuffer:
    def test_ring_buffer_evicts_oldest(self):
        log = TelemetryLogger(capacity=5)
        for i in range(10):
            log.info("s", f"msg {i}")
        assert log.current_size() == 5
        entries = log.snapshot()
        # Should have msgs 5-9
        assert entries[0].message == "msg 5"
        assert entries[-1].message == "msg 9"

    def test_dropped_counter(self):
        log = TelemetryLogger(capacity=3)
        for i in range(6):
            log.info("s", f"msg {i}")
        assert log.dropped == 3

    def test_total_logged_counts_all_including_dropped(self):
        log = TelemetryLogger(capacity=3)
        for i in range(6):
            log.info("s", f"msg {i}")
        assert log.total_logged() == 6

    def test_clear(self):
        log = TelemetryLogger(capacity=10)
        log.info("s", "msg")
        log.clear()
        assert log.current_size() == 0


class TestLoggerQuery:
    def setup_method(self):
        self.log = TelemetryLogger(capacity=100)
        self.log.debug("hal", "d1")
        self.log.info("hal", "i1")
        self.log.warning("ecc", "w1")
        self.log.error("ecc", "e1")
        self.log.info("safety", "i2")

    def test_filter_by_level(self):
        errors = self.log.filter_by_level(LogLevel.WARNING)
        assert len(errors) == 2  # warning + error

    def test_filter_by_source(self):
        ecc = self.log.filter_by_source("ecc")
        assert len(ecc) == 2

    def test_latest_n(self):
        last2 = self.log.latest(2)
        assert len(last2) == 2
        assert last2[-1].source == "safety"

    def test_latest_more_than_size(self):
        all_entries = self.log.latest(100)
        assert len(all_entries) == 5


# ===========================================================================
# ThermalZone tests
# ===========================================================================

class TestThermalZoneStates:
    def setup_method(self):
        cfg = ThermalZoneConfig(
            warning_c=75.0, throttle_c=85.0, critical_c=95.0, shutdown_c=105.0
        )
        self.zone = ThermalZone("cpu", cfg)

    def test_nominal_state(self):
        r = self.zone.update(50.0)
        assert r.state == ThermalState.NOMINAL
        assert not r.throttle_active

    def test_warning_state(self):
        r = self.zone.update(80.0)
        assert r.state == ThermalState.WARNING

    def test_throttle_state(self):
        r = self.zone.update(88.0)
        assert r.state == ThermalState.THROTTLED
        assert r.throttle_active

    def test_critical_state(self):
        r = self.zone.update(97.0)
        assert r.state == ThermalState.CRITICAL

    def test_shutdown_raises(self):
        with pytest.raises(ThermalShutdownError):
            self.zone.update(106.0)
        assert self.zone.state == ThermalState.SHUTDOWN

    def test_exact_warning_threshold(self):
        r = self.zone.update(75.0)
        assert r.state == ThermalState.WARNING

    def test_exact_throttle_threshold(self):
        r = self.zone.update(85.0)
        assert r.state == ThermalState.THROTTLED

    def test_just_below_warning(self):
        r = self.zone.update(74.9)
        assert r.state == ThermalState.NOMINAL


class TestThermalZoneClock:
    def setup_method(self):
        cfg = ThermalZoneConfig(throttle_pct=50.0)
        self.zone = ThermalZone("cpu", cfg)

    def test_full_clock_nominal(self):
        self.zone.update(50.0)
        assert self.zone.effective_clock_pct() == 100.0

    def test_reduced_clock_throttled(self):
        self.zone.update(88.0)
        assert self.zone.effective_clock_pct() == 50.0

    def test_zero_clock_critical(self):
        self.zone.update(97.0)
        assert self.zone.effective_clock_pct() == 0.0


class TestThermalZoneStats:
    def test_peak_temp_tracked(self):
        zone = ThermalZone("cpu")
        zone.update(60.0)
        zone.update(80.0)
        zone.update(70.0)
        assert zone.peak_temp == 80.0

    def test_sample_count(self):
        zone = ThermalZone("cpu")
        for _ in range(5):
            zone.update(50.0)
        assert zone.sample_count == 5

    def test_throttle_count(self):
        zone = ThermalZone("cpu", ThermalZoneConfig(throttle_c=80.0))
        zone.update(85.0)
        zone.update(87.0)
        zone.update(50.0)
        assert zone.throttle_count == 2

    def test_slope_rising(self):
        zone = ThermalZone("cpu")
        for t in range(20, 50, 2):
            zone.update(float(t))
        assert zone.temperature_slope() > 0

    def test_slope_flat(self):
        zone = ThermalZone("cpu")
        for _ in range(8):
            zone.update(50.0)
        assert abs(zone.temperature_slope()) < 0.01

    def test_slope_single_sample(self):
        zone = ThermalZone("cpu")
        zone.update(50.0)
        assert zone.temperature_slope() == 0.0

    def test_repr(self):
        zone = ThermalZone("cpu")
        r = repr(zone)
        assert "ThermalZone" in r


class TestThermalMonitor:
    def setup_method(self):
        self.monitor = ThermalMonitor()

    def test_add_zone(self):
        self.monitor.add_zone("cpu")
        assert "cpu" in self.monitor.zone_names()

    def test_duplicate_zone_raises(self):
        self.monitor.add_zone("cpu")
        with pytest.raises(ThermalError):
            self.monitor.add_zone("cpu")

    def test_remove_zone(self):
        self.monitor.add_zone("cpu")
        self.monitor.remove_zone("cpu")
        assert "cpu" not in self.monitor.zone_names()

    def test_remove_nonexistent_raises(self):
        with pytest.raises(ThermalError):
            self.monitor.remove_zone("ghost")

    def test_update_zone(self):
        self.monitor.add_zone("cpu", ThermalZoneConfig())
        r = self.monitor.update_zone("cpu", 50.0)
        assert r.state == ThermalState.NOMINAL

    def test_update_unknown_zone_raises(self):
        with pytest.raises(ThermalError):
            self.monitor.update_zone("ghost", 50.0)

    def test_any_throttled(self):
        cfg = ThermalZoneConfig(throttle_c=80.0)
        self.monitor.add_zone("cpu", cfg)
        self.monitor.update_zone("cpu", 85.0)
        assert self.monitor.any_throttled()

    def test_any_throttled_false(self):
        self.monitor.add_zone("cpu")
        self.monitor.update_zone("cpu", 50.0)
        assert not self.monitor.any_throttled()

    def test_any_critical(self):
        cfg = ThermalZoneConfig(critical_c=90.0)
        self.monitor.add_zone("npu", cfg)
        self.monitor.update_zone("npu", 92.0)
        assert self.monitor.any_critical()

    def test_hottest_zone(self):
        self.monitor.add_zone("cpu")
        self.monitor.add_zone("npu")
        self.monitor.update_zone("cpu", 60.0)
        self.monitor.update_zone("npu", 80.0)
        hot = self.monitor.hottest_zone()
        assert hot.name == "npu"

    def test_hottest_zone_empty(self):
        assert self.monitor.hottest_zone() is None

    def test_summary(self):
        self.monitor.add_zone("cpu")
        self.monitor.update_zone("cpu", 70.0)
        s = self.monitor.summary()
        assert "cpu" in s
        assert s["cpu"]["temp_c"] == 70.0

    def test_repr(self):
        r = repr(self.monitor)
        assert "ThermalMonitor" in r


# ===========================================================================
# FaultPredictor tests
# ===========================================================================

class TestFaultPredictorBasic:
    def setup_method(self):
        self.fp = FaultPredictor()
        self.fp.add_metric(MetricConfig(
            "ecc_rate", warning_threshold=5.0, critical_threshold=20.0, window_size=8
        ))

    def test_add_metric(self):
        assert "ecc_rate" in self.fp.metric_names()

    def test_duplicate_metric_raises(self):
        with pytest.raises(FaultPredictorError):
            self.fp.add_metric(MetricConfig("ecc_rate", 5.0, 20.0))

    def test_remove_metric(self):
        self.fp.remove_metric("ecc_rate")
        assert "ecc_rate" not in self.fp.metric_names()

    def test_remove_nonexistent_raises(self):
        with pytest.raises(FaultPredictorError):
            self.fp.remove_metric("ghost")

    def test_push_unknown_raises(self):
        with pytest.raises(FaultPredictorError):
            self.fp.push("ghost", 1.0)

    def test_latest_before_push_raises(self):
        with pytest.raises(FaultPredictorError):
            self.fp.latest("ecc_rate")

    def test_repr(self):
        r = repr(self.fp)
        assert "FaultPredictor" in r


class TestFaultPredictorRisk:
    def setup_method(self):
        self.fp = FaultPredictor()
        self.fp.add_metric(MetricConfig(
            "temp", warning_threshold=75.0, critical_threshold=95.0, window_size=16
        ))

    def test_nominal_no_risk(self):
        pred = self.fp.push("temp", 50.0)
        assert pred.risk == FaultRisk.NONE

    def test_above_warning_low_risk(self):
        pred = self.fp.push("temp", 76.0)
        assert pred.risk in (FaultRisk.LOW, FaultRisk.MEDIUM)

    def test_at_critical_threshold_critical_risk(self):
        pred = self.fp.push("temp", 95.0)
        assert pred.risk == FaultRisk.CRITICAL
        assert pred.confidence >= 0.9

    def test_above_critical_threshold_critical_risk(self):
        pred = self.fp.push("temp", 100.0)
        assert pred.risk == FaultRisk.CRITICAL

    def test_prediction_has_stats(self):
        for v in [50.0, 52.0, 54.0, 56.0]:
            pred = self.fp.push("temp", v)
        assert pred.mean > 0
        assert pred.std_dev >= 0
        assert pred.current_value == 56.0

    def test_any_high_risk_false(self):
        self.fp.push("temp", 50.0)
        assert not self.fp.any_high_risk()

    def test_any_high_risk_true(self):
        self.fp.push("temp", 96.0)
        assert self.fp.any_high_risk()


class TestFaultPredictorSpike:
    def test_spike_detected(self):
        fp = FaultPredictor()
        fp.add_metric(MetricConfig(
            "voltage", warning_threshold=50.0, critical_threshold=100.0,
            window_size=16, spike_std_multiplier=2.0
        ))
        # Establish baseline
        for _ in range(8):
            fp.push("voltage", 10.0)
        # Inject spike
        pred = fp.push("voltage", 40.0)
        assert pred.risk.value >= FaultRisk.MEDIUM.value


class TestFaultPredictorTrend:
    def test_rising_trend_escalates_risk(self):
        fp = FaultPredictor()
        fp.add_metric(MetricConfig(
            "ecc_rate", warning_threshold=5.0, critical_threshold=10.0,
            window_size=8
        ))
        # Slow rise that will hit critical soon
        for v in [1, 2, 3, 4, 5, 6, 7, 8]:
            pred = fp.push("ecc_rate", float(v))
        # Should have detected rising trend toward critical
        assert pred.trend_slope > 0

    def test_flat_trend_no_escalation(self):
        fp = FaultPredictor()
        fp.add_metric(MetricConfig(
            "util", warning_threshold=90.0, critical_threshold=99.0, window_size=8
        ))
        for _ in range(8):
            pred = fp.push("util", 50.0)
        assert abs(pred.trend_slope) < 0.1


class TestFaultPredictorMultiMetric:
    def test_highest_risk_returns_worst(self):
        fp = FaultPredictor()
        fp.add_metric(MetricConfig("a", 5.0, 10.0))
        fp.add_metric(MetricConfig("b", 5.0, 10.0))
        fp.push("a", 2.0)   # NONE risk
        fp.push("b", 11.0)  # CRITICAL risk
        worst = fp.highest_risk()
        assert worst.metric == "b"
        assert worst.risk == FaultRisk.CRITICAL

    def test_highest_risk_empty(self):
        fp = FaultPredictor()
        assert fp.highest_risk() is None

    def test_tracker_window_data(self):
        fp = FaultPredictor()
        fp.add_metric(MetricConfig("x", 5.0, 10.0, window_size=4))
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            fp.push("x", v)
        data = fp.tracker("x").window_data()
        assert len(data) == 4
        assert data == [2.0, 3.0, 4.0, 5.0]

    def test_tracker_sample_count(self):
        fp = FaultPredictor()
        fp.add_metric(MetricConfig("x", 5.0, 10.0))
        for _ in range(6):
            fp.push("x", 1.0)
        assert fp.tracker("x").sample_count == 6
