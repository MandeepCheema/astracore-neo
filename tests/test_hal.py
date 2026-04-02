"""
AstraCore Neo — HAL Testbench

Covers:
  - Device power states and transitions
  - Clock / DVFS control
  - Register read/write, bitfields, named access
  - Read-only / write-only enforcement
  - Interrupt enable, fire, mask, clear, handler dispatch
  - Reset clears all state
  - Error paths (wrong state, out-of-range, bad addresses)
"""

import sys
import os
import pytest

# Allow running from project root or tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.hal import (
    AstraCoreDevice, PowerState,
    RegisterFile, InterruptController,
    DeviceError, RegisterError, InterruptError, ClockError,
)
from src.hal.interrupts import (
    IRQ_MAC_DONE, IRQ_DMA_DONE, IRQ_MEM_ECC_ERR,
    IRQ_THERMAL_WARN, IRQ_DMS_ALERT,
)
from src.hal.registers import REGISTER_MAP


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def dev() -> AstraCoreDevice:
    """Powered-on device in IDLE state."""
    d = AstraCoreDevice(name="test-device")
    d.power_on()
    return d


@pytest.fixture
def active_dev(dev: AstraCoreDevice) -> AstraCoreDevice:
    """Device in ACTIVE state."""
    dev.start()
    return dev


@pytest.fixture
def reg() -> RegisterFile:
    return RegisterFile()


@pytest.fixture
def irq() -> InterruptController:
    return InterruptController()


# ===========================================================================
# 1. Device lifecycle
# ===========================================================================

class TestDeviceLifecycle:

    def test_initial_state_is_off(self):
        d = AstraCoreDevice()
        assert d.state == PowerState.OFF

    def test_power_on_reaches_idle(self, dev):
        assert dev.state == PowerState.IDLE

    def test_chip_id_correct(self, dev):
        assert dev.chip_id == AstraCoreDevice.CHIP_ID

    def test_chip_rev_correct(self, dev):
        assert dev.chip_rev == AstraCoreDevice.CHIP_REV

    def test_power_on_sets_default_clock(self, dev):
        assert dev.clock_ghz == 3.2

    def test_power_off_returns_off(self, dev):
        dev.power_off()
        assert dev.state == PowerState.OFF

    def test_power_off_resets_clock(self, dev):
        dev.power_off()
        assert dev.clock_ghz == 0.0

    def test_double_power_on_raises(self, dev):
        with pytest.raises(DeviceError):
            dev.power_on()

    def test_power_on_off_cycle(self):
        d = AstraCoreDevice()
        d.power_on()
        d.power_off()
        d.power_on()
        assert d.state == PowerState.IDLE

    def test_uptime_increases(self, dev):
        import time
        t0 = dev.uptime_seconds
        time.sleep(0.05)
        assert dev.uptime_seconds > t0

    def test_uptime_zero_when_off(self):
        d = AstraCoreDevice()
        assert d.uptime_seconds == 0.0

    def test_repr_contains_state(self, dev):
        r = repr(dev)
        assert "IDLE" in r
        assert "3.20GHz" in r


# ===========================================================================
# 2. Power state transitions
# ===========================================================================

class TestPowerStateTransitions:

    def test_idle_to_active(self, dev):
        dev.start()
        assert dev.state == PowerState.ACTIVE

    def test_active_to_idle(self, active_dev):
        active_dev.stop()
        assert active_dev.state == PowerState.IDLE

    def test_stop_from_idle_raises(self, dev):
        with pytest.raises(DeviceError):
            dev.stop()

    def test_start_from_active_raises(self, active_dev):
        with pytest.raises(DeviceError):
            active_dev.start()

    def test_idle_to_low_power(self, dev):
        dev.enter_low_power()
        assert dev.state == PowerState.LOW_POWER

    def test_active_to_low_power(self, active_dev):
        active_dev.enter_low_power()
        assert active_dev.state == PowerState.LOW_POWER

    def test_low_power_clock_is_500mhz(self, dev):
        dev.enter_low_power()
        assert dev.clock_ghz == 0.5

    def test_exit_low_power_returns_idle(self, dev):
        dev.enter_low_power()
        dev.exit_low_power()
        assert dev.state == PowerState.IDLE

    def test_exit_low_power_restores_clock(self, dev):
        dev.enter_low_power()
        dev.exit_low_power()
        assert dev.clock_ghz == 3.2

    def test_reset_from_idle(self, dev):
        dev.reset()
        assert dev.state == PowerState.IDLE

    def test_reset_from_active(self, active_dev):
        active_dev.reset()
        assert active_dev.state == PowerState.IDLE

    def test_reset_powered_off_raises(self):
        d = AstraCoreDevice()
        with pytest.raises(DeviceError):
            d.reset()

    def test_reset_clears_registers(self, dev):
        dev.regs.write(0x0008, 0xDEAD_BEEF)  # CTRL
        dev.reset()
        assert dev.regs.read(0x0008) == 0x0000_0000

    def test_reset_clears_interrupts(self, dev):
        dev.irq.enable(IRQ_MAC_DONE)
        dev.irq.fire(IRQ_MAC_DONE)
        dev.reset()
        assert not dev.irq.is_pending(IRQ_MAC_DONE)
        assert not dev.irq.is_enabled(IRQ_MAC_DONE)


# ===========================================================================
# 3. Clock / DVFS
# ===========================================================================

class TestClockDvfs:

    def test_set_clock_min(self, dev):
        dev.set_clock_ghz(2.5)
        assert dev.clock_ghz == 2.5

    def test_set_clock_max(self, dev):
        dev.set_clock_ghz(3.2)
        assert dev.clock_ghz == 3.2

    def test_set_clock_mid(self, dev):
        dev.set_clock_ghz(2.8)
        assert dev.clock_ghz == 2.8

    def test_set_clock_below_min_raises(self, dev):
        with pytest.raises(ClockError):
            dev.set_clock_ghz(2.4)

    def test_set_clock_above_max_raises(self, dev):
        with pytest.raises(ClockError):
            dev.set_clock_ghz(3.3)

    def test_set_clock_updates_clk_ctrl_register(self, dev):
        dev.set_clock_ghz(2.5)
        assert dev.regs.read(0x0010) == 2500

    def test_set_clock_updates_clk_status_register(self, dev):
        dev.set_clock_ghz(3.0)
        assert dev.regs.read(0x0014) == 3000

    def test_set_clock_off_raises(self):
        d = AstraCoreDevice()
        with pytest.raises(DeviceError):
            d.set_clock_ghz(3.0)

    def test_set_clock_in_low_power_fixed(self, dev):
        dev.enter_low_power()
        dev.set_clock_ghz(0.5)  # only valid value
        assert dev.clock_ghz == 0.5

    def test_set_clock_non_lp_in_low_power_raises(self, dev):
        dev.enter_low_power()
        with pytest.raises(ClockError):
            dev.set_clock_ghz(3.2)


# ===========================================================================
# 4. Register file — basic
# ===========================================================================

class TestRegisterBasic:

    def test_read_chip_id(self, reg):
        assert reg.read(0x0000) == 0xA2_4E_E0_01

    def test_write_read_ctrl(self, reg):
        reg.write(0x0008, 0xAB_CD_12_34)
        assert reg.read(0x0008) == 0xAB_CD_12_34

    def test_write_read_roundtrip(self, reg):
        for val in [0x0, 0x1, 0xFFFF_FFFF, 0x1234_5678]:
            reg.write(0x0008, val)
            assert reg.read(0x0008) == val

    def test_read_invalid_addr_raises(self, reg):
        with pytest.raises(RegisterError):
            reg.read(0xDEAD)

    def test_write_invalid_addr_raises(self, reg):
        with pytest.raises(RegisterError):
            reg.write(0xDEAD, 0)

    def test_write_read_only_raises(self, reg):
        with pytest.raises(RegisterError):
            reg.write(0x0000, 0)  # CHIP_ID

    def test_read_write_only_raises(self, reg):
        with pytest.raises(RegisterError):
            reg.read(0x0020)  # RESET_CTRL

    def test_write_value_out_of_range_raises(self, reg):
        with pytest.raises(RegisterError):
            reg.write(0x0008, 0x1_0000_0000)

    def test_reset_restores_defaults(self, reg):
        reg.write(0x0008, 0xFFFF_FFFF)
        reg.reset()
        assert reg.read(0x0008) == 0x0000_0000


# ===========================================================================
# 5. Register file — bitfields
# ===========================================================================

class TestRegisterBitfields:

    def test_read_field_full_width(self, reg):
        reg.write(0x0008, 0xABCD_1234)
        assert reg.read_field(0x0008, 31, 0) == 0xABCD_1234

    def test_read_field_lower_byte(self, reg):
        reg.write(0x0008, 0x0000_00AB)
        assert reg.read_field(0x0008, 7, 0) == 0xAB

    def test_read_field_upper_nibble(self, reg):
        reg.write(0x0008, 0xF000_0000)
        assert reg.read_field(0x0008, 31, 28) == 0xF

    def test_write_field_lower_nibble(self, reg):
        reg.write(0x0008, 0x0000_00F0)
        reg.write_field(0x0008, 3, 0, 0xA)
        assert reg.read(0x0008) == 0x0000_00FA

    def test_write_field_does_not_disturb_other_bits(self, reg):
        reg.write(0x0008, 0xFFFF_FF00)
        reg.write_field(0x0008, 7, 0, 0xAB)
        assert reg.read(0x0008) == 0xFFFF_FFAB

    def test_write_field_value_too_wide_raises(self, reg):
        with pytest.raises(RegisterError):
            reg.write_field(0x0008, 3, 0, 0x10)  # 5-bit value in 4-bit field

    def test_invalid_bitfield_msb_lt_lsb_raises(self, reg):
        with pytest.raises(RegisterError):
            reg.read_field(0x0008, 3, 7)

    def test_invalid_bitfield_oob_raises(self, reg):
        with pytest.raises(RegisterError):
            reg.read_field(0x0008, 32, 0)


# ===========================================================================
# 6. Register file — named access
# ===========================================================================

class TestRegisterNamed:

    def test_named_read_ctrl(self, reg):
        reg.write(0x0008, 0x1234)
        assert reg.named_read("CTRL") == 0x1234

    def test_named_write_ctrl(self, reg):
        reg.named_write("CTRL", 0x5678)
        assert reg.read(0x0008) == 0x5678

    def test_named_read_unknown_raises(self, reg):
        with pytest.raises(RegisterError):
            reg.named_read("NONEXISTENT")

    def test_named_write_unknown_raises(self, reg):
        with pytest.raises(RegisterError):
            reg.named_write("NONEXISTENT", 0)

    def test_dump_contains_all_registers(self, reg):
        d = reg.dump()
        assert "CHIP_ID" in d
        assert "CTRL" in d
        assert len(d) == len(REGISTER_MAP)


# ===========================================================================
# 7. Interrupt controller — enable / disable
# ===========================================================================

class TestInterruptEnableDisable:

    def test_irq_disabled_by_default(self, irq):
        assert not irq.is_enabled(IRQ_MAC_DONE)

    def test_enable_irq(self, irq):
        irq.enable(IRQ_MAC_DONE)
        assert irq.is_enabled(IRQ_MAC_DONE)

    def test_disable_irq(self, irq):
        irq.enable(IRQ_MAC_DONE)
        irq.disable(IRQ_MAC_DONE)
        assert not irq.is_enabled(IRQ_MAC_DONE)

    def test_enable_multiple(self, irq):
        irq.enable(IRQ_MAC_DONE)
        irq.enable(IRQ_DMA_DONE)
        assert irq.is_enabled(IRQ_MAC_DONE)
        assert irq.is_enabled(IRQ_DMA_DONE)

    def test_enable_mask_bits(self, irq):
        irq.enable(IRQ_MAC_DONE)
        irq.enable(IRQ_DMA_DONE)
        assert irq.enable_mask == 0b11

    def test_invalid_irq_raises(self, irq):
        with pytest.raises(InterruptError):
            irq.enable(32)

    def test_negative_irq_raises(self, irq):
        with pytest.raises(InterruptError):
            irq.enable(-1)


# ===========================================================================
# 8. Interrupt controller — fire / pending / clear
# ===========================================================================

class TestInterruptFireClear:

    def test_fire_sets_pending(self, irq):
        irq.fire(IRQ_MAC_DONE)
        assert irq.is_pending(IRQ_MAC_DONE)

    def test_clear_clears_pending(self, irq):
        irq.fire(IRQ_MAC_DONE)
        irq.clear(IRQ_MAC_DONE)
        assert not irq.is_pending(IRQ_MAC_DONE)

    def test_clear_all(self, irq):
        irq.fire(IRQ_MAC_DONE)
        irq.fire(IRQ_DMA_DONE)
        irq.clear_all()
        assert irq.pending_mask == 0

    def test_fire_disabled_irq_still_sets_pending(self, irq):
        # Pending is set regardless; handler not called if disabled
        irq.fire(IRQ_MEM_ECC_ERR)
        assert irq.is_pending(IRQ_MEM_ECC_ERR)

    def test_pending_mask_reflects_multiple(self, irq):
        irq.fire(IRQ_MAC_DONE)
        irq.fire(IRQ_THERMAL_WARN)
        expected = (1 << IRQ_MAC_DONE) | (1 << IRQ_THERMAL_WARN)
        assert irq.pending_mask == expected


# ===========================================================================
# 9. Interrupt controller — handler dispatch
# ===========================================================================

class TestInterruptHandlers:

    def test_handler_called_when_enabled(self, irq):
        called = []
        irq.enable(IRQ_MAC_DONE)
        irq.register_handler(IRQ_MAC_DONE, lambda n: called.append(n))
        irq.fire(IRQ_MAC_DONE)
        assert called == [IRQ_MAC_DONE]

    def test_handler_not_called_when_disabled(self, irq):
        called = []
        irq.register_handler(IRQ_DMA_DONE, lambda n: called.append(n))
        irq.fire(IRQ_DMA_DONE)   # not enabled
        assert called == []

    def test_multiple_handlers(self, irq):
        results = []
        irq.enable(IRQ_MAC_DONE)
        irq.register_handler(IRQ_MAC_DONE, lambda n: results.append("A"))
        irq.register_handler(IRQ_MAC_DONE, lambda n: results.append("B"))
        irq.fire(IRQ_MAC_DONE)
        assert results == ["A", "B"]

    def test_unregister_handlers(self, irq):
        called = []
        irq.enable(IRQ_DMS_ALERT)
        irq.register_handler(IRQ_DMS_ALERT, lambda n: called.append(n))
        irq.unregister_handlers(IRQ_DMS_ALERT)
        irq.fire(IRQ_DMS_ALERT)
        assert called == []

    def test_re_enable_fires_handler(self, irq):
        called = []
        irq.enable(IRQ_MAC_DONE)
        irq.disable(IRQ_MAC_DONE)
        irq.register_handler(IRQ_MAC_DONE, lambda n: called.append(n))
        irq.fire(IRQ_MAC_DONE)
        assert called == []
        irq.enable(IRQ_MAC_DONE)
        irq.fire(IRQ_MAC_DONE)
        assert called == [IRQ_MAC_DONE]

    def test_handler_receives_irq_number(self, irq):
        received = []
        irq.enable(IRQ_THERMAL_WARN)
        irq.register_handler(IRQ_THERMAL_WARN, lambda n: received.append(n))
        irq.fire(IRQ_THERMAL_WARN)
        assert received[0] == IRQ_THERMAL_WARN


# ===========================================================================
# 10. Interrupt controller — reset
# ===========================================================================

class TestInterruptReset:

    def test_reset_clears_enable_mask(self, irq):
        irq.enable(IRQ_MAC_DONE)
        irq.reset()
        assert irq.enable_mask == 0

    def test_reset_clears_pending(self, irq):
        irq.fire(IRQ_MAC_DONE)
        irq.reset()
        assert irq.pending_mask == 0

    def test_reset_removes_handlers(self, irq):
        called = []
        irq.enable(IRQ_MAC_DONE)
        irq.register_handler(IRQ_MAC_DONE, lambda n: called.append(n))
        irq.reset()
        irq.enable(IRQ_MAC_DONE)
        irq.fire(IRQ_MAC_DONE)
        assert called == []
