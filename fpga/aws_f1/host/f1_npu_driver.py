"""Host-side driver for the AstraCore Neo NPU running on AWS F1.

Talks to the CL via AWS's FPGA Management Tools (`fpga-local-cmd`,
`fpga-describe-local-image`) + the kernel's `/sys/class/fpga/`
register interface. Uses the OCL BAR to poke the register file
defined in `cl_npu/design/axi_lite_regfile.sv`.

Reference layout matches the register map in `cl_npu.sv`:

    0x000  cfg_start                 (pulse, self-clears)
    0x004  cfg_k
    0x008  cfg_ai_base
    0x00C  cfg_ao_base
    0x010  cfg_afu_mode
    0x014  cfg_acc_init_mode
    0x018  cfg_precision_mode
    0x01C  status  {30'b0, done, busy}  (READ)
    0x020  dma_start
    ...
    0x040/044  ext_w write path (addr, data)
    0x048/04C  ext_ai write path
    0x050/054  ext_ao read path
    0x100+   cfg_acc_init_data (multi-word)
    0xFF0    device-ID probe: reads "ASTR" (0x41535452)

Requires: AWS FPGA Developer AMI (includes fpga-tools package).
Python 3.8+. numpy.
"""

from __future__ import annotations

import ctypes
import mmap
import os
import struct
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Register offsets (bytes)
REG_CFG_START          = 0x000
REG_CFG_K              = 0x004
REG_CFG_AI_BASE        = 0x008
REG_CFG_AO_BASE        = 0x00C
REG_CFG_AFU_MODE       = 0x010
REG_CFG_ACC_INIT_MODE  = 0x014
REG_CFG_PRECISION      = 0x018
REG_STATUS             = 0x01C
REG_DMA_START          = 0x020
REG_DMA_SRC_ADDR       = 0x024
REG_DMA_AI_BASE        = 0x028
REG_DMA_TILE_H         = 0x02C
REG_DMA_SRC_STRIDE     = 0x030
REG_DMA_STATUS         = 0x034
REG_EXT_W_WADDR        = 0x040
REG_EXT_W_WDATA        = 0x044
REG_EXT_AI_WADDR       = 0x048
REG_EXT_AI_WDATA0      = 0x04C
REG_EXT_AO_RADDR       = 0x050
REG_EXT_AO_RDATA0      = 0x054
REG_EXT_AO_RDATA1      = 0x058
REG_EXT_AO_RDATA2      = 0x05C
REG_EXT_AO_RDATA3      = 0x060
REG_CFG_ACC_INIT_DATA0 = 0x100
REG_DEVICE_ID          = 0xFF0

DEVICE_ID_MAGIC = 0x41535452   # "ASTR"


class F1NpuDriver:
    """Thin wrapper around /sys/class/fpga/ + mmap'd BAR.

    Usage:
        with F1NpuDriver(slot=0) as dev:
            assert dev.device_id() == DEVICE_ID_MAGIC
            dev.load_weight(addr=0, data_byte=42)
            dev.run_tile(k=4, ai_base=0, ao_base=0)
            dev.wait_done()
            vec = dev.read_ao(addr=0, n_cols=16)
    """

    def __init__(self, slot: int = 0, bar: int = 0):
        self.slot = slot
        self.bar = bar
        self._fd: Optional[int] = None
        self._mm: Optional[mmap.mmap] = None

    # ---------- lifecycle ----------
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()

    def open(self) -> None:
        """Open the OCL BAR as a memory-mapped file."""
        # AWS F1 exposes the FPGA via /sys/bus/pci/devices/.../resourceN
        # resource0 is the OCL BAR by default on the cl_dram_dma shell.
        # Users typically use `fpga-describe-local-image` to locate.
        path = self._find_ocl_resource()
        self._fd = os.open(path, os.O_RDWR | os.O_SYNC)
        self._mm = mmap.mmap(self._fd, length=4096,
                              prot=mmap.PROT_READ | mmap.PROT_WRITE,
                              flags=mmap.MAP_SHARED, offset=0)

    def close(self) -> None:
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def _find_ocl_resource(self) -> str:
        """Locate the FPGA slot's OCL BAR sysfs path. On AWS F1 the
        standard location is /sys/bus/pci/devices/.../resource1 for
        the cl_dram_dma / cl_npu shell (slot N)."""
        cmd = ["fpga-describe-local-image", "-S", str(self.slot), "-R"]
        try:
            res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            raise RuntimeError(
                f"fpga-describe-local-image failed: {e}. Is the FPGA "
                f"Developer AMI running and slot {self.slot} loaded?"
            )
        # Parse "PCI Device: 0000:XX:00.0" out of the output and build
        # the resource path. On the Developer AMI, resource1 is OCL.
        for line in res.stdout.splitlines():
            if line.strip().startswith("AFIDEVICE"):
                pci_id = line.split()[-1]
                return f"/sys/bus/pci/devices/{pci_id}/resource1"
        raise RuntimeError("could not parse AFIDEVICE from "
                             "fpga-describe-local-image output")

    # ---------- register I/O ----------
    def write_reg(self, offset: int, value: int) -> None:
        if self._mm is None:
            raise RuntimeError("driver not open; call open() first")
        self._mm[offset:offset + 4] = struct.pack("<I", value & 0xFFFFFFFF)

    def read_reg(self, offset: int) -> int:
        if self._mm is None:
            raise RuntimeError("driver not open; call open() first")
        return struct.unpack("<I", self._mm[offset:offset + 4])[0]

    # ---------- device-level primitives ----------
    def device_id(self) -> int:
        return self.read_reg(REG_DEVICE_ID)

    def load_weight(self, addr: int, data_byte: int) -> None:
        self.write_reg(REG_EXT_W_WDATA, data_byte & 0xFF)
        self.write_reg(REG_EXT_W_WADDR, addr & 0xFFFF)   # pulses we

    def load_activation(self, addr: int, packed: int) -> None:
        """packed: N_ROWS * DATA_W bits, little-endian byte order."""
        # Write data LSWs first — up to 4 × 32-bit words of activation.
        for i in range((packed.bit_length() + 31) // 32 or 1):
            chunk = (packed >> (i * 32)) & 0xFFFFFFFF
            self.write_reg(REG_EXT_AI_WDATA0 + i * 4, chunk)
        self.write_reg(REG_EXT_AI_WADDR, addr & 0xFFFF)  # pulses we

    def run_tile(self, k: int, ai_base: int, ao_base: int, *,
                  afu_mode: int = 0,
                  acc_init_mode: int = 0,
                  acc_init_data: Optional[bytes] = None,
                  precision: int = 0) -> None:
        self.write_reg(REG_CFG_K, k & 0xFFFF)
        self.write_reg(REG_CFG_AI_BASE, ai_base & 0xFFFF)
        self.write_reg(REG_CFG_AO_BASE, ao_base & 0xFFFF)
        self.write_reg(REG_CFG_AFU_MODE, afu_mode & 0x7)
        self.write_reg(REG_CFG_ACC_INIT_MODE, acc_init_mode & 0x1)
        self.write_reg(REG_CFG_PRECISION, precision & 0x3)
        if acc_init_mode == 1 and acc_init_data is not None:
            for i in range(0, len(acc_init_data), 4):
                word = int.from_bytes(acc_init_data[i:i + 4],
                                        "little", signed=False)
                self.write_reg(REG_CFG_ACC_INIT_DATA0 + i, word)
        self.write_reg(REG_CFG_START, 1)     # pulse — HW self-clears

    def wait_done(self, timeout_s: float = 1.0) -> None:
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            status = self.read_reg(REG_STATUS)
            busy, done = status & 1, (status >> 1) & 1
            if done and not busy:
                return
        raise TimeoutError(f"tile did not complete in {timeout_s} s")

    def read_ao(self, addr: int, n_cols: int, acc_w: int = 32) -> np.ndarray:
        self.write_reg(REG_EXT_AO_RADDR, addr & 0xFFFF)  # pulses re
        # Tiny pipe delay before the data registers settle.
        time.sleep(1e-5)
        total_bits = n_cols * acc_w
        words = (total_bits + 31) // 32
        raw = 0
        for i in range(words):
            chunk = self.read_reg(REG_EXT_AO_RDATA0 + i * 4)
            raw |= chunk << (i * 32)
        out = np.empty(n_cols, dtype=np.int32)
        mask = (1 << acc_w) - 1
        signbit = 1 << (acc_w - 1)
        for c in range(n_cols):
            v = (raw >> (c * acc_w)) & mask
            out[c] = v - (1 << acc_w) if v & signbit else v
        return out


# ---------------------------------------------------------------
# Helpers for running a compiled Program (from tools/npu_ref/compiler.py)
# through the FPGA. Mirrors sim/npu_top/test_npu_compiled.py's
# _execute_program but against real hardware.
# ---------------------------------------------------------------
def execute_program(dev: F1NpuDriver, program, *, n_rows: int,
                      n_cols: int, acc_w: int = 32) -> dict:
    """Walk a compiler.Program through the FPGA. Returns a dict of
    AO-address → unpacked-column-vector. Handles the
    ACC_INIT_FROM_PREV_AO sentinel like the cocotb harness."""
    from tools.npu_ref.compiler import (
        ACC_INIT_FROM_PREV_AO, LoadActivation, LoadWeight, ReadAO, RunTile,
    )

    def _pack_signed(vec):
        out = 0
        for i, v in enumerate(vec):
            out |= (int(v) & ((1 << acc_w) - 1)) << (i * acc_w)
        return out

    results = {}
    last_ao = None
    for instr in program:
        if isinstance(instr, LoadWeight):
            dev.load_weight(instr.addr, instr.data)
        elif isinstance(instr, LoadActivation):
            dev.load_activation(instr.addr, instr.packed)
        elif isinstance(instr, RunTile):
            seed = instr.acc_init_data
            seed_bytes = None
            if seed == ACC_INIT_FROM_PREV_AO:
                if last_ao is None:
                    raise RuntimeError("chained RunTile without prior ReadAO")
                packed = _pack_signed(last_ao)
                seed_bytes = packed.to_bytes((n_cols * acc_w + 7) // 8,
                                               "little", signed=False)
            dev.run_tile(k=instr.k, ai_base=instr.ai_base,
                          ao_base=instr.ao_base,
                          afu_mode=instr.afu_mode,
                          acc_init_mode=instr.acc_init_mode,
                          acc_init_data=seed_bytes)
            dev.wait_done()
        elif isinstance(instr, ReadAO):
            vec = dev.read_ao(instr.addr, n_cols, acc_w)
            results[instr.addr] = vec
            last_ao = vec.tolist()
        else:
            raise TypeError(f"unknown instruction {type(instr).__name__}")
    return results


if __name__ == "__main__":
    # Smoke test: open slot 0, probe device-ID register.
    with F1NpuDriver(slot=0) as dev:
        id_ = dev.device_id()
        print(f"device-id: 0x{id_:08x}  {'OK' if id_ == DEVICE_ID_MAGIC else 'MISMATCH!'}")
