# Fault-Injection Campaign — ecc_secded_bf_10k

**Document ID:** ASTR-FI-ECC_SECDED_BF_10K-V0.1  
**Generated:** 2026-04-21  
**Standard:** ISO 26262-5 §11 + ISO 26262-11 §7  
**Target module:** `rtl/ecc_secded/`  
**Oracle signal:** `tb_ecc_secded_fi.single_err`  
**Expected safe response:** single_err asserts (and corrected data matches original) for any single-bit flip in data_in[0..63]; double_err asserts for any double-bit flip wholly within the data field. Parity-bit single flips for {data[0,1,3,7,15,31,63]} aliasing positions are detected but corrected_data is wrong (F4-D-6 limitation, see ecc_ref.py).


## 1. Aggregate

| Metric | Value |
|---|---:|
| Planned injections | 21 |
| Runs completed | 21 |
| Perturbed an output | 20 |
| Detected by oracle | 20 |
| Missed by oracle | 0 |
| Benign (no output change) | 1 |
| False positives | 0 |
| **Diagnostic coverage** | **100.00 %** |

## 2. Per-injection results (first 50)

| # | Target | Detected? | Cycle | Perturbed? |
|---|---|:---:|---:|:---:|
| 0 | `tb_ecc_secded_fi.data_in_reg` | ✅ | 101 | yes |
| 1 | `tb_ecc_secded_fi.data_in_reg` | ✅ | 101 | yes |
| 2 | `tb_ecc_secded_fi.data_in_reg` | ✅ | 101 | yes |
| 3 | `tb_ecc_secded_fi.data_in_reg` | ✅ | 101 | yes |
| 4 | `tb_ecc_secded_fi.data_in_reg` | ✅ | 101 | yes |
| 5 | `tb_ecc_secded_fi.data_in_reg` | ✅ | 101 | yes |
| 6 | `tb_ecc_secded_fi.data_in_reg` | ✅ | 101 | yes |
| 7 | `tb_ecc_secded_fi.data_in_reg` | ✅ | 101 | yes |
| 8 | `tb_ecc_secded_fi.data_in_reg` | ✅ | 101 | yes |
| 9 | `tb_ecc_secded_fi.parity_in_reg` | ✅ | 201 | yes |
| 10 | `tb_ecc_secded_fi.parity_in_reg` | ✅ | 201 | yes |
| 11 | `tb_ecc_secded_fi.parity_in_reg` | ✅ | 201 | yes |
| 12 | `tb_ecc_secded_fi.parity_in_reg` | ✅ | 201 | yes |
| 13 | `tb_ecc_secded_fi.parity_in_reg` | ✅ | 201 | yes |
| 14 | `tb_ecc_secded_fi.parity_in_reg` | ✅ | 201 | yes |
| 15 | `tb_ecc_secded_fi.parity_in_reg` | ✅ | 201 | yes |
| 16 | `tb_ecc_secded_fi.parity_in_reg` | ✅ | 201 | yes |
| 17 | `tb_ecc_secded_fi.data_in_reg` | ✅ | 301 | yes |
| 18 | `tb_ecc_secded_fi.data_in_reg` | ❌ | — | no |
| 19 | `tb_ecc_secded_fi.data_in_reg` | ✅ | 411 | yes |
| 20 | `tb_ecc_secded_fi.data_in_reg` | ✅ | 501 | yes |

## 3. FMEDA traceability

This campaign validates the diagnostic-coverage assumption for the following failure-mode rows in `tools/safety/failure_modes.yaml`:

- `ecc_secded.data_out.seu`
- `ecc_secded.flag_regs.seu`
- `ecc_secded.parity_out.seu`
- `npu_top.sram_data.seu`

After this report is filed, update the corresponding mechanism `target_dc_pct` in `tools/safety/safety_mechanisms.yaml` to the measured **100.00 %** if it differs from the declared target by more than 5 percentage points (per SEooC §9.1 revision trigger #2).

## 4. Reproduce

```bash
# (WSL Ubuntu 22.04, Verilator 5.030, cocotb 2.0.1)
cd sim/fault_injection
make CAMPAIGN=ecc_secded_bf_10k
python -m tools.safety.fault_injection \
    --campaign campaigns/ecc_secded_bf_10k.yaml \
    --results out/ecc_secded_bf_10k.jsonl \
    --output ../../docs/safety/fault_injection/ecc_secded_bf_10k.md
```

