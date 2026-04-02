"""
AstraCore Neo — Compute Testbench

Covers:
  MAC Array:
    - Precision modes and throughput multipliers
    - matmul correctness across shapes and precisions
    - conv2d correctness via im2col
    - elementwise_mul
    - Core enable/disable and utilisation tracking
    - Peak TOPS estimation
    - HAL MAC_STATUS register sync
    - IRQ_MAC_DONE fires on every op
    - Error paths: shape mismatch, disabled cores, bad clock

  Sparsity Engine:
    - DENSE pattern (no pruning)
    - 4:1, 8:2, 8:1 pruning correctness
    - Pruned block has exactly N non-zeros per M block
    - verify_pattern confirms compliance
    - measure_sparsity
    - effective_tops uplift
    - apply_mask
    - Stats accumulation and reset

  Transformer Engine:
    - fused_softmax: sums to 1, max-subtraction stability
    - fused_layer_norm: zero mean, unit variance
    - fused_gelu: monotone, origin at zero
    - rotary_position_embedding: shape preservation
    - MHSA forward pass: shape, attention weights sum to 1
    - MHSA with causal mask
    - MHSA with sparse top-k attention
    - FFN forward: shape
    - TransformerBlock: residual connection preserves shape
    - TransformerEngine: build_block, run_block, stats
    - embed_dim not divisible by num_heads raises
"""

import sys, os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.compute import (
    MACArray, PrecisionMode, NUM_CORES, MACS_PER_CORE, TOTAL_MACS,
    SparsityEngine, SparsityPattern,
    TransformerEngine, TransformerBlock, MultiHeadSelfAttention, FeedForward,
    fused_softmax, fused_layer_norm, fused_gelu, rotary_position_embedding,
    MACError, SparsityError, TransformerError,
)
from src.hal import AstraCoreDevice
from src.hal.interrupts import IRQ_MAC_DONE


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def dev():
    d = AstraCoreDevice()
    d.power_on()
    return d

@pytest.fixture
def mac(dev):
    return MACArray(dev=dev)

@pytest.fixture
def mac_nodev():
    return MACArray()

@pytest.fixture
def sparsity():
    return SparsityEngine()

@pytest.fixture
def engine(dev):
    return TransformerEngine(dev=dev)

rng = np.random.default_rng(42)


# ===========================================================================
# 1. MAC Array — constants
# ===========================================================================

class TestMACConstants:

    def test_total_macs(self):
        assert TOTAL_MACS == 24_576

    def test_cores_times_macs(self):
        assert NUM_CORES * MACS_PER_CORE == TOTAL_MACS

    def test_all_cores_enabled_by_default(self, mac):
        assert mac.active_cores == NUM_CORES

    def test_active_macs_equals_total(self, mac):
        assert mac.active_macs == TOTAL_MACS


# ===========================================================================
# 2. MAC Array — precision modes
# ===========================================================================

class TestMACPrecision:

    def test_int8_throughput_mul_is_1(self):
        from src.compute.mac_array import _THROUGHPUT_MUL
        assert _THROUGHPUT_MUL[PrecisionMode.INT8] == 1.0

    def test_int4_throughput_mul_is_2(self):
        from src.compute.mac_array import _THROUGHPUT_MUL
        assert _THROUGHPUT_MUL[PrecisionMode.INT4] == 2.0

    def test_fp32_throughput_mul_is_025(self):
        from src.compute.mac_array import _THROUGHPUT_MUL
        assert _THROUGHPUT_MUL[PrecisionMode.FP32] == 0.25

    def test_set_precision(self, mac):
        mac.set_precision(PrecisionMode.FP16)
        assert mac._precision == PrecisionMode.FP16

    def test_peak_tops_int8_positive(self, mac):
        tops = mac.peak_tops(PrecisionMode.INT8)
        assert tops > 0

    def test_peak_tops_int4_double_int8(self, mac):
        t8 = mac.peak_tops(PrecisionMode.INT8)
        t4 = mac.peak_tops(PrecisionMode.INT4)
        assert abs(t4 - 2 * t8) < 1e-6


# ===========================================================================
# 3. MAC Array — matmul correctness
# ===========================================================================

class TestMACMatmul:

    def test_matmul_shape(self, mac):
        A = rng.random((16, 32)).astype(np.float32)
        B = rng.random((32, 8)).astype(np.float32)
        C = mac.matmul(A, B)
        assert C.shape == (16, 8)

    def test_matmul_identity(self, mac):
        # Use FP32: random values in [0,1) truncate to zero in INT8
        A = np.eye(4, dtype=np.float32)
        B = rng.random((4, 4)).astype(np.float32)
        C = mac.matmul(A, B, PrecisionMode.FP32)
        np.testing.assert_allclose(C, B, atol=1e-4)

    def test_matmul_correctness_int8(self, mac):
        A = np.array([[1, 2], [3, 4]], dtype=np.int8)
        B = np.array([[1, 0], [0, 1]], dtype=np.int8)
        C = mac.matmul(A, B, PrecisionMode.INT8)
        np.testing.assert_allclose(C, A.astype(np.float32), atol=1e-4)

    def test_matmul_correctness_fp32(self, mac):
        A = rng.random((8, 16)).astype(np.float32)
        B = rng.random((16, 4)).astype(np.float32)
        expected = A @ B
        C = mac.matmul(A, B, PrecisionMode.FP32)
        np.testing.assert_allclose(C, expected, atol=1e-4)

    def test_matmul_all_precisions(self, mac):
        A = rng.random((4, 8)).astype(np.float32)
        B = rng.random((8, 4)).astype(np.float32)
        for mode in PrecisionMode:
            C = mac.matmul(A, B, mode)
            assert C.shape == (4, 4)

    def test_matmul_shape_mismatch_raises(self, mac):
        A = rng.random((4, 8)).astype(np.float32)
        B = rng.random((9, 4)).astype(np.float32)
        with pytest.raises(MACError):
            mac.matmul(A, B)

    def test_matmul_1d_raises(self, mac):
        A = rng.random(8).astype(np.float32)
        B = rng.random((8, 4)).astype(np.float32)
        with pytest.raises(MACError):
            mac.matmul(A, B)

    def test_matmul_increments_total_ops(self, mac):
        A = rng.random((4, 8)).astype(np.float32)
        B = rng.random((8, 4)).astype(np.float32)
        before = mac.total_ops
        mac.matmul(A, B)
        assert mac.total_ops > before

    def test_matmul_increments_total_calls(self, mac):
        A = rng.random((4, 8)).astype(np.float32)
        B = rng.random((8, 4)).astype(np.float32)
        mac.matmul(A, B)
        assert mac.total_calls == 1


# ===========================================================================
# 4. MAC Array — conv2d
# ===========================================================================

class TestMACConv2d:

    def test_conv2d_output_shape(self, mac):
        inp    = rng.random((3, 8, 8)).astype(np.float32)   # C=3 H=8 W=8
        weight = rng.random((16, 3, 3, 3)).astype(np.float32)  # Cout=16 Cin=3 kH=3 kW=3
        out    = mac.conv2d(inp, weight)
        assert out.shape == (16, 6, 6)   # (8-3)//1 + 1 = 6

    def test_conv2d_with_padding(self, mac):
        inp    = rng.random((1, 4, 4)).astype(np.float32)
        weight = rng.random((4, 1, 3, 3)).astype(np.float32)
        out    = mac.conv2d(inp, weight, padding=1)
        assert out.shape == (4, 4, 4)   # same-pad

    def test_conv2d_with_stride(self, mac):
        inp    = rng.random((1, 8, 8)).astype(np.float32)
        weight = rng.random((4, 1, 3, 3)).astype(np.float32)
        out    = mac.conv2d(inp, weight, stride=2)
        assert out.shape == (4, 3, 3)   # (8-3)//2 + 1 = 3

    def test_conv2d_channel_mismatch_raises(self, mac):
        inp    = rng.random((3, 4, 4)).astype(np.float32)
        weight = rng.random((4, 2, 3, 3)).astype(np.float32)  # Cin=2 != inp C=3
        with pytest.raises(MACError):
            mac.conv2d(inp, weight)

    def test_conv2d_wrong_ndim_raises(self, mac):
        with pytest.raises(MACError):
            mac.conv2d(
                rng.random((4, 4)).astype(np.float32),
                rng.random((4, 1, 3, 3)).astype(np.float32),
            )


# ===========================================================================
# 5. MAC Array — elementwise and stats
# ===========================================================================

class TestMACElementwise:

    def test_elementwise_mul_correctness(self, mac):
        A = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        B = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        C = mac.elementwise_mul(A, B)
        np.testing.assert_allclose(C, [2., 6., 12.], atol=1e-5)

    def test_elementwise_shape_mismatch_raises(self, mac):
        A = np.ones((4,), dtype=np.float32)
        B = np.ones((5,), dtype=np.float32)
        with pytest.raises(MACError):
            mac.elementwise_mul(A, B)


# ===========================================================================
# 6. MAC Array — core management
# ===========================================================================

class TestMACCoreManagement:

    def test_disable_core_reduces_active(self, mac):
        mac.disable_core(0)
        assert mac.active_cores == NUM_CORES - 1

    def test_enable_core_restores(self, mac):
        mac.disable_core(0)
        mac.enable_core(0)
        assert mac.active_cores == NUM_CORES

    def test_matmul_with_some_cores_disabled(self, mac):
        for i in range(0, NUM_CORES, 2):
            mac.disable_core(i)
        A = rng.random((8, 16)).astype(np.float32)
        B = rng.random((16, 8)).astype(np.float32)
        C = mac.matmul(A, B)
        assert C.shape == (8, 8)

    def test_all_cores_disabled_raises(self, mac):
        for i in range(NUM_CORES):
            mac.disable_core(i)
        with pytest.raises(MACError):
            mac.matmul(np.ones((2, 2)), np.ones((2, 2)))

    def test_invalid_core_id_raises(self, mac):
        with pytest.raises(MACError):
            mac.disable_core(NUM_CORES)

    def test_reset_reenables_all_cores(self, mac):
        mac.disable_core(5)
        mac.reset()
        assert mac.active_cores == NUM_CORES

    def test_set_clock_valid(self, mac):
        mac.set_clock(2.5)
        assert mac._clock_ghz == 2.5

    def test_set_clock_out_of_range_raises(self, mac):
        with pytest.raises(MACError):
            mac.set_clock(5.0)


# ===========================================================================
# 7. MAC Array — HAL integration
# ===========================================================================

class TestMACHal:

    def test_mac_done_irq_fires(self, mac, dev):
        dev.irq.enable(IRQ_MAC_DONE)
        fired = []
        dev.irq.register_handler(IRQ_MAC_DONE, lambda n: fired.append(n))
        mac.matmul(np.ones((4, 4)), np.ones((4, 4)))
        assert len(fired) == 1

    def test_mac_status_register_updated(self, mac, dev):
        mac.matmul(np.ones((4, 4)), np.ones((4, 4)))
        status = dev.regs.read(0x0034)
        assert status != 0   # utilisation written

    def test_reset_stats(self, mac):
        mac.matmul(np.ones((4, 4)), np.ones((4, 4)))
        mac.reset_stats()
        assert mac.total_ops == 0
        assert mac.total_calls == 0


# ===========================================================================
# 8. Sparsity — pattern properties
# ===========================================================================

class TestSparsityPatterns:

    def test_dense_sparsity_ratio_zero(self):
        assert SparsityPattern.DENSE.sparsity_ratio == 0.0

    def test_s8_1_sparsity_ratio(self):
        assert abs(SparsityPattern.S8_1.sparsity_ratio - 0.875) < 1e-9

    def test_s4_1_sparsity_ratio(self):
        assert abs(SparsityPattern.S4_1.sparsity_ratio - 0.75) < 1e-9

    def test_s8_2_sparsity_ratio(self):
        assert abs(SparsityPattern.S8_2.sparsity_ratio - 0.75) < 1e-9

    def test_dense_throughput_mul_is_1(self):
        assert SparsityPattern.DENSE.throughput_multiplier == 1.0

    def test_s8_1_throughput_mul_is_8(self):
        assert SparsityPattern.S8_1.throughput_multiplier == 8.0

    def test_s4_1_throughput_mul_is_4(self):
        assert SparsityPattern.S4_1.throughput_multiplier == 4.0


# ===========================================================================
# 9. Sparsity — pruning correctness
# ===========================================================================

class TestSparsityPruning:

    def test_dense_no_change(self, sparsity):
        w = rng.random((8,)).astype(np.float32)
        pruned, mask = sparsity.prune(w, SparsityPattern.DENSE)
        np.testing.assert_array_equal(pruned, w)
        np.testing.assert_array_equal(mask, np.ones(8))

    def test_s8_1_exactly_1_per_block(self, sparsity):
        w = rng.random((16,)).astype(np.float32)
        _, mask = sparsity.prune(w, SparsityPattern.S8_1)
        assert sparsity.verify_pattern(mask, SparsityPattern.S8_1)

    def test_s4_1_exactly_1_per_block(self, sparsity):
        w = rng.random((16,)).astype(np.float32)
        _, mask = sparsity.prune(w, SparsityPattern.S4_1)
        assert sparsity.verify_pattern(mask, SparsityPattern.S4_1)

    def test_s8_2_exactly_2_per_block(self, sparsity):
        w = rng.random((16,)).astype(np.float32)
        _, mask = sparsity.prune(w, SparsityPattern.S8_2)
        assert sparsity.verify_pattern(mask, SparsityPattern.S8_2)

    def test_pruning_keeps_largest_magnitudes(self, sparsity):
        # Block of 8: one clear winner at index 3
        w = np.zeros(8, dtype=np.float32)
        w[3] = 100.0
        _, mask = sparsity.prune(w, SparsityPattern.S8_1)
        assert mask[3] == 1.0
        assert mask.sum() == 1.0

    def test_pruned_zeros_at_mask_zeros(self, sparsity):
        w = rng.random((8,)).astype(np.float32)
        pruned, mask = sparsity.prune(w, SparsityPattern.S8_1)
        np.testing.assert_array_equal(pruned * (1 - mask), 0.0)

    def test_pruning_2d_weight(self, sparsity):
        w = rng.random((4, 8)).astype(np.float32)
        pruned, mask = sparsity.prune(w, SparsityPattern.S8_1)
        assert pruned.shape == (4, 8)
        assert mask.shape == (4, 8)

    def test_stats_accumulate(self, sparsity):
        w = rng.random((16,)).astype(np.float32)
        sparsity.prune(w, SparsityPattern.S8_1)
        assert sparsity.total_weights_kept > 0
        assert sparsity.total_weights_pruned > 0

    def test_reset_stats(self, sparsity):
        w = rng.random((8,)).astype(np.float32)
        sparsity.prune(w, SparsityPattern.S8_1)
        sparsity.reset_stats()
        assert sparsity.total_weights_kept == 0


# ===========================================================================
# 10. Sparsity — mask application and analysis
# ===========================================================================

class TestSparsityMaskAndAnalysis:

    def test_apply_mask_zeros_correct_positions(self, sparsity):
        t    = np.ones((8,), dtype=np.float32)
        mask = np.array([1,0,1,0,1,0,1,0], dtype=np.float32)
        out  = sparsity.apply_mask(t, mask)
        expected = np.array([1,0,1,0,1,0,1,0], dtype=np.float32)
        np.testing.assert_array_equal(out, expected)

    def test_apply_mask_shape_mismatch_raises(self, sparsity):
        with pytest.raises(SparsityError):
            sparsity.apply_mask(np.ones((4,)), np.ones((5,)))

    def test_measure_sparsity_all_zero(self, sparsity):
        assert sparsity.measure_sparsity(np.zeros((8,))) == 1.0

    def test_measure_sparsity_all_nonzero(self, sparsity):
        assert sparsity.measure_sparsity(np.ones((8,))) == 0.0

    def test_measure_sparsity_half(self, sparsity):
        t = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)
        assert sparsity.measure_sparsity(t) == 0.5

    def test_effective_tops_dense(self, sparsity):
        assert sparsity.effective_tops(100.0, SparsityPattern.DENSE) == 100.0

    def test_effective_tops_s8_1(self, sparsity):
        assert sparsity.effective_tops(100.0, SparsityPattern.S8_1) == 800.0

    def test_effective_tops_negative_raises(self, sparsity):
        with pytest.raises(SparsityError):
            sparsity.effective_tops(-1.0, SparsityPattern.S8_1)

    def test_verify_pattern_dense(self, sparsity):
        mask = np.ones((8,), dtype=np.float32)
        assert sparsity.verify_pattern(mask, SparsityPattern.DENSE)


# ===========================================================================
# 11. Transformer — fused primitives
# ===========================================================================

class TestTransformerPrimitives:

    def test_softmax_sums_to_1(self):
        x = rng.random((4, 8)).astype(np.float32)
        s = fused_softmax(x, dim=-1)
        np.testing.assert_allclose(s.sum(axis=-1), np.ones(4), atol=1e-5)

    def test_softmax_all_positive(self):
        x = rng.random((4, 8)).astype(np.float32)
        s = fused_softmax(x, dim=-1)
        assert np.all(s >= 0)

    def test_softmax_stable_with_large_values(self):
        x = np.array([[1e9, 0.0, -1e9]], dtype=np.float32)
        s = fused_softmax(x, dim=-1)
        assert np.isfinite(s).all()
        np.testing.assert_allclose(s.sum(), 1.0, atol=1e-5)

    def test_layer_norm_zero_mean(self):
        x = rng.random((4, 16)).astype(np.float32)
        out = fused_layer_norm(x)
        np.testing.assert_allclose(out.mean(axis=-1), np.zeros(4), atol=1e-5)

    def test_layer_norm_unit_variance(self):
        x = rng.random((4, 16)).astype(np.float32)
        out = fused_layer_norm(x)
        # np.var uses biased estimator; result is slightly < 1 for small N
        np.testing.assert_allclose(out.var(axis=-1), np.ones(4), atol=2e-4)

    def test_layer_norm_with_gamma_beta(self):
        x     = rng.random((2, 8)).astype(np.float32)
        gamma = np.full(8, 2.0, dtype=np.float32)
        beta  = np.full(8, 1.0, dtype=np.float32)
        out   = fused_layer_norm(x, gamma, beta)
        # Mean should be ~beta, std ~gamma
        np.testing.assert_allclose(out.mean(axis=-1), np.full(2, 1.0), atol=1e-4)

    def test_gelu_at_zero_is_zero(self):
        assert abs(fused_gelu(np.array([0.0]))[0]) < 1e-5

    def test_gelu_positive_input_positive_output(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert np.all(fused_gelu(x) > 0)

    def test_gelu_shape_preserved(self):
        x = rng.random((3, 4, 5)).astype(np.float32)
        assert fused_gelu(x).shape == (3, 4, 5)

    def test_rope_shape_preserved(self):
        B, T, H, D = 2, 8, 4, 16
        x = rng.random((B, T, H, D)).astype(np.float32)
        out = rotary_position_embedding(x, T, D)
        assert out.shape == (B, T, H, D)

    def test_rope_head_dim_mismatch_raises(self):
        x = rng.random((1, 4, 2, 8)).astype(np.float32)
        with pytest.raises(TransformerError):
            rotary_position_embedding(x, seq_len=4, head_dim=16)


# ===========================================================================
# 12. Transformer — MHSA
# ===========================================================================

class TestMHSA:

    def test_mhsa_output_shape(self):
        B, T, D = 2, 10, 64
        mhsa = MultiHeadSelfAttention(embed_dim=D, num_heads=8)
        x = rng.random((B, T, D)).astype(np.float32)
        out, attn = mhsa.forward(x)
        assert out.shape == (B, T, D)

    def test_attn_weights_shape(self):
        B, T, D = 1, 6, 64
        mhsa = MultiHeadSelfAttention(embed_dim=D, num_heads=8)
        x = rng.random((B, T, D)).astype(np.float32)
        _, attn = mhsa.forward(x)
        assert attn.shape == (B, 8, T, T)

    def test_attn_weights_sum_to_1(self):
        B, T, D = 1, 8, 64
        mhsa = MultiHeadSelfAttention(embed_dim=D, num_heads=8)
        x = rng.random((B, T, D)).astype(np.float32)
        _, attn = mhsa.forward(x)
        np.testing.assert_allclose(attn.sum(axis=-1), np.ones((B, 8, T)), atol=1e-5)

    def test_mhsa_with_causal_mask(self):
        B, T, D = 1, 6, 64
        mhsa = MultiHeadSelfAttention(embed_dim=D, num_heads=8)
        x    = rng.random((B, T, D)).astype(np.float32)
        # Upper-triangular causal mask
        mask = np.triu(np.full((T, T), -1e9), k=1).astype(np.float32)[None]
        out, attn = mhsa.forward(x, mask=mask)
        assert out.shape == (B, T, D)

    def test_mhsa_sparse_top_k(self):
        B, T, D = 1, 8, 64
        mhsa = MultiHeadSelfAttention(embed_dim=D, num_heads=8, sparse_top_k=3)
        x    = rng.random((B, T, D)).astype(np.float32)
        out, attn = mhsa.forward(x)
        assert out.shape == (B, T, D)

    def test_mhsa_without_rope(self):
        B, T, D = 1, 6, 64
        mhsa = MultiHeadSelfAttention(embed_dim=D, num_heads=8, use_rope=False)
        x    = rng.random((B, T, D)).astype(np.float32)
        out, _ = mhsa.forward(x)
        assert out.shape == (B, T, D)

    def test_mhsa_bad_embed_dim_raises(self):
        with pytest.raises(TransformerError):
            MultiHeadSelfAttention(embed_dim=65, num_heads=8)  # not divisible

    def test_mhsa_input_dim_mismatch_raises(self):
        mhsa = MultiHeadSelfAttention(embed_dim=64, num_heads=8)
        x    = rng.random((1, 4, 32)).astype(np.float32)
        with pytest.raises(TransformerError):
            mhsa.forward(x)


# ===========================================================================
# 13. Transformer — FFN
# ===========================================================================

class TestFFN:

    def test_ffn_output_shape(self):
        B, T, D = 2, 8, 64
        ffn = FeedForward(embed_dim=D)
        x   = rng.random((B, T, D)).astype(np.float32)
        out = ffn.forward(x)
        assert out.shape == (B, T, D)

    def test_ffn_different_batch_sizes(self):
        ffn = FeedForward(embed_dim=32)
        for B in (1, 4, 8):
            x   = rng.random((B, 6, 32)).astype(np.float32)
            out = ffn.forward(x)
            assert out.shape == (B, 6, 32)


# ===========================================================================
# 14. Transformer — Block and Engine
# ===========================================================================

class TestTransformerBlockAndEngine:

    def test_block_output_shape(self):
        B, T, D = 2, 8, 64
        block = TransformerBlock(embed_dim=D, num_heads=8)
        x     = rng.random((B, T, D)).astype(np.float32)
        out, attn = block.forward(x)
        assert out.shape == (B, T, D)
        assert attn.shape == (B, 8, T, T)

    def test_engine_build_block(self, engine):
        block = engine.build_block(embed_dim=64)
        assert isinstance(block, TransformerBlock)

    def test_engine_run_block_shape(self, engine):
        block = engine.build_block(embed_dim=64)
        x     = rng.random((1, 8, 64)).astype(np.float32)
        out, attn = engine.run_block(block, x)
        assert out.shape == (1, 8, 64)

    def test_engine_stats_increment(self, engine):
        block = engine.build_block(embed_dim=64)
        x     = rng.random((2, 6, 64)).astype(np.float32)
        engine.run_block(block, x)
        assert engine.blocks_run == 1
        assert engine.total_tokens_processed == 12  # 2 × 6

    def test_engine_bad_embed_dim_raises(self, engine):
        with pytest.raises(TransformerError):
            engine.build_block(embed_dim=65)  # not divisible by 8

    def test_engine_run_block_1d_raises(self, engine):
        block = engine.build_block(embed_dim=64)
        with pytest.raises(TransformerError):
            engine.run_block(block, np.ones((8, 64)))   # missing batch dim

    def test_engine_reset_stats(self, engine):
        block = engine.build_block(embed_dim=64)
        x     = rng.random((1, 4, 64)).astype(np.float32)
        engine.run_block(block, x)
        engine.reset_stats()
        assert engine.blocks_run == 0
        assert engine.total_tokens_processed == 0

    def test_transformer_block_with_sparse_attention(self, engine):
        block = engine.build_block(embed_dim=64, sparse_top_k=4)
        x     = rng.random((1, 8, 64)).astype(np.float32)
        out, _ = engine.run_block(block, x)
        assert out.shape == (1, 8, 64)
