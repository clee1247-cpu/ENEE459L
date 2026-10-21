from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from pathlib import Path
import json

from tensors import (
    SCALE_BYTES,
    ZERO_BYTES,
    Sample,
    Tensor,
    TensorError,
    load_model,
    dtype_bytes,
    elements,
    total_parameters,
    materialise_samples,
)

from quantize_helper import (
    _channel_slice,
    _qmin_qmax,
    _range_of,
    _round_half_up,
    _scale_zero,
    CALIBRATIONS,
    CLIP_METHODS,
    DEFAULT_BITS,
    DEFAULT_PERCENTILE,
    GRANULARITIES,
    MSE_CANDIDATES,
    PRECISIONS,
)


# helper class and functions to carry the quantized tensor and its metadata around together
@dataclass(frozen=True)
class Quantized:
    name: str
    shape: tuple[int, ...]
    codes: tuple[int, ...]
    scales: tuple[float, ...]
    zeros: tuple[int, ...]
    bits: int
    granularity: str
    symmetric: bool
    clipped: int = 0

    @property
    def parameters(self) -> int:
        return elements(self.shape)

    @property
    def channels(self) -> int:
        return self.shape[0]

    @property
    def channel_stride(self) -> int:
        return elements(self.shape[1:]) if len(self.shape) > 1 else 1

    @property
    def scale_count(self) -> int:
        """How many (scale, zero) pairs this tensor carries on disk."""
        return len(self.scales)

    @property
    def clipped_fraction(self) -> float:
        return round(self.clipped / len(self.codes), 8) if self.codes else 0.0

def to_fp16(t: Tensor) -> Tensor:
    out = []
    for v in t.data:
        if not -65504.0 <= v <= 65504.0:
            raise TensorError(
                f"{t.name}: {v} is outside binary16's range; FP16 would store "
                "infinity and every number after this one would be a lie"
            )
        out.append(round(struct.unpack("e", struct.pack("e", float(v)))[0], 8))
    return Tensor(name=t.name, shape=t.shape, data=tuple(out), dtype="fp16")


def as_tensor(name: str, values: Sequence[float]) -> Tensor:
    vals = tuple(float(v) for v in values)
    if not vals:
        raise TensorError(f"{name}: no values")
    return Tensor(name=name, shape=(len(vals),), data=vals, dtype="fp32")


def select_calibration(activations: Sequence[Sample],
                       policy: str = "all") -> tuple[float, ...]:
    acts = tuple(activations)
    if policy == "all":
        chosen = acts
    elif policy == "two_class":
        chosen = tuple(s for s in acts if s.cls in (0, 1))
    elif policy == "first_8":
        chosen = acts[:8]
    elif policy == "one_class":
        chosen = tuple(s for s in acts if s.cls == 0)
    else:
        raise TensorError(
            f"unknown calibration policy {policy!r}; known: {list(CALIBRATIONS)}"
        )
    if not chosen:
        raise TensorError(f"calibration policy {policy!r} selected no samples")
    return tuple(v for s in chosen for v in s.data)


########################################
# start filling in the code from here.
#########################################

def quantize_tensor(t: Tensor, bits: int = DEFAULT_BITS,
                    granularity: str = "per_tensor", symmetric: bool = True,
                    clip: str = "minmax",
                    percentile: float = DEFAULT_PERCENTILE,
                    given_range: tuple[float, float] | None = None) -> Quantized:
    if bits < 2:
        raise TensorError(f"{t.name}: bits must be at least 2")
    if granularity not in GRANULARITIES:
        raise TensorError(f"unknown granularity {granularity!r}; known: {list(GRANULARITIES)}")
    if clip not in CLIP_METHODS:
        raise TensorError(f"unknown clip method {clip!r}; known: {list(CLIP_METHODS)}")

    ranges = []
    if granularity == "per_tensor":
        ranges.append(given_range if given_range is not None else
                      _range_of(t.data, clip, percentile, bits))
    else:
        if given_range is not None:
            ranges = [given_range] * t.channels
        else:
            ranges = [_range_of(t.data[_channel_slice(t, c)[0]:_channel_slice(t, c)[1]],
                                clip, percentile, bits)
                      for c in range(t.channels)]

    scales = []
    zeros = []
    for rmin, rmax in ranges:
        scale, zero = _scale_zero(rmin, rmax, bits, symmetric)
        scales.append(scale)
        zeros.append(zero)

    qmin, qmax = _qmin_qmax(bits, symmetric)
    codes = []
    clipped = 0
    for i, value in enumerate(t.data):
        metadata_index = 0 if granularity == "per_tensor" else i // t.channel_stride
        scale, zero = scales[metadata_index], zeros[metadata_index]
        raw = _round_half_up(float(value) / scale + zero)
        if raw < qmin or raw > qmax:
            clipped += 1
        codes.append(max(qmin, min(qmax, raw)))
    return Quantized(t.name, t.shape, tuple(codes), tuple(scales), tuple(zeros),
                     bits, granularity, symmetric, clipped)


def dequantize(qt: Quantized) -> Tensor:
    if len(qt.codes) != elements(qt.shape):
        raise TensorError(f"{qt.name}: code count does not match shape")
    if len(qt.scales) != len(qt.zeros):
        raise TensorError(f"{qt.name}: scale and zero counts differ")
    expected = 1 if qt.granularity == "per_tensor" else qt.shape[0]
    if qt.granularity not in GRANULARITIES or len(qt.scales) != expected:
        raise TensorError(f"{qt.name}: invalid quantization metadata")
    values = []
    stride = elements(qt.shape[1:]) if len(qt.shape) > 1 else 1
    for i, code in enumerate(qt.codes):
        j = 0 if qt.granularity == "per_tensor" else i // stride
        values.append((int(code) - qt.zeros[j]) * qt.scales[j])
    return Tensor(qt.name, qt.shape, tuple(values), "fp32")



def quantization_error(before: Sequence[Tensor],
                       after: Sequence[Tensor]) -> dict[str, float]:
    if len(before) != len(after):
        raise TensorError("error comparison requires the same number of tensors")
    squared = 0.0
    reference_squared = 0.0
    max_abs = 0.0
    for left, right in zip(before, after):
        if left.shape != right.shape or len(left.data) != len(right.data):
            raise TensorError(f"{left.name}: tensor shapes do not match")
        for original, reconstructed in zip(left.data, right.data):
            difference = float(original) - float(reconstructed)
            squared += difference * difference
            reference_squared += float(original) * float(original)
            max_abs = max(max_abs, abs(difference))
    frobenius = math.sqrt(squared)
    return {
        "frobenius": round(frobenius, 8),
        "relative": round(frobenius / math.sqrt(reference_squared), 8)
        if reference_squared else 0.0,
        "max_abs": round(max_abs, 8),
    }



def bytes_stored(tensors: Sequence[Tensor], precision: str = "fp32",
                 granularity: str = "per_tensor") -> int:
    if precision not in PRECISIONS:
        raise TensorError(f"unknown precision {precision!r}; known: {list(PRECISIONS)}")
    if granularity not in GRANULARITIES:
        raise TensorError(f"unknown granularity {granularity!r}; known: {list(GRANULARITIES)}")
    if precision != "int8":
        return int(sum(t.parameters * dtype_bytes(precision) for t in tensors))
    metadata = sum((1 if granularity == "per_tensor" else t.channels)
                   * (SCALE_BYTES + ZERO_BYTES) for t in tensors)
    return int(sum(t.parameters * dtype_bytes("int8") for t in tensors) + metadata)


def effective_bits(tensors: Sequence[Tensor], precision: str = "fp32",
                   granularity: str = "per_tensor") -> float:
    parameters = total_parameters(tensors)
    if parameters == 0:
        raise TensorError("cannot compute effective bits for no parameters")
    return round(bytes_stored(tensors, precision, granularity) * 8.0 / parameters, 4)


def precision_row(model_name: str, config: Mapping[str, Any],
                  tensors: Sequence[Tensor],
                  calibration_values: Sequence[float],
                  all_values: Sequence[float]) -> dict[str, Any]:
    precision = str(config.get("precision", "fp32"))
    parameters = total_parameters(tensors)
    activation_range = (min(all_values), max(all_values))
    if precision == "fp32":
        weight_after = tuple(tensors)
        activation_after = as_tensor("activations", all_values)
        granularity = "n/a"
        symmetric = None
        calibration = "not calibrated"
        clip = "n/a"
        clipped_fraction = 0.0
        scale_count = 0
    elif precision == "fp16":
        weight_after = tuple(to_fp16(t) for t in tensors)
        activation_after = to_fp16(as_tensor("activations", all_values))
        granularity = "n/a"
        symmetric = None
        calibration = "not calibrated"
        clip = "n/a"
        clipped_fraction = 0.0
        scale_count = 0
    elif precision == "int8":
        granularity = str(config.get("granularity", "per_tensor"))
        symmetric = bool(config.get("symmetric", True))
        clip = str(config.get("clip", "minmax"))
        calibration = str(config.get("calibration", "all"))
        quantized_weights = tuple(
            quantize_tensor(t, bits=8, granularity=granularity, symmetric=symmetric,
                            clip=clip) for t in tensors
        )
        weight_after = tuple(dequantize(qt) for qt in quantized_weights)
        activation_range = _range_of(calibration_values, clip, float(config.get("percentile", DEFAULT_PERCENTILE)))
        activation_q = quantize_tensor(as_tensor("activations", all_values), bits=8,
                                       granularity="per_tensor", symmetric=False,
                                       clip=clip, given_range=activation_range)
        activation_after = dequantize(activation_q)
        clipped_fraction = activation_q.clipped_fraction
        scale_count = sum(qt.scale_count for qt in quantized_weights)
    else:
        raise TensorError(f"unknown precision {precision!r}; known: {list(PRECISIONS)}")

    weight_error = quantization_error(tensors, weight_after)
    activation_error = quantization_error([as_tensor("activations", all_values)], [activation_after])
    return {
        "model": model_name,
        "label": config.get("label", precision),
        "precision": precision,
        "granularity": granularity,
        "symmetric": symmetric,
        "calibration": calibration,
        "clip": clip,
        "parameters": parameters,
        "weight_error": weight_error["relative"],
        "activation_error": activation_error["relative"],
        "clipped_fraction": clipped_fraction,
        "activation_range": [round(float(activation_range[0]), 8), round(float(activation_range[1]), 8)],
        "scale_count": scale_count,
        "bytes_stored": bytes_stored(tensors, precision, granularity if precision == "int8" else "per_tensor"),
        "effective_bits": effective_bits(tensors, precision, granularity if precision == "int8" else "per_tensor"),
    }


def pareto_front(points: Sequence[Mapping[str, Any]],
                 objectives: Mapping[str, float]) -> tuple[int, ...]:
    if any(direction not in (-1.0, 1.0) for direction in objectives.values()):
        raise TensorError("objective directions must be 1.0 or -1.0")
    front = []
    for i, point in enumerate(points):
        dominated = False
        for j, other in enumerate(points):
            if i == j:
                continue
            no_worse = all(direction * other[key] <= direction * point[key]
                           for key, direction in objectives.items())
            strictly_better = any(direction * other[key] < direction * point[key]
                                  for key, direction in objectives.items())
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(i)
    return tuple(front)


if __name__ == "__main__":
    json_path = Path("model.json")

    model_spec = load_model(json_path)
    model_tensors = model_spec["tensors"]
    model_name = model_spec["name"]
    total_params = total_parameters(model_tensors)

    test_tensor = model_tensors[0]  # conv1: shape (20, 3, 3, 3)

    raw_samples = materialise_samples(layer="conv3", width=40, count=32, seed=model_spec["seed"] + 100)
    all_act_vals = tuple(v for s in raw_samples for v in s.data)
    calib_all = select_calibration(raw_samples, policy="all")
    calib_one_cls = select_calibration(raw_samples, policy="one_class")

    # -----------------------------------------------------------------
    # 1. Test quantize_tensor
    # -----------------------------------------------------------------
    print("\n[1/7] Testing quantize_tensor...")
    q_pt = quantize_tensor(test_tensor, bits=8, granularity="per_tensor", symmetric=True)
    q_pc = quantize_tensor(test_tensor, bits=8, granularity="per_channel", symmetric=True)

    assert len(q_pt.codes) == test_tensor.parameters
    assert q_pt.scale_count == 1
    assert q_pc.scale_count == test_tensor.channels
    print(f"  ✓ Per-tensor quantized:  {len(q_pt.codes)} codes, {q_pt.scale_count} scale")
    print(f"  ✓ Per-channel quantized: {len(q_pc.codes)} codes, {q_pc.scale_count} scales")

    # -----------------------------------------------------------------
    # 2. Test dequantize
    # -----------------------------------------------------------------
    print("\n[2/7] Testing dequantize...")
    deq_pt = dequantize(q_pt)
    deq_pc = dequantize(q_pc)

    assert deq_pt.shape == test_tensor.shape
    assert len(deq_pt.data) == len(test_tensor.data)
    assert deq_pc.shape == test_tensor.shape
    print(f"  ✓ Reconstructed tensor shape: {deq_pt.shape}, dtype: {deq_pt.dtype}")

    # -----------------------------------------------------------------
    # 3. Test quantization_error
    # -----------------------------------------------------------------
    print("\n[3/7] Testing quantization_error...")
    err_pt = quantization_error([test_tensor], [deq_pt])
    err_pc = quantization_error([test_tensor], [deq_pc])

    assert err_pt["relative"] >= 0.0 and err_pc["relative"] >= 0.0
    print(f"  ✓ Per-tensor relative error:  {err_pt['relative']:.6f} (max_abs: {err_pt['max_abs']:.6f})")
    print(f"  ✓ Per-channel relative error: {err_pc['relative']:.6f} (max_abs: {err_pc['max_abs']:.6f})")

    # -----------------------------------------------------------------
    # 4. Test bytes_stored
    # -----------------------------------------------------------------
    print("\n[4/7] Testing bytes_stored...")
    bytes_fp32 = bytes_stored(model_tensors, precision="fp32")
    bytes_fp16 = bytes_stored(model_tensors, precision="fp16")
    bytes_int8_pt = bytes_stored(model_tensors, precision="int8", granularity="per_tensor")
    bytes_int8_pc = bytes_stored(model_tensors, precision="int8", granularity="per_channel")

    assert bytes_fp32 > bytes_fp16 > bytes_int8_pt
    assert bytes_int8_pc >= bytes_int8_pt  # per-channel carries scale/zero per output channel
    print(f"  ✓ FP32 bytes:            {bytes_fp32} B")
    print(f"  ✓ FP16 bytes:            {bytes_fp16} B")
    print(f"  ✓ INT8 per-tensor bytes: {bytes_int8_pt} B")
    print(f"  ✓ INT8 per-channel bytes:{bytes_int8_pc} B")

    # -----------------------------------------------------------------
    # 5. Test effective_bits
    # -----------------------------------------------------------------
    print("\n[5/7] Testing effective_bits...")
    eff_fp32 = effective_bits(model_tensors, precision="fp32")
    eff_fp16 = effective_bits(model_tensors, precision="fp16")
    eff_int8_pt = effective_bits(model_tensors, precision="int8", granularity="per_tensor")
    eff_int8_pc = effective_bits(model_tensors, precision="int8", granularity="per_channel")

    assert eff_fp32 == 32.0
    assert eff_fp16 == 16.0
    assert eff_int8_pc > 8.0  # accounts for scales and zeros overhead
    print(f"  ✓ FP32 effective bits:            {eff_fp32:.4f}")
    print(f"  ✓ FP16 effective bits:            {eff_fp16:.4f}")
    print(f"  ✓ INT8 per-tensor effective bits: {eff_int8_pt:.4f}")
    print(f"  ✓ INT8 per-channel effective bits:{eff_int8_pc:.4f}")

    # -----------------------------------------------------------------
    # 6. Test precision_row
    # -----------------------------------------------------------------
    print("\n[6/7] Testing precision_row...")
    configs = [
        {"label": "FP32 Baseline", "precision": "fp32"},
        {"label": "FP16", "precision": "fp16"},
        {
            "label": "INT8 Per-Tensor (All)",
            "precision": "int8",
            "granularity": "per_tensor",
            "symmetric": False,
            "clip": "minmax",
            "calibration": "all",
        },
        {
            "label": "INT8 Per-Channel (All)",
            "precision": "int8",
            "granularity": "per_channel",
            "symmetric": True,
            "clip": "minmax",
            "calibration": "all",
        },
        {
            "label": "INT8 Per-Channel (Class 0 Only)",
            "precision": "int8",
            "granularity": "per_channel",
            "symmetric": True,
            "clip": "minmax",
            "calibration": "one_class",
        },
    ]

    rows = []
    for cfg in configs:
        calib_vals = calib_one_cls if cfg.get("calibration") == "one_class" else calib_all
        row = precision_row(
            model_name=model_name,
            config=cfg,
            tensors=model_tensors,
            calibration_values=calib_vals,
            all_values=all_act_vals,
        )
        rows.append(row)
        print(f"  ✓ {row['label']:<32} | W_Err: {row['weight_error']:<8.6f} | Act_Err: {row['activation_error']:<8.6f} | Bits: {row['effective_bits']:<6.3f}")

    # -----------------------------------------------------------------
    # 7. Test pareto_front & Save Output
    # -----------------------------------------------------------------
    print("\n[7/7] Testing pareto_front...")
    # Minimize effective_bits and minimize weight_error (sense = -1.0)
    objectives = {
        "effective_bits": -1.0,
        "weight_error": -1.0,
    }
    front_indices = pareto_front(rows, objectives)

    assert len(front_indices) > 0
    print(f"  ✓ Pareto optimal candidate indices: {front_indices}")
    for idx in front_indices:
        p = rows[idx]
        print(f"    * Index {idx}: {p['label']} (Bits: {p['effective_bits']}, Weight Err: {p['weight_error']})")

    # Output to JSON
    report_path = Path("quantization_test_results.json")
    results = {
        "model": model_name,
        "total_parameters": total_params,
        "objectives": objectives,
        "pareto_front_indices": list(front_indices),
        "configurations": rows,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Saved] Full output written to {report_path.resolve()}")

