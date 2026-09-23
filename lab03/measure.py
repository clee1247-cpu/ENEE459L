from __future__ import annotations

import statistics
from typing import Any

from bench import Bench, measured, read_first, read_text, unknown

import json

# A sample is still warm-up while it exceeds the settled rate by this fraction.
WARMUP_TOL = 0.5

# How many samples must sit strictly above a quantile before that quantile is an
# estimate rather than "the biggest number we saw, wearing a hat".
MIN_SAMPLES_ABOVE = 5

# Percentiles the record carries, in the order the schema lists them.
PERCENTILES = (50, 95, 99)

# The widest gap between neighbouring measurements, as a multiple of the typical
# gap, beyond which the sample is treated as coming from two populations.
MULTIMODAL_GAP_RATIO = 20.0

# Neither side of that gap is a mode unless it holds at least this fraction.
MIN_MODE_FRACTION = 0.10

# Below this many retained samples, modality is not a question worth answering.
MIN_SAMPLES_FOR_MODALITY = 20

# How far the last third of a run may drift from the first third, relative to
# the run's own median, before the run is not one population either.
STATIONARITY_TOL = 0.10
MIN_SAMPLES_FOR_STATIONARITY = 12

THERMAL_ZONES = "sys/devices/virtual/thermal"

POWER_RAIL_CANDIDATES = (
    "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/in1_input",
    "sys/bus/i2c/drivers/ina3221/1-0040/iio:device0/in_power0_input",
    "sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
)

GPU_LOAD_CANDIDATES = (
    "sys/devices/platform/gpu.0/load",
    "sys/devices/gpu.0/load",
)

CPUFREQ_MIN = "sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq"
CPUFREQ_MAX = "sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"


# ===========================================================================
# 1. The loop
# ===========================================================================
def run_timed_iterations(bench: Bench, repeats: int = 100) -> list[float]:
    bench.workload.synchronize()
    samples = []
    for _ in range(repeats):
        start = bench.clock()
        bench.workload.run()
        bench.workload.synchronize()
        end = bench.clock()
        samples.append((end - start) / 1_000_000.0)
    return samples


def find_warmup_boundary(samples: list[float]) -> dict[str, Any]:
    source = "leading prefix above (1 + 0.5) x median of the run's second half"
    if len(samples) < 4:
        return unknown(source, "too few samples to estimate a settled rate")

    settled = statistics.median(samples[len(samples) // 2 :])
    if settled <= 0:
        return unknown(source, "settled median is not positive")

    threshold = settled * (1 + WARMUP_TOL)
    discarded = 0
    for sample in samples:
        if sample <= threshold:
            break
        discarded += 1
    return measured(
        discarded,
        source,
        settled_rate_ms=round(settled, 4),
        threshold_ms=round(threshold, 4),
        tolerance=WARMUP_TOL,
        retained=len(samples) - discarded,
    )



def summarize(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return {"n": 0, **{name: None for name in ("mean", "std", "min", "max", "p50", "p95", "p99")}}

    ordered = sorted(samples)
    count = len(ordered)

    def percentile(q: float) -> float:
        position = (count - 1) * q
        lower = int(position)
        upper = min(lower + 1, count - 1)
        return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])

    return {
        "n": count,
        "mean": round(statistics.fmean(ordered), 4),
        "std": round(statistics.stdev(ordered), 4) if count > 2 else 0.0,
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
        "p50": round(percentile(0.50), 4),
        "p95": round(percentile(0.95), 4),
        "p99": round(percentile(0.99), 4),
    }

def is_multimodal(samples: list[float]) -> dict[str, Any]:
    source = "widest trimmed gap >= 20.0x the median gap, with >= 10% of samples on each side"
    if len(samples) < MIN_SAMPLES_FOR_MODALITY:
        return unknown(source, "not enough samples to test for multimodality")

    ordered = sorted(samples)
    trim = int(len(ordered) * 0.05)
    trimmed = ordered[trim : len(ordered) - trim] if trim else ordered
    gaps = [right - left for left, right in zip(trimmed, trimmed[1:])]
    typical_gap = statistics.median(gaps)
    if typical_gap <= 0:
        return unknown(source, "timer resolution is too coarse to measure sample gaps")

    widest_index, widest_gap = max(enumerate(gaps), key=lambda item: item[1])
    split_left = trimmed[widest_index]
    split_right = trimmed[widest_index + 1]
    left = [sample for sample in ordered if sample <= split_left]
    right = [sample for sample in ordered if sample >= split_right]
    ratio = widest_gap / typical_gap
    left_share = len(left) / len(ordered)
    right_share = len(right) / len(ordered)

    return measured(
        ratio >= MULTIMODAL_GAP_RATIO
        and left_share >= MIN_MODE_FRACTION
        and right_share >= MIN_MODE_FRACTION,
        source,
        gap_ratio=round(ratio, 2),
        widest_gap_ms=round(widest_gap, 4),
        typical_gap_ms=round(typical_gap, 5),
        modes=[
            {"n": len(left), "share": round(left_share, 2), "median_ms": round(statistics.median(left), 4)},
            {"n": len(right), "share": round(right_share, 2), "median_ms": round(statistics.median(right), 4)},
        ],
    )

# ===========================================================================
# 7. The clock ceiling the run happened under
# ===========================================================================


def probe_power_state(bench: Bench) -> dict[str, Any]:
    source = "nvpmodel -q"
    result = bench.runner(["nvpmodel", "-q"])
    if not result.ok or result.returncode != 0:
        detail = result.error or f"command exited with status {result.returncode}"
        return unknown(source, f"could not query power mode: {detail}")

    lines = [line.strip() for line in result.stdout.splitlines()]
    mode_name = None
    mode_index = None
    for index, line in enumerate(lines):
        if "NV Power Mode:" in line:
            mode_name = line.split("NV Power Mode:", 1)[1].strip()
            if index + 1 < len(lines):
                try:
                    mode_index = int(lines[index + 1])
                except ValueError:
                    mode_index = None
            break
    if not mode_name or mode_index is None:
        return unknown(source, "power mode output could not be parsed")

    minimum = read_text(bench.telemetry, CPUFREQ_MIN)
    maximum = read_text(bench.telemetry, CPUFREQ_MAX)
    clocks_source = f"{CPUFREQ_MIN} vs {CPUFREQ_MAX}"
    if minimum is None or maximum is None:
        clocks = unknown(clocks_source, "CPU frequency limits could not be read")
    else:
        clocks = measured(
            f"scaling_min_freq={minimum}, scaling_max_freq={maximum}",
            clocks_source,
        )
    return measured(
        mode_name,
        source,
        mode_index=mode_index,
        jetson_clocks=(minimum == maximum) if minimum is not None and maximum is not None else None,
        jetson_clocks_source=clocks,
    )



def probe_telemetry(bench: Bench) -> dict[str, Any]:
    thermal_root = bench.telemetry / THERMAL_ZONES
    temperatures = []
    for temp_path in thermal_root.glob("*/temp"):
        raw = read_text(bench.telemetry, str(temp_path.relative_to(bench.telemetry)))
        if raw is None:
            continue
        try:
            celsius = int(raw) / 1000.0
        except ValueError:
            continue
        if celsius <= -1000:
            continue
        zone_name = read_text(
            bench.telemetry,
            str((temp_path.parent / "type").relative_to(bench.telemetry)),
        ) or temp_path.parent.name
        temperatures.append((celsius, zone_name))

    if temperatures:
        temperature, zone = max(temperatures)
        temperature_record = measured(
            round(temperature, 2),
            f"{THERMAL_ZONES}/*/temp",
            zone=zone,
            zones_read=len(temperatures),
        )
    else:
        temperature_record = unknown(
            f"{THERMAL_ZONES}/*/temp",
            "no readable thermal zones were found",
        )

    power = read_first(bench.telemetry, POWER_RAIL_CANDIDATES)
    if power is None:
        power_record = unknown(
            " | ".join(POWER_RAIL_CANDIDATES),
            "none of the documented INA3221 rail paths could be read",
        )
    else:
        power_path, power_raw = power
        try:
            power_record = measured(int(power_raw), power_path)
        except ValueError:
            power_record = unknown(power_path, "power value is not an integer")

    load = read_first(bench.telemetry, GPU_LOAD_CANDIDATES)
    if load is None:
        load_record = unknown(
            " | ".join(GPU_LOAD_CANDIDATES),
            "none of the documented GPU load paths could be read",
        )
    else:
        load_path, load_raw = load
        try:
            load_record = measured(int(load_raw) / 10.0, load_path, units="per-mille / 10")
        except ValueError:
            load_record = unknown(load_path, "GPU load value is not an integer")

    return {
        "temperature_c": temperature_record,
        "power_mw": power_record,
        "gpu_utilization_percent": load_record,
    }

## for debugging - uncomment the following lines for debugging.
# if __name__ == "__main__":
    # env = Bench.real()
    # out = find_warmup_boundary(samples)
    # print(out)

# for generating system_report.json
if __name__ == "__main__":
    # calling base environment
    env = Bench.real()

    # get your samples
    samples = run_timed_iterations(env, repeats=100)

    # testing measurments and probes
    report = {
        "warmup_boundary": find_warmup_boundary(samples),
        "summarize_setup": summarize(samples),
        "is_multimodal": is_multimodal(samples),
        "probe_power_state": probe_power_state(env),
        "probe_telemetry": probe_telemetry(env),
    }

    # save samples
    path = "samples_analysis.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4)

    # save report
    path = "system_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)