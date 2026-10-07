from __future__ import annotations

from typing import Any

from graph import (
    Graph,
    Layer,
    computed,
    dtype_bytes,
    is_answered,
    unknown,
)


FLOPS_PER_MAC = 2

# The conventions `to_flops` will honour by name. Anything else is unknown
# rather than an assumption, because the whole point of the parameter is that
# the caller has to say which one they mean.
FLOP_CONVENTIONS = {
    "mac_is_two_flops": 2,
    "mac_is_one_flop": 1,
}

# Batch normalisation holds two learnable vectors per channel (scale and shift)
# and two non-learnable ones (running mean and variance). The first pair are
# parameters; the second pair are buffers. Both are in the file.
BN_PARAMS_PER_CHANNEL = 2
BN_BUFFERS_PER_CHANNEL = 2

# Buffers are kept in FP32 even when the weights are not. Halving them saves
# nothing worth having and a denormal running variance is a real failure mode.
BUFFER_DTYPE = "fp32"

# Below this many models there is no line to fit and no residual to report.
MIN_MODELS_FOR_FIT = 3

# Two floats are the same MAC count when they are the same integer. There is no
# tolerance here on purpose: MAC counts are integers, and a tolerance would let
# two genuinely different architectures be reported as tied.
TIE_EXACT = True


# ===========================================================================
# 1. How many numbers are stored
# ===========================================================================

def count_parameters(graph: Graph) -> dict[str, Any]:
  per_layer: dict[str, int] = {}
  total = 0

  for layer in graph:
    parameters = 0
    if layer.kind == "conv":
      if layer.kernel is None or layer.groups < 1:
        return unknown(graph.name, f"cannot determine parameters for {layer.name}")

      channels_in = layer.in_shape[0]
      channels_out = layer.out_shape[0]
      if channels_in % layer.groups or channels_out % layer.groups:
        return unknown(graph.name, f"invalid groups for {layer.name}")

      parameters = (
        channels_out * (channels_in // layer.groups)
        * layer.kernel[0] * layer.kernel[1]
      )
      if layer.bias:
        parameters += channels_out
    elif layer.kind == "linear":
      if len(layer.in_shape) != 1 or len(layer.out_shape) != 1:
        return unknown(graph.name, f"cannot determine parameters for {layer.name}")

      parameters = layer.in_shape[0] * layer.out_shape[0]
      if layer.bias:
        parameters += layer.out_shape[0]
    elif layer.kind == "bn":
      if not layer.out_shape:
        return unknown(graph.name, f"cannot determine parameters for {layer.name}")
      parameters = BN_PARAMS_PER_CHANNEL * layer.out_shape[0]

    per_layer[layer.name] = parameters
    total += parameters

  return computed(
    total,
    f"{graph.name}: {len(graph)} layers, shapes from the description",
    per_layer=per_layer,
    includes_bias=True,
    excludes_bn_buffers=True,
    bn_params_per_channel=BN_PARAMS_PER_CHANNEL,
  )


# ===========================================================================
# 2. What those numbers weigh, which is not the size of the file
# ===========================================================================


def model_size_bytes(graph: Graph) -> dict[str, Any]:
    """Bytes of stored tensors: parameters plus buffers, at their own dtypes.

    Lecture 04 slide 8 gives the formula as `#Parameters × bit width` and slide
    9 spends a page on why the file on disk is not that number. Three reasons,
    two of which this function has to get right:

      * a model is not stored in one dtype. `Layer.weight_dtype` is per layer
        and a network with FP16 weights and FP32 normalisation is completely
        ordinary. Multiplying a single total by a single bit width is the
        mistake, and on these four descriptions it is worth several per cent
      * buffers are in the file. Batch norm's running statistics are two
        vectors per channel that no optimiser ever touched, and they are still
        bytes you have to ship
      * the container is in the file too — the pickle framing, the state-dict
        keys, the archive directory. This function does *not* try to model
        that, and it says so in `container_overhead_excluded` rather than
        quietly letting the caller assume it did

    Returns a `computed` finding whose value is bytes, with the per-dtype
    breakdown that makes the first bullet checkable.
    """
    per_layer: dict[str, float] = {}
    per_dtype: dict[str, float] = {}
    buffer_bytes = 0.0

    for layer in graph:
      layer_graph = Graph(graph.name, graph.input_shape, [layer], graph.precision)
      parameters = count_parameters(layer_graph)
      if not is_answered(parameters):
        return unknown(graph.name, f"cannot determine size for {layer.name}")
      try:
        weight_bytes = parameters["value"] * dtype_bytes(layer.weight_dtype)
      except (KeyError, TypeError):
        return unknown(graph.name, f"unknown weight dtype for {layer.name}")

      layer_bytes = weight_bytes
      if layer.kind == "bn":
        try:
          buffers = BN_BUFFERS_PER_CHANNEL * layer.out_shape[0]
          buffer_bytes += buffers * dtype_bytes(BUFFER_DTYPE)
          layer_bytes += buffers * dtype_bytes(BUFFER_DTYPE)
        except (IndexError, KeyError, TypeError):
          return unknown(graph.name, f"cannot determine buffers for {layer.name}")

      per_layer[layer.name] = layer_bytes
      per_dtype[layer.weight_dtype] = per_dtype.get(layer.weight_dtype, 0.0) + weight_bytes

    per_dtype[BUFFER_DTYPE] = per_dtype.get(BUFFER_DTYPE, 0.0) + buffer_bytes
    
    return computed(
      sum(per_layer.values()),
      f"{graph.name}: per-layer dtypes, buffers at {BUFFER_DTYPE}",
      per_layer=per_layer,
      per_dtype=per_dtype,
      buffer_bytes=buffer_bytes,
      container_overhead_excluded=True,
      note="not the size of the file on disk; see the handout, Stage A step 3",
    )

# ===========================================================================
# 3. The memory nobody puts in the table
# ===========================================================================

def count_activations(graph: Graph) -> dict[str, Any]:
    """Total and peak activation footprint, in elements and in bytes.

    UNC COMP 790-150 Lec 2 p. 70 gives AlexNet as total 932,264 and peak
    440,928, and the two numbers answer two different questions. Total is what
    the whole forward pass produced. Peak is how much had to be resident at
    once, and peak is the one that decides whether the model runs.

    Peak is not `max(out_elements)`. Three things make it larger than that:

      * a layer's input is still resident while its output is being written.
        The live set at layer *i* contains both
      * a tensor consumed by a later layer stays resident in between. `add`
        layers name two inputs in `Layer.reads`, and the earlier one has been
        sitting in memory across every layer of the block. This is the residual
        connection and it is the single largest contributor to peak in
        ResNet-shaped networks
      * the network's own input is a tensor too

    The implementation is a liveness pass: work out the last layer that reads
    each tensor, then walk forward keeping a live set and taking the maximum of
    its total size. Anything simpler than that is wrong on any graph with a
    skip connection, and it is wrong quietly, in the direction that says the
    model fits.

    Returns a `computed` finding whose value is peak *bytes*, because bytes are
    what a memory budget is denominated in, with elements and the layer where
    the peak occurs alongside.
    """
    layers = list(graph)
    consumers: dict[str, list[int]] = {layer.name: [] for layer in layers}
    input_consumers: list[int] = []
    for index, layer in enumerate(layers):
      reads = layer.reads or ((layers[index - 1].name,) if index else ("__input__",))
      for read in reads:
        if read == "__input__":
          input_consumers.append(index)
        elif read in consumers:
          consumers[read].append(index)
        else:
          return unknown(graph.name, f"unknown activation dependency {read!r}")

    last_use = {name: max(indices) for name, indices in consumers.items() if indices}
    input_last_use = max(input_consumers) if input_consumers else 0

    elements = 1
    for dimension in graph.input_shape:
      elements *= dimension

    live: dict[str, tuple[int, float]] = {
      "__input__": (elements, dtype_bytes(graph.precision))
    }
    peak_bytes = 0.0
    peak_elements = 0
    peak_at = "input"
    total_elements = 0
    total_bytes = 0.0

    for index, layer in enumerate(layers):
      output = (layer.out_elements, dtype_bytes(layer.act_dtype))
      live[layer.name] = output
      total_elements += layer.out_elements
      total_bytes += layer.out_elements * output[1]
      current_bytes = sum(count * size for count, size in live.values())
      current_elements = sum(count for count, _ in live.values())
      if current_bytes > peak_bytes:
        peak_bytes = current_bytes
        peak_elements = current_elements
        peak_at = layer.name

      reads = layer.reads or ((layers[index - 1].name,) if index else ("__input__",))
      for read in reads:
        if read == "__input__" and input_last_use == index:
          live.pop(read, None)
        elif read != "__input__" and last_use.get(read) == index:
          live.pop(read, None)

    return computed(
      peak_bytes,
      f"{graph.name}: liveness over {len(graph)} layers, input included",
      peak_at=peak_at,
      peak_elements=peak_elements,
      total_elements=total_elements,
      total_bytes=total_bytes,
      includes_network_input=True,
      note="peak is the resident set, not the largest single tensor",
    )

# ===========================================================================
# 4. The factor of two that halves everybody's numbers
# ===========================================================================

def to_flops(macs: dict[str, Any], convention: str = "mac_is_two_flops") -> dict[str, Any]:
    """Convert a MAC finding to a FLOP finding, naming the convention used.

    A multiply-accumulate is one multiply and one add, so it is two
    floating-point operations. Roughly half the published literature calls a
    MAC one FLOP anyway, and the two conventions differ by exactly the factor
    that makes two papers' numbers incomparable.

    Three requirements, and the third is the graded one:

      * multiply once. `FLOPS_PER_MAC` exists so that the number 2 appears in
        this file exactly once
      * an unknown MAC count converts to an unknown FLOP count. It does not
        convert to zero and it does not raise
      * the convention goes in the finding. A FLOP count that does not say
        which convention produced it is not a FLOP count, it is a number, and
        `to_flops(x, "mac_is_one_flop")` has to be as clearly labelled as the
        default

    An unrecognised convention is `unknown`, not a default. The caller asked
    for something this function does not know how to do.
    """
    if convention not in FLOP_CONVENTIONS:
      return unknown("to_flops", f"unrecognised FLOP convention {convention!r}")
    if not is_answered(macs) or not isinstance(macs.get("value"), (int, float)):
      source = macs.get("source", "to_flops") if isinstance(macs, dict) else "to_flops"
      return unknown(source, "MAC count is unknown")

    scale = FLOP_CONVENTIONS[convention]
    result = computed(
      macs["value"] * scale,
      macs.get("source", "to_flops"),
      convention=convention,
      flops_per_mac=scale,
    )
    if isinstance(macs.get("per_layer"), dict):
      result["per_layer"] = {
        name: value * scale for name, value in macs["per_layer"].items()
      }
    result["note"] = "a count of operations contains no unit of time"
    return result
