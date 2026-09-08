# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Payload sizing for prefill MoE dispatch/combine.

Keep dimension calculations import-safe: model configs are also imported by host-only
producers. Architecture detection is deferred until the fabric router is configured.
"""


def moe_fabric_payload_size(routed_embedding_size: int, *, bytes_per_element: int = 2) -> int:
    """Bytes needed for one routed token, aligned to a 64-byte DRAM row.

    The shared MoE path sends BF16 rows, even when expert weights use lower precision.
    Latent MoE must pass its routed width, not the model's residual embedding width.
    Use one byte only when both dispatch and combine actually send FP8 rows.
    Dispatch sends routing metadata separately. The separate combine_fabric2d op
    needs additional forwarding space and must request that explicitly.
    """
    if routed_embedding_size <= 0 or bytes_per_element <= 0:
        raise ValueError("Routed embedding size and bytes per element must be positive")
    row_bytes = routed_embedding_size * bytes_per_element
    return (row_bytes + 63) // 64 * 64


def limit_fabric_payload_size(payload_size: int, arch: str) -> int:
    """Cap a requested payload at Metal's architecture-specific fabric limit.

    Limits mirror FabricEriscDatamoverBuilder in
    tt_metal/fabric/erisc_datamover_builder.hpp. Larger rows are fragmented by
    the ordinary dispatch/combine send helpers.
    """
    limits = {"wormhole_b0": 7616, "blackhole": 15232}
    if payload_size <= 0:
        raise ValueError("Fabric payload size must be positive")
    return min(payload_size, limits[arch])


def create_fabric_router_config(max_payload_size: int):
    """Create a router config for the requested bytes on the current architecture."""
    import ttnn

    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = limit_fabric_payload_size(max_payload_size, ttnn.get_arch_name())
    return config
