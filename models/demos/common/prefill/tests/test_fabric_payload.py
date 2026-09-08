# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Host-only sizing checks; runnable with pytest --noconftest without TTNN."""

import importlib
import sys
from types import SimpleNamespace

import pytest

from models.demos.common.prefill.fabric import (
    create_fabric_router_config,
    limit_fabric_payload_size,
    moe_fabric_payload_size,
)


@pytest.mark.parametrize(
    "module_name,class_name,expected_bytes",
    [
        ("deepseek_v3_config", "DeepSeekV3Config", 14336),
        ("deepseek_v4_flash_config", "DeepSeekV4FlashConfig", 8192),
        ("deepseek_v4_pro_config", "DeepSeekV4ProConfig", 14336),
        ("glm_5_1_config", "GLM51Config", 12288),
        ("glm_5_2_config", "GLM52Config", 12288),
        ("gpt_oss_20b_config", "GptOss20BConfig", 5760),
        ("gpt_oss_120b_config", "GptOss120BConfig", 5760),
        ("kimi_k2_6_config", "KimiK26Config", 14336),
        ("kimi_k2_7_config", "KimiK27Config", 14336),
        ("kimi_k3_config", "KimiK3Config", 7168),
        ("minimax_m2_7_config", "MiniMaxM27Config", 6144),
        ("minimax_m3_config", "MiniMaxM3Config", 12288),
    ],
)
def test_model_payload_fits_one_bf16_routed_row(module_name, class_name, expected_bytes):
    config = getattr(importlib.import_module(f"models.demos.deepseek_v3_d_p.reference.{module_name}"), class_name)
    assert config.FABRIC_PAYLOAD_SIZE == expected_bytes
    assert limit_fabric_payload_size(config.FABRIC_PAYLOAD_SIZE, "blackhole") == expected_bytes


@pytest.mark.parametrize("width,bytes_per_element,expected", [(7168, 1, 7168), (3584, 2, 7168), (33, 2, 128)])
def test_transport_width_and_row_alignment(width, bytes_per_element, expected):
    assert moe_fabric_payload_size(width, bytes_per_element=bytes_per_element) == expected


@pytest.mark.parametrize(
    "arch,requested,expected",
    [
        ("blackhole", 14336, 14336),
        ("blackhole", 12288, 12288),
        ("blackhole", 32768, 15232),
        ("wormhole_b0", 14336, 7616),
        ("wormhole_b0", 12288, 7616),
        ("wormhole_b0", 7168, 7168),
        ("wormhole_b0", 5760, 5760),
    ],
)
def test_router_config_applies_architecture_limit(monkeypatch, arch, requested, expected):
    # Exercise the entry point shared by the runner and operator fixtures without opening hardware.
    monkeypatch.setitem(
        sys.modules,
        "ttnn",
        SimpleNamespace(
            get_arch_name=lambda: arch,
            _ttnn=SimpleNamespace(fabric=SimpleNamespace(FabricRouterConfig=SimpleNamespace)),
        ),
    )
    config = create_fabric_router_config(requested)
    assert config.max_packet_payload_size_bytes == expected
