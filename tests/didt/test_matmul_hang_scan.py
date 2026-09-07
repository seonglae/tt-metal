# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Deployment screening gate for a multicast matmul hang: a heavy matmul that a healthy die
completes and a defective one hangs on.

The workload is the heaviest single matmul in DeepSeek prefill, the MLA wkv_b1 projection: a batched
[1,64,5120,128] @ [1,64,128,512] with a computed MatmulMultiCoreReuseMultiCast1D config over the
full compute grid, 21.5 GMAC on the die under test. It opens one die at a time so a hang names a
card outright: on a whole mesh one wedged die stalls its neighbours and the runtime blames whoever
it noticed, and a mesh-sharded shape would also thin the per-die load below what provokes it. The
defect was observed on a single die of a 32-chip Blackhole galaxy (bh-glx-120-d07u08, chip 19),
where the multicast write acknowledgement never arrives at the barrier and the op never retires.

Directory caveat: tests/didt is the di/dt (current-transient) directory, and this failure is NOT
proven to be di/dt. It does not respond to matmul throttling at all -- it was reproduced at every
TT_MM_THROTTLE_PERF level 0 through 5 (100%, 73%, 67%, 50%, 40% and 33% of throughput) and all six
hung. For contrast, a die that really was di/dt-marginal, on a different host (bh-glx-120-d08u02,
device 21), was cleared by throttle level 1, and its symptom was non-deterministic output rather
than a hang. This one also survives tt-smi -glx_reset_auto, with roughly 44 reset-then-hang cycles
observed, and is rate-independent. All that is proven is that the acknowledgement does not arrive at
the barrier, not which hardware block dropped it. The test lives here because tests/didt is where
single-op hardware-screening tests live, not because di/dt is the mechanism.

On a hang it reports the die's BDF, /dev/tenstorrent minor, tray:slot and ASIC_ID -- a chip id alone
cannot identify a card, since ids are BDF-sorted positions that shift if a chip drops out.

Self-arming: a session fixture sets the dispatch op-to-op gap watchdog and the auto-triage hook the
way CI does, before any device is opened, so a hang is reported and triaged rather than blocking.
Exported values win, so a runner or CI keeps control.

One die, with auto-triage armed:

    TT_VISIBLE_DEVICES=<chip> pytest tests/didt/test_matmul_hang_scan.py -svv

Every die of the host, one process each with its own cold cache, ~280s for 32 dies on 64 cores.
Exits non-zero if any die hangs, so it works as a gate in a script:

    python tests/didt/test_matmul_hang_scan.py

The same from Python, returning the hung chip ids (empty list if the host is clean):

    from tests.didt.test_matmul_hang_scan import scan_all_dies
    hung = scan_all_dies(iters=None, watchdog=60, work_dir=None)

    iters     MATMUL_ITERS for each child; None keeps this test's own default
    watchdog  TT_METAL_OPERATION_TIMEOUT_SECONDS per child, in seconds
    work_dir  holds each die's cache and pytest.log; defaults under the system temp dir

Reset the box with tt-smi -glx_reset_auto before any re-run. This gate detects a hang but cannot
clear one, so a failing die stays wedged, its neighbours then fail too, and a repeat sweep is not
confirmation.

Knobs: MATMUL_HEADS, MATMUL_SEQ, MATMUL_GRID, MATMUL_ITERS, TT_METAL_OPERATION_TIMEOUT_SECONDS.
"""

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import skip_for_wormhole_b0

# In the PCIe bus byte the high nibble selects the tray and the low nibble the slot within it,
# matching the T<tray>:N<slot> label tt-smi prints for the card.
TRAY_OF_BUS_NIBBLE = {0x00: 1, 0x40: 2, 0xC0: 3, 0x80: 4}
WATCHDOG_ENV = "TT_METAL_OPERATION_TIMEOUT_SECONDS"
TRIAGE_ENV = "TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE"


def device_params_for(fabric_config):
    return {
        "fabric_config": fabric_config,
        "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        "l1_small_size": 1152,
    }


@pytest.fixture(scope="session", autouse=True)
def arm_hang_watchdog():
    """Arm the dispatch op-to-op gap watchdog and auto-triage exactly as CI does
    (.github/actions/setup-job/action.yml), before the function-scoped mesh_device opens a device --
    the runtime reads these at device open, and without them a hang blocks forever unreported.
    Anything already exported wins, so a runner or CI keeps control."""
    out_dir = Path(os.environ.get("OUT_DIR", "generated/matmul_hang_scan"))
    out_dir.mkdir(parents=True, exist_ok=True)
    sentinel = out_dir / ".triaged"
    sentinel.unlink(missing_ok=True)
    triage = (
        f"{sys.executable} {os.environ.get('TT_METAL_HOME', Path.cwd())}/tools/triage/triage.py --disable-progress "
        "--run=dump_running_operations --run=dump_op_window --run=dump_callstacks --llm-output"
    )
    os.environ.setdefault(WATCHDOG_ENV, "120")
    os.environ.setdefault("TT_TRIAGE_ENABLE_AGGREGATED_CALLSTACKS", "1")
    # First hang only: triage halts the cores, so anything running after it is on a poisoned device.
    os.environ.setdefault(
        TRIAGE_ENV,
        f"if [ -e {sentinel} ]; then exit 0; fi; touch {sentinel}; {triage} > {out_dir}/triage.log 2>&1",
    )
    logger.info(f"hang watchdog: {WATCHDOG_ENV}={os.environ[WATCHDOG_ENV]}s, triage -> {out_dir}/triage.log")


def chip_identity():
    """chip id -> (bdf, minor). Chip ids are the BDF-sorted position of each /dev/tenstorrent node."""
    minor_to_bdf = {
        int(node.name.split("!")[1]): Path(node, "device").resolve().name
        for node in Path("/sys/class/tenstorrent").glob("tenstorrent!*")
    }
    return dict(enumerate(sorted((bdf, minor) for minor, bdf in minor_to_bdf.items())))


def asic_ids():
    """chip id -> ASIC_ID. tt-smi -s is BDF-sorted, so its index is the chip id."""
    try:
        out = subprocess.run(["tt-smi", "-s"], capture_output=True, text=True, timeout=240).stdout
        devices = json.loads(out)["device_info"]
    except Exception as e:  # a wedged box may refuse telemetry; BDF alone still identifies the card
        logger.warning(f"could not read ASIC_IDs from tt-smi: {e!r}")
        return {}
    ids = {}
    for chip, dev in enumerate(devices):
        hi, lo = (dev.get("smbus_telem", {}).get(k) for k in ("ASIC_ID_HIGH", "ASIC_ID_LOW"))
        if hi and lo:
            ids[chip] = f"0x{int(hi, 16):08x}{int(lo, 16):08x}"
    return ids


def to_physical(runtime_id):
    """TT_VISIBLE_DEVICES renumbers visible chips to 0..N-1 by sorted chip id, so a runtime id is an
    index into that sorted set, not the physical chip. Without the filter the two coincide."""
    visible = os.environ.get("TT_VISIBLE_DEVICES", "").strip()
    if not visible:
        return runtime_id
    chips = sorted(int(v) for v in visible.split(",") if v.strip())
    return chips[runtime_id] if runtime_id < len(chips) else runtime_id


def describe(runtime_id, identity, asics=None):
    chip = to_physical(runtime_id)
    bdf, minor = identity.get(chip, ("?", "?"))
    bus = int(bdf.split(":")[1], 16) if bdf != "?" else None
    slot = f"T{TRAY_OF_BUS_NIBBLE.get(bus & 0xF0, '?')}:N{bus & 0x0F}" if bus is not None else "?"
    asic = f" asic_id={asics[chip]}" if asics and chip in asics else ""
    return f"chip {chip} (runtime id {runtime_id}): bdf={bdf} /dev/tenstorrent/{minor} {slot}{asic}"


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param((1, 1), device_params_for(ttnn.FabricConfig.FABRIC_2D), id="fabric2d-1x1"),
    ],
    indirect=["mesh_device", "device_params"],
)
@skip_for_wormhole_b0("shapes and program config are tuned for Blackhole")
@pytest.mark.timeout(0)
def test_matmul_hang_scan(mesh_device):
    if float(os.environ.get(WATCHDOG_ENV, 0)) <= 0:
        pytest.fail(f"{WATCHDOG_ENV} is non-positive, which disables hang detection entirely")

    grid = mesh_device.compute_with_storage_grid_size()
    gx, gy = (int(v) for v in os.environ.get("MATMUL_GRID", f"{grid.x - 1}x{grid.y}").split("x"))
    heads = int(os.environ.get("MATMUL_HEADS", 64))
    seq = int(os.environ.get("MATMUL_SEQ", 5120))
    m_tiles = seq // 32
    per_core_m = math.ceil(m_tiles / (gx * gy))
    while m_tiles % per_core_m:
        per_core_m += 1

    identity = chip_identity()
    for runtime_id in mesh_device.get_device_ids():
        logger.info(f"participating {describe(runtime_id, identity)}")
    logger.info(f"matmul [1,{heads},{seq},128] @ [1,{heads},128,512] on {gx}x{gy} cores, per_core_M={per_core_m}")

    dram = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    to_mesh = dict(
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=dram,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    in0 = ttnn.from_torch(torch.randn([1, heads, seq, 128]), dtype=ttnn.bfloat16, **to_mesh)
    in1 = ttnn.from_torch(torch.randn([1, heads, 128, 512]), dtype=ttnn.bfloat8_b, **to_mesh)

    try:
        for _ in range(int(os.environ.get("MATMUL_ITERS", 20))):
            out = ttnn.linear(
                in0,
                in1,
                memory_config=dram,
                dtype=ttnn.bfloat16,
                program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(gx, gy),
                    in0_block_w=4,
                    out_subblock_h=1,
                    out_subblock_w=8,
                    per_core_M=per_core_m,
                    per_core_N=16,
                    fuse_batch=False,
                    fused_activation=None,
                    mcast_in0=False,
                ),
                compute_kernel_config=ttnn.init_device_compute_kernel_config(
                    mesh_device.arch(),
                    math_fidelity=ttnn.MathFidelity.HiFi2,
                    math_approx_mode=False,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=True,
                ),
            )
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(out)
    except Exception:
        # The runtime blames a renumbered runtime id, not the physical chip, so name the card here.
        asics = asic_ids()
        for runtime_id in mesh_device.get_device_ids():
            logger.error(f"HUNG {describe(runtime_id, identity, asics)}")
        raise
    logger.success(f"no hang on chips {[to_physical(r) for r in mesh_device.get_device_ids()]}")


def scan_all_dies(iters=None, watchdog=60, work_dir=None):
    """Run this test once per die of the host, one process each with its own cold cache, and report
    which dies hang. Returns the list of hung chip ids.

    Triage is disabled in the children because it halts cores and would disturb the other
    processes; re-run a failing die on its own to triage it. This gate detects a hang but cannot
    clear one, so the failing die stays wedged: reset the box (tt-smi -glx_reset_auto) before any
    re-run, or the wedged die will fail its neighbours and they will look faulty too."""
    work_dir = Path(work_dir or Path(tempfile.gettempdir()) / "matmul_hang_scan")
    shutil.rmtree(work_dir, ignore_errors=True)
    chips = sorted(chip_identity())
    running = {}
    for chip in chips:
        out = work_dir / f"chip{chip}"
        out.mkdir(parents=True)
        env = os.environ | {
            "TT_VISIBLE_DEVICES": str(chip),
            "TT_METAL_CACHE": str(out / "cache"),
            "OUT_DIR": str(out),
            WATCHDOG_ENV: str(watchdog),
            TRIAGE_ENV: "true",
        }
        if iters:
            env["MATMUL_ITERS"] = str(iters)
        log = (out / "pytest.log").open("w")
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--timeout=0", __file__, "-k", "1x1"]
        running[chip] = (subprocess.Popen(cmd, env=env, stdout=log, stderr=log), log)
    logger.info(f"scanning {len(chips)} dies, watchdog={watchdog}s, logs under {work_dir}")

    hung = []
    for chip, (proc, log) in running.items():
        proc.wait()
        log.close()
        text = (work_dir / f"chip{chip}" / "pytest.log").read_text(errors="replace")
        if "1 passed" in text:
            logger.success(f"chip {chip} PASS")
            continue
        # A killed child (Bus error, OOM) writes no diagnostic at all, so report that rather than
        # an empty reason. Ordered by usefulness: only the first names the card, and the runtime's
        # own line says "Device 0" for every die because TT_VISIBLE_DEVICES renumbers it.
        why = "no diagnostic line -- process killed (Bus error / OOM); check dmesg"
        for pattern in (r"HUNG chip \d+.*", r"Device \d+: Timeout[^.]*\.", r"TT_THROW: [^(]*"):
            found = re.search(pattern, text)
            if found:
                why = found.group(0)
                break
        logger.error(f"chip {chip} FAIL: {why}")
        logger.error(f"  cat {work_dir / f'chip{chip}' / 'pytest.log'}")
        hung.append(chip)

    if hung:
        logger.error(f"hung dies: {hung} -- reset before re-running: tt-smi -glx_reset_auto")
    else:
        logger.success(f"all {len(chips)} dies passed")
    return hung


if __name__ == "__main__":
    sys.exit(1 if scan_all_dies() else 0)
