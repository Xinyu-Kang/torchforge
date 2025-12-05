"""
Manual multi-node torchstore RDMA smoke test.

This mirrors the weight-sync pattern that fails in GRPO: storage volumes
live on one host; a reader on another host pulls a large tensor via torchstore
RDMA. It uses the Forge provisioner/Slurm launcher to allocate two hosts.

Usage (manual, opt-in):
    RUN_TORCHSTORE_MULTINODE_RDMA=1 \
    TORCHSTORE_RDMA_CHUNK_SIZE_MB=4 \
    MONARCH_RDMA_TIMEOUT_SECS=60 \
    pytest torchforge/tests/test_torchstore_multinode_rdma.py -k rdma

Environment knobs:
    RUN_TORCHSTORE_MULTINODE_RDMA=1   -> enable the test (otherwise skipped)
    TORCHSTORE_RDMA_CHUNK_SIZE_MB     -> chunk size for torchstore RDMA (default 4MB)
    MONARCH_RDMA_TIMEOUT_SECS         -> RDMA op timeout (default 60s)
    TORCHSTORE_TEST_TENSOR_MB         -> tensor size in MB (default 64)
"""

import os
from pathlib import Path

import pytest
import torch
import torchstore as ts

from forge.controller.provisioner import Provisioner, stop_proc_mesh
from forge.types import Launcher, LauncherConfig, ProvisionerConfig
from monarch.actor import Actor, endpoint, this_host
from monarch.rdma import is_rdma_available


needs_rdma = pytest.mark.skipif(
    not is_rdma_available(), reason="RDMA not available on this host"
)

needs_opt_in = pytest.mark.skipif(
    os.environ.get("RUN_TORCHSTORE_MULTINODE_RDMA") != "1",
    reason="Set RUN_TORCHSTORE_MULTINODE_RDMA=1 to run this test",
)


def _tensor_for_mb(megabytes: int) -> torch.Tensor:
    # int32 tensor sized to requested MB
    num_bytes = megabytes * 1024 * 1024
    num_elems = num_bytes // 4
    return torch.arange(num_elems, dtype=torch.int32)


class _PutGetActor(Actor):
    @endpoint
    async def roundtrip(self, key: str, mb: int) -> int:
        tensor = _tensor_for_mb(mb)
        await ts.put(key, tensor)
        fetched = await ts.get(key)
        return int(fetched.sum().item())


@needs_rdma
@needs_opt_in
@pytest.mark.asyncio
async def test_torchstore_rdma_two_hosts():
    tensor_mb = int(os.environ.get("TORCHSTORE_TEST_TENSOR_MB", "64"))
    repo_root = Path(__file__).resolve().parents[1].parent  # /home/.../forge-learning
    pythonpath = os.pathsep.join(
        [
            str(repo_root),
            str(repo_root / "torchforge"),
            os.environ.get("PYTHONPATH", ""),
        ]
    )
    # Ensure both local and remote procs can import this module.
    os.environ["PYTHONPATH"] = pythonpath

    # Require a Slurm launcher; otherwise skip to avoid accidental local runs.
    cfg = ProvisionerConfig(launcher_config=LauncherConfig(launcher=Launcher.SLURM))
    provisioner = Provisioner(cfg)
    await provisioner.initialize()
    if provisioner.launcher is None:
        pytest.skip("No launcher available for multi-host allocation")

    # Local client/actor is single rank; set LOCAL_RANK so LocalRankStrategy works.
    os.environ.setdefault("LOCAL_RANK", "0")

    # Propagate PYTHONPATH for completeness, though the torchstore actors live in the installed package.
    env_vars = {"PYTHONPATH": pythonpath}
    # Propagate RDMA tuning knobs to remote procs when set.
    for var in (
        "MONARCH_RDMA_DEVICE",
        "MONARCH_RDMA_GID_INDEX",
        "MONARCH_RDMA_TIMEOUT_SECS",
        "TORCHSTORE_RDMA_CHUNK_SIZE_MB",
    ):
        val = os.environ.get(var)
        if val is not None and val != "":
            env_vars[var] = val

    # Allocate a 1-proc storage mesh on a remote host.
    store_procs = await provisioner.get_proc_mesh(
        num_procs=1, num_hosts=1, mesh_name="ts_store", env_vars=env_vars
    )

    # Place storage volumes on the remote host.
    await ts.initialize(mesh=store_procs, strategy=ts.LocalRankStrategy())

    # Local actor that performs put/get; RDMA happens between local proc and remote storage volumes.
    local_procs = this_host().spawn_procs(per_host={"procs": 1})
    putget = local_procs.spawn("ts_putget_actor", _PutGetActor)

    try:
        key = "rdma_test_tensor"
        got_sum = await putget.roundtrip.call_one(key, tensor_mb)
        expected_sum = int(_tensor_for_mb(tensor_mb).sum().item())
        assert got_sum == expected_sum, f"RDMA fetch mismatch: expected {expected_sum}, got {got_sum}"
    finally:
        await ts.shutdown()
        await stop_proc_mesh(local_procs)
        await stop_proc_mesh(store_procs)
