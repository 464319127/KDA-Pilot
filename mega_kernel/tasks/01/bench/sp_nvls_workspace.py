"""Single-process NVLS multimem workspace for the 8-GPU A/B harness.

Builds the same protocol state the flashinfer MNNVL fused-allreduce kernel
expects (see baseline/trtllm_mnnvl_allreduce.cuh and
baseline/upstream_ref/flashinfer/trtllm_mnnvl_ar.py):

  - one multicast object bound to a physical allocation on every device
    (stores through the MC VA fan out to every rank's unicast buffer),
  - per-device unicast VAs of all ranks' buffers (uc_ptrs table),
  - 3 Lamport buffers per rank, pre-filled with the -0.0 bit pattern,
  - per-rank buffer_flags uint32[9]: [current=0, dirty=2, bytesPerBuffer,
    dirtyNumStages=0, bytesToClear[4]=0, accessCounter=0]  (mirrors
    MNNVLAllReduceFusionWorkspace.__init__).

Driver-API port of the flow in baseline/upstream_ref/flashinfer/mnnvl.py
(_alloc_mn_mcast_mem), simplified for one process on one NVLink domain:
no handle export/import is needed because every device lives in this process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch

try:  # cuda-python new layout
    from cuda.bindings import driver as cud
except ImportError:  # legacy layout
    from cuda import cuda as cud


def _ck(ret):
    """Unpack (err, value...) tuples from cuda-python and raise on error."""
    err = ret[0]
    if int(err) != 0:
        _, name = cud.cuGetErrorName(err)
        raise RuntimeError(f"CUDA driver error: {name}")
    vals = ret[1:]
    if len(vals) == 0:
        return None
    if len(vals) == 1:
        return vals[0]
    return vals


def _round_up(v: int, g: int) -> int:
    return (v + g - 1) // g * g


# fp32 -0.0 fill pattern, identical to flashinfer's `tensor.fill_(-0.0)` on a
# float32 view. The kernel polls the Lamport buffer as fp32 lanes (isNegZero on
# float elements), so the fp32 pattern is the correct sentinel; note the bytes
# are NOT two bf16 -0.0 halves (LE halves are 0x0000, 0x8000) — bf16-lane
# polling would be a different protocol and is not what the kernel does.
NEG_ZERO_U32 = 0x8000_0000
NUM_LAMPORT_BUFFERS = 3


@dataclass
class RankView:
    """Per-rank handles handed to an implementation's launch()."""

    rank: int
    world_size: int
    device_index: int
    mc_ptr: int  # this rank's VA of the multicast object
    uc_ptr_local: int  # this rank's own unicast buffer VA
    uc_ptrs_dev: int  # device pointer to int64[world] table of all UC VAs (on this device)
    buffer_flags: torch.Tensor  # uint32[9] on this device


@dataclass
class SpNvlsWorkspace:
    """One multicast object + per-device UC buffers + Lamport flags."""

    world_size: int = 8
    buffer_size_bytes: int = 2 * 1024 * 1024  # per Lamport buffer (>= max row need)

    total_bytes: int = field(init=False)
    ranks: List[RankView] = field(init=False, default_factory=list)

    def __post_init__(self):
        n = self.world_size
        assert torch.cuda.device_count() >= n, "workspace needs all 8 GPUs"
        _ck(cud.cuInit(0))

        # Make sure primary contexts exist (torch owns them).
        for i in range(n):
            torch.cuda.set_device(i)
            torch.cuda.current_stream(i).synchronize()

        devs = [_ck(cud.cuDeviceGet(i)) for i in range(n)]

        # --- granularities -------------------------------------------------
        raw_total = self.buffer_size_bytes * NUM_LAMPORT_BUFFERS

        mc_prop = cud.CUmulticastObjectProp()
        mc_prop.numDevices = n
        mc_prop.handleTypes = cud.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        mc_prop.flags = 0
        mc_prop.size = raw_total  # provisional; re-set after alignment
        # MINIMUM granularity keeps the footprint small (RECOMMENDED is the
        # 512 MB fabric page size on this system — measured 2026-07-10 — which
        # would pin 512 MB per GPU for a <2 MB working set).
        mc_gran = _ck(
            cud.cuMulticastGetGranularity(
                mc_prop, cud.CUmulticastGranularity_flags.CU_MULTICAST_GRANULARITY_MINIMUM
            )
        )

        mem_prop = cud.CUmemAllocationProp()
        mem_prop.type = cud.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        mem_prop.location.type = cud.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        mem_prop.location.id = 0
        mem_prop.requestedHandleTypes = (
            cud.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        )
        mem_gran = _ck(
            cud.cuMemGetAllocationGranularity(
                mem_prop,
                cud.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            )
        )

        self.total_bytes = _round_up(raw_total, max(int(mc_gran), int(mem_gran)))
        # Keep the per-buffer stride the flags advertise consistent with what we
        # actually allocated: floor(total/3) rounded down to 16B (mirrors
        # MNNVLAllReduceFusionWorkspace).
        self.buffer_size_bytes = (self.total_bytes // NUM_LAMPORT_BUFFERS) // 16 * 16
        mc_prop.size = self.total_bytes

        # --- multicast object: create, add every device --------------------
        self._mc_handle = _ck(cud.cuMulticastCreate(mc_prop))
        for d in devs:
            _ck(cud.cuMulticastAddDevice(self._mc_handle, d))

        # --- per-device physical allocations + multicast bind --------------
        self._mem_handles = []
        for i, d in enumerate(devs):
            torch.cuda.set_device(i)
            mem_prop.location.id = i
            h = _ck(cud.cuMemCreate(self.total_bytes, mem_prop, 0))
            self._mem_handles.append(h)
            _ck(cud.cuMulticastBindMem(self._mc_handle, 0, h, 0, self.total_bytes, 0))

        # --- unicast VAs: one VA per physical buffer, visible to all devices
        self._uc_vas = []
        all_dev_access = []
        for i in range(n):
            desc = cud.CUmemAccessDesc()
            desc.location.type = cud.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            desc.location.id = i
            desc.flags = cud.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            all_dev_access.append(desc)
        for i in range(n):
            va = _ck(cud.cuMemAddressReserve(self.total_bytes, max(int(mem_gran), 4096), 0, 0))
            _ck(cud.cuMemMap(va, self.total_bytes, 0, self._mem_handles[i], 0))
            _ck(cud.cuMemSetAccess(va, self.total_bytes, all_dev_access, n))
            self._uc_vas.append(int(va))

        # --- per-device multicast VA (each rank maps the same MC object) ---
        self._mc_vas = []
        for i in range(n):
            va = _ck(cud.cuMemAddressReserve(self.total_bytes, max(int(mc_gran), 4096), 0, 0))
            _ck(cud.cuMemMap(va, self.total_bytes, 0, self._mc_handle, 0))
            desc = all_dev_access[i]
            _ck(cud.cuMemSetAccess(va, self.total_bytes, [desc], 1))
            self._mc_vas.append(int(va))

        # --- per-rank flag tensors + UC pointer tables ---------------------
        self._flags = []
        self._uc_tables = []
        for i in range(n):
            torch.cuda.set_device(i)
            flags = torch.tensor(
                [0, 2, self.buffer_size_bytes, 0, 0, 0, 0, 0, 0],
                dtype=torch.uint32,
                device=f"cuda:{i}",
            )
            table = torch.tensor(self._uc_vas, dtype=torch.int64, device=f"cuda:{i}")
            self._flags.append(flags)
            self._uc_tables.append(table)
            self.ranks.append(
                RankView(
                    rank=i,
                    world_size=n,
                    device_index=i,
                    mc_ptr=self._mc_vas[i],
                    uc_ptr_local=self._uc_vas[i],
                    uc_ptrs_dev=int(table.data_ptr()),
                    buffer_flags=flags,
                )
            )

        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Lamport re-init: fill every UC buffer with -0.0 and reset flags.

        Gives every implementation an identical starting state so A/B runs
        are bit-comparable.
        """
        n = self.world_size
        for i in range(n):
            torch.cuda.set_device(i)
            stream = torch.cuda.current_stream(i).cuda_stream
            _ck(
                cud.cuMemsetD32Async(
                    self._uc_vas[i], NEG_ZERO_U32, self.total_bytes // 4, stream
                )
            )
            self._flags[i].copy_(
                torch.tensor(
                    [0, 2, self.buffer_size_bytes, 0, 0, 0, 0, 0, 0], dtype=torch.uint32
                )
            )
        for i in range(n):
            torch.cuda.synchronize(i)

    # ------------------------------------------------------------------
    def memcpy_into_uc(self, dst_rank: int, dst_byte_off: int, src: torch.Tensor) -> None:
        """Copy a contiguous device tensor into dst_rank's UC buffer.

        Used by the NCU solo-rank mode to pre-feed Lamport slots so a single
        rank's kernel can be profiled without live peers.
        """
        nbytes = src.numel() * src.element_size()
        assert dst_byte_off + nbytes <= self.total_bytes
        _ck(cud.cuMemcpyDtoD(self._uc_vas[dst_rank] + dst_byte_off, src.data_ptr(), nbytes))

    def uc_bytes(self, rank: int) -> torch.Tensor:
        """Debug view: copy this rank's UC buffer to host as uint32."""
        out = torch.empty(self.total_bytes // 4, dtype=torch.uint32)
        _ck(
            cud.cuMemcpyDtoH(
                out.data_ptr(), self._uc_vas[rank], self.total_bytes
            )
        )
        return out

    def destroy(self) -> None:
        # Best-effort teardown; process exit reclaims everything otherwise.
        for va in getattr(self, "_mc_vas", []):
            try:
                cud.cuMemUnmap(va, self.total_bytes)
                cud.cuMemAddressFree(va, self.total_bytes)
            except Exception:
                pass
        for va in getattr(self, "_uc_vas", []):
            try:
                cud.cuMemUnmap(va, self.total_bytes)
                cud.cuMemAddressFree(va, self.total_bytes)
            except Exception:
                pass
        for i, h in enumerate(getattr(self, "_mem_handles", [])):
            try:
                cud.cuMulticastUnbind(self._mc_handle, i, 0, self.total_bytes)
            except Exception:
                pass
            try:
                cud.cuMemRelease(h)
            except Exception:
                pass
