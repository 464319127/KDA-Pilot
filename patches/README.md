# SGLang ↔ kda_kernels patch

Single small patch (`sglang_kda_kernels.patch`) that activates the
kernel-pilot `kda_kernels/` overlay on top of any compatible sglang
checkout.

## Apply

```bash
# 1. Make sure kda_kernels is importable from the python that imports sglang.
export PYTHONPATH=/path/to/kernel-pilot:$PYTHONPATH

# 2. Apply the patch inside the sglang checkout.
cd /path/to/sglang
git apply /path/to/kernel-pilot/patches/sglang_kda_kernels.patch
```

That's it. Any `import sglang` will now run `kda_kernels.install()`
which iterates `kda_kernels._registry.KERNEL_REGISTRY` and swaps in the
promoted KDA-optimized kernels. Functions still on the baseline are
left alone, so partial export is safe.

## Inspect

```python
import sglang
import kda_kernels
print(kda_kernels.status())
# => list of sglang paths whose attribute is now bound to the kda version
```

## Revert

```bash
cd /path/to/sglang
git apply -R /path/to/kernel-pilot/patches/sglang_kda_kernels.patch
```

Or at runtime without touching the sglang checkout:

```python
import kda_kernels
kda_kernels.uninstall()
```

## Compatibility

The patch only touches `python/sglang/__init__.py`. It adds 18 lines
at the end (after the existing `__all__` list). The hunk's context
is the last two lines of `__all__`, so the patch survives upstream
edits to earlier parts of `__init__.py`.

If upstream renames the file, edits the `__all__` block, or moves
the import into a different file, run:

```bash
cd /path/to/kernel-pilot
scripts/export_kda_kernels/export.py --regen-patch
```

which regenerates `patches/sglang_kda_kernels.patch` from the current
sglang HEAD pointed at by `SGLANG_REPO` (default
`/home/sglang-omni/bbuf/repos/sglang`).
