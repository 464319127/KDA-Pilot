"""Auto-loaded by Python when on sys.path; install diffusion shape capture."""

try:
    import kernel_shape_capture  # noqa: F401
except Exception as exc:
    import sys

    print(f"[shape-capture] failed to install: {exc!r}", file=sys.stderr)
