#!/usr/bin/env bash
#
# Install/upgrade KernelPilot Humanize for Claude Code.
#
# Claude Code plugin installation copies the plugin into ~/.claude/plugins/cache
# but does not hydrate SKILL.md placeholders. This wrapper performs the normal
# marketplace install, exposes kernel-knowledge as a Claude skill, hydrates the
# installed plugin cache, and verifies the result.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
HUMANIZE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
KERNELPILOT_ROOT="$(cd "$HUMANIZE_ROOT/.." && pwd)"
CLAUDE_BIN="${CLAUDE_BIN:-}"
CLAUDE_CONFIG_DIR="${CLAUDE_CONFIG_DIR:-${HOME}/.claude}"
INSTALL_PIP="true"
DISABLE_POLYARCH="true"
DRY_RUN="false"

usage() {
    cat <<'EOF'
Install KernelPilot Humanize for Claude Code.

Usage:
  humanize/scripts/install-skills-claude.sh [options]

Options:
  --kernelpilot-root PATH     KernelPilot checkout root (default: auto-detect)
  --claude-bin PATH           Claude Code binary (default: claude or ~/.local/bin/claude)
  --claude-config-dir PATH    Claude config dir (default: ~/.claude)
  --skip-pip                  Do not run pip install for knowledge requirements
  --keep-polyarch-enabled     Do not disable an existing humanize@PolyArch install
  --dry-run                   Print actions without writing
  -h, --help                  Show this help
EOF
}

log() {
    printf '[install-claude-skills] %s\n' "$*"
}

die() {
    printf '[install-claude-skills] Error: %s\n' "$*" >&2
    exit 1
}

resolve_claude_bin() {
    if [[ -n "$CLAUDE_BIN" ]]; then
        [[ -x "$CLAUDE_BIN" ]] || die "Claude binary is not executable: $CLAUDE_BIN"
        return
    fi
    if command -v claude >/dev/null 2>&1; then
        CLAUDE_BIN="$(command -v claude)"
        return
    fi
    if [[ -x "${HOME}/.local/bin/claude" ]]; then
        CLAUDE_BIN="${HOME}/.local/bin/claude"
        return
    fi
    die "Claude Code CLI not found; pass --claude-bin PATH"
}

plugin_version() {
    python3 - "$HUMANIZE_ROOT/.claude-plugin/plugin.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
print(json.loads(path.read_text(encoding="utf-8"))["version"])
PY
}

hydrate_plugin_cache() {
    local plugin_root="$1"

    [[ -d "$plugin_root/skills" ]] || die "installed plugin skills not found: $plugin_root/skills"

    if [[ "$DRY_RUN" == "true" ]]; then
        log "DRY-RUN hydrate skills under $plugin_root/skills"
        return
    fi

    python3 - "$plugin_root" "$KERNELPILOT_ROOT" <<'PY'
import pathlib
import sys

plugin_root = pathlib.Path(sys.argv[1])
kernelpilot_root = pathlib.Path(sys.argv[2])

replacements = {
    "{{HUMANIZE_RUNTIME_ROOT}}": str(plugin_root),
    "{{KERNELPILOT_ROOT}}": str(kernelpilot_root),
}

for path in sorted((plugin_root / "skills").glob("*/SKILL.md")):
    text = path.read_text(encoding="utf-8")
    updated = text
    for old, new in replacements.items():
        updated = updated.replace(old, new)
    if updated != text:
        path.write_text(updated, encoding="utf-8")
PY
}

verify_no_placeholders() {
    local plugin_root="$1"

    if grep -R '{{HUMANIZE_RUNTIME_ROOT}}\|{{KERNELPILOT_ROOT}}' "$plugin_root/skills" >/dev/null 2>&1; then
        grep -R -n '{{HUMANIZE_RUNTIME_ROOT}}\|{{KERNELPILOT_ROOT}}' "$plugin_root/skills" >&2 || true
        die "unhydrated placeholders remain in Claude plugin skills"
    fi
}

install_knowledge_skill_link() {
    local skills_dir="$CLAUDE_CONFIG_DIR/skills"
    local link_path="$skills_dir/kernel-knowledge"
    local target="$KERNELPILOT_ROOT/knowledge"

    [[ -f "$target/SKILL.md" ]] || die "KernelPilot knowledge skill not found: $target/SKILL.md"
    [[ -d "$target/evidence/pull-bundles" ]] || die "KernelPilot PR evidence bundles not found: $target/evidence/pull-bundles"

    if [[ "$DRY_RUN" == "true" ]]; then
        log "DRY-RUN link $link_path -> $target"
        return
    fi

    mkdir -p "$skills_dir"
    if [[ -L "$link_path" ]]; then
        rm "$link_path"
    elif [[ -e "$link_path" ]]; then
        die "$link_path exists and is not a symlink; move it aside before reinstalling"
    fi
    ln -s "$target" "$link_path"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --kernelpilot-root)
            [[ -n "${2:-}" ]] || die "--kernelpilot-root requires a value"
            KERNELPILOT_ROOT="$2"
            shift 2
            ;;
        --claude-bin)
            [[ -n "${2:-}" ]] || die "--claude-bin requires a value"
            CLAUDE_BIN="$2"
            shift 2
            ;;
        --claude-config-dir)
            [[ -n "${2:-}" ]] || die "--claude-config-dir requires a value"
            CLAUDE_CONFIG_DIR="$2"
            shift 2
            ;;
        --skip-pip)
            INSTALL_PIP="false"
            shift
            ;;
        --keep-polyarch-enabled)
            DISABLE_POLYARCH="false"
            shift
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

KERNELPILOT_ROOT="$(cd "$KERNELPILOT_ROOT" 2>/dev/null && pwd || true)"
[[ -n "$KERNELPILOT_ROOT" ]] || die "could not resolve KernelPilot root"
[[ -f "$KERNELPILOT_ROOT/.claude-plugin/marketplace.json" ]] || die "KernelPilot marketplace not found: $KERNELPILOT_ROOT/.claude-plugin/marketplace.json"
[[ -f "$KERNELPILOT_ROOT/knowledge/SKILL.md" ]] || die "KernelPilot knowledge skill not found: $KERNELPILOT_ROOT/knowledge/SKILL.md"

resolve_claude_bin
VERSION="$(plugin_version)"
PLUGIN_ROOT="$CLAUDE_CONFIG_DIR/plugins/cache/KernelPilot/humanize/$VERSION"

log "kernelpilot root: $KERNELPILOT_ROOT"
log "claude binary: $CLAUDE_BIN"
log "claude config dir: $CLAUDE_CONFIG_DIR"
log "plugin cache root: $PLUGIN_ROOT"

if [[ "$DRY_RUN" == "true" ]]; then
    log "DRY-RUN claude plugin marketplace add ./"
    log "DRY-RUN claude plugin install humanize@KernelPilot"
else
    (
        cd "$KERNELPILOT_ROOT"
        "$CLAUDE_BIN" plugin marketplace add ./ >/dev/null 2>&1 || "$CLAUDE_BIN" plugin marketplace update KernelPilot
        "$CLAUDE_BIN" plugin install humanize@KernelPilot >/dev/null 2>&1 || "$CLAUDE_BIN" plugin update humanize@KernelPilot
    )
fi

if [[ "$DISABLE_POLYARCH" == "true" && "$DRY_RUN" != "true" ]]; then
    "$CLAUDE_BIN" plugin disable humanize@PolyArch >/dev/null 2>&1 || true
fi

install_knowledge_skill_link

if [[ "$INSTALL_PIP" == "true" ]]; then
    if [[ "$DRY_RUN" == "true" ]]; then
        log "DRY-RUN python3 -m pip install -r $KERNELPILOT_ROOT/knowledge/requirements.txt"
    else
        python3 -m pip install -r "$KERNELPILOT_ROOT/knowledge/requirements.txt"
    fi
fi

hydrate_plugin_cache "$PLUGIN_ROOT"

if [[ "$DRY_RUN" != "true" ]]; then
    verify_no_placeholders "$PLUGIN_ROOT"
    "$CLAUDE_BIN" plugin details humanize@KernelPilot >/dev/null
fi

cat <<EOF

Done.

Claude Code plugin:
  humanize@KernelPilot $VERSION

Hydrated runtime root:
  $PLUGIN_ROOT

Kernel knowledge skill:
  $CLAUDE_CONFIG_DIR/skills/kernel-knowledge -> $KERNELPILOT_ROOT/knowledge
EOF
