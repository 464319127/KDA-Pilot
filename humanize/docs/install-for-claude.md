# Install KernelPilot Humanize for Claude Code

## Prerequisites

- [codex](https://github.com/openai/codex) -- OpenAI Codex CLI (for review). Verify with `codex --version`.
- `jq` -- JSON processor. Verify with `jq --version`.
- `git` -- Git version control. Verify with `git --version`.

## Option 1: KernelPilot Marketplace (Recommended)

Clone KernelPilot, add the repository root as a Claude Code marketplace, install
the Humanize plugin, and expose the KernelPilot knowledge base as a Claude Code
skill:

```bash
git clone https://github.com/BBuf/kernel-pilot.git
cd kernel-pilot

humanize/scripts/install-skills-claude.sh
```

The installer performs the marketplace install, links `kernel-knowledge`,
installs the query dependency, hydrates Claude Code's installed skill cache with
absolute `HUMANIZE_RUNTIME_ROOT` and `KERNELPILOT_ROOT` paths, and fails if
either placeholder remains. Use the wrapper after manual plugin updates too,
because Claude Code does not hydrate `SKILL.md` placeholders during
`plugin install`.

Manual equivalent:

```bash
claude plugin marketplace add ./
claude plugin install humanize@KernelPilot

mkdir -p ~/.claude/skills
ln -s "$PWD/knowledge" ~/.claude/skills/kernel-knowledge
python3 -m pip install -r knowledge/requirements.txt
humanize/scripts/install-skills-claude.sh --skip-pip
```

Restart Claude Code after installing. If you prefer to run the marketplace
commands inside an existing Claude Code session, the equivalent slash commands
are:

```text
/plugin marketplace add /path/to/kernel-pilot
/plugin install humanize@KernelPilot
```

## Option 2: One-session Local Development

If you have the plugin cloned locally:

```bash
claude --plugin-dir /path/to/kernel-pilot/humanize \
  --add-dir /path/to/kernel-pilot
```

This loads the plugin only for that Claude Code session. Add the knowledge skill
separately if you want `kernel-knowledge` discovery:

```bash
mkdir -p ~/.claude/skills
ln -s /path/to/kernel-pilot/knowledge ~/.claude/skills/kernel-knowledge
```

## Option 3: Upstream Humanize Only

If you only need generic Humanize RLCR and do not need KernelPilot's kernel
loop or knowledge pack, install the upstream Humanize marketplace instead:

```text
/plugin marketplace add PolyArch/humanize
/plugin install humanize@PolyArch
```

That upstream plugin is useful for general implementation loops, but it does
not provide `kernel-knowledge` from this repository.

## Verify Installation

After installing the KernelPilot marketplace, you should see Humanize commands
and the kernel-loop skill:

```text
/humanize:start-rlcr-loop
/humanize:gen-plan
/humanize:refine-plan
/humanize:ask-codex
humanize-kernel-agent-loop
kernel-knowledge
ncu-report
```

You can also inspect the installed plugin from a shell:

```bash
claude plugin list
claude plugin details humanize@KernelPilot
```

## Monitor Setup (Optional)

Add the monitoring helper to your shell for real-time progress tracking:

```bash
# Add to your .bashrc or .zshrc
source ~/.claude/plugins/cache/KernelPilot/humanize/<LATEST.VERSION>/scripts/humanize.sh
```

Then use:

```bash
humanize monitor rlcr   # Monitor RLCR loop
```

## Other Install Guides

- [Install for Codex](install-for-codex.md)
- [Install for Kimi](install-for-kimi.md)

## Next Steps

See the [Usage Guide](usage.md) for detailed command reference and configuration options.
