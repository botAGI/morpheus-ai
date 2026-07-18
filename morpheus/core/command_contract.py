"""Canonical Morpheus CLI command grammar shared by semantic learning."""
import re
import shlex


MORPHEUS_TOP_LEVEL_COMMANDS = frozenset({
    "agent-connect",
    "bootstrap-agent",
    "check",
    "compile",
    "consolidate",
    "diagnostics",
    "eval",
    "handoff",
    "init",
    "integrate",
    "learn",
    "model-smoke",
    "prepare-agent",
    "review",
    "serve",
    "stale",
    "status",
    "train",
    "verify",
    "version",
    "wake",
})
MORPHEUS_NESTED_COMMANDS = {
    "learn": frozenset({
        "activate",
        "benchmark",
        "dataset",
        "eval",
        "lab",
        "list-adapters",
        "quality",
        "rollback",
        "status",
        "team-loop",
        "train",
    }),
    "review": frozenset({
        "accept",
        "accept-batch",
        "accept-proposed",
        "apply",
        "auto-accept",
        "diff",
        "doctor",
        "export-pack",
        "interactive",
        "list",
        "propose",
        "reject",
        "reject-batch",
        "show",
        "suggest-accept",
    }),
}
MORPHEUS_REQUIRED_POSITIONAL_ARITY = {
    ("learn", "activate"): 1,
    ("review", "accept"): 1,
    ("review", "reject"): 1,
    ("review", "show"): 1,
}
MORPHEUS_SHORT_OPTION_ALIASES = {
    "-a": "--all",
    "-f": "--force",
    "-v": "--verbose",
}
MORPHEUS_BOOLEAN_OPTIONS = frozenset({
    "--all",
    "--allow-stale-state",
    "--base-only",
    "--create-training-corrections",
    "--dogfood",
    "--dry-run",
    "--execute",
    "--fail-on-unknown",
    "--fixture-only",
    "--force",
    "--include-corrections",
    "--include-refusals",
    "--json",
    "--lab-only",
    "--list",
    "--local",
    "--no-dry-run",
    "--no-include-corrections",
    "--no-include-refusals",
    "--no-train",
    "--offline",
    "--private",
    "--proposed",
    "--reload",
    "--review",
    "--semantic",
    "--source-backed",
    "--strict",
    "--strict-freshness",
    "--trainable",
    "--ui",
    "--verbose",
    "--version",
    "--yes-i-know-this-can-degrade",
    "--yes-i-know-this-is-legacy-raw-training",
    "--yes-i-know-this-will-train",
})
MORPHEUS_COMMAND_EVAL_CATEGORIES = frozenset({
    "command_cli_capability_claims",
    "commands_and_cli_behavior",
})


CommandRequirement = tuple[
    tuple[str, ...],
    tuple[str, ...],
    frozenset[tuple[str, str | None]],
]


def canonical_command_answer_passes(
    category: str,
    expected: str,
    answer: str,
) -> bool | None:
    """Compare recognized Morpheus commands, or defer non-command text scoring."""
    if category not in MORPHEUS_COMMAND_EVAL_CATEGORIES:
        return None
    expected_commands = _morpheus_command_requirements(expected)
    if not expected_commands:
        return None
    return _command_requirements_satisfied(
        expected_commands,
        _morpheus_command_requirements(answer),
    )


def _morpheus_command_requirements(text: str) -> list[CommandRequirement]:
    requirements = []
    seen = set()
    for tokens, exact_segment in _morpheus_command_token_sequences(text):
        requirement = _canonical_command_requirement(
            tokens,
            exact_segment=exact_segment,
        )
        if requirement is not None and requirement not in seen:
            requirements.append(requirement)
            seen.add(requirement)
    return requirements


def _morpheus_command_token_sequences(
    text: str,
) -> list[tuple[list[str], bool]]:
    sequences: list[tuple[list[str], bool]] = []
    covered: list[tuple[int, int]] = []
    for match in re.finditer(r"`([^`\n]*\bmorpheus\b[^`\n]*)`", text, re.IGNORECASE):
        try:
            tokens = shlex.split(match.group(1))
        except ValueError:
            continue
        morpheus_index = next(
            (
                index
                for index, token in enumerate(tokens)
                if _clean_command_token(token) == "morpheus"
            ),
            None,
        )
        if morpheus_index is not None:
            sequences.append((tokens[morpheus_index:], True))
            covered.append(match.span())

    folded = text.casefold()
    for match in re.finditer(r"\bmorpheus\b", folded):
        if any(start <= match.start() < end for start, end in covered):
            continue
        segment = re.split(r"[`\n;]", folded[match.start():], maxsplit=1)[0]
        tokens = re.findall(
            r"--[a-z0-9-]+(?:=[^\s,;`]+)?|-[a-z]+(?![a-z0-9-])|\.\.?|[a-z0-9][a-z0-9_./:+-]*",
            segment,
        )
        sequences.append((tokens, False))
    return sequences


def _canonical_command_requirement(
    tokens: list[str],
    *,
    exact_segment: bool,
) -> CommandRequirement | None:
    cleaned = [_clean_command_token(token) for token in tokens]
    if len(cleaned) < 2 or cleaned[0] != "morpheus":
        return None
    top_level = cleaned[1]
    if top_level not in MORPHEUS_TOP_LEVEL_COMMANDS:
        return None
    base = ["morpheus", top_level]
    index = 2
    nested = MORPHEUS_NESTED_COMMANDS.get(top_level, frozenset())
    if index < len(cleaned) and cleaned[index] in nested:
        base.append(cleaned[index])
        index += 1

    required_positional_arity = MORPHEUS_REQUIRED_POSITIONAL_ARITY.get(
        tuple(base[1:]),
        0,
    )
    positionals = []
    options: set[tuple[str, str | None]] = set()
    while index < len(cleaned):
        short_options = _expanded_short_options(cleaned[index])
        if short_options:
            options.update((option, None) for option in short_options)
            index += 1
            continue
        token = cleaned[index]
        if token.startswith("--"):
            if "=" in token:
                option, value = token.split("=", 1)
                options.add((option, value or None))
            else:
                value = None
                if (
                    token not in MORPHEUS_BOOLEAN_OPTIONS
                    and index + 1 < len(cleaned)
                    and not cleaned[index + 1].startswith("--")
                ):
                    value = cleaned[index + 1]
                    index += 1
                options.add((token, value))
        elif (
            len(positionals) < required_positional_arity
            or exact_segment
            or _looks_like_bare_command_positional(token)
        ):
            positionals.append(token)
        index += 1
    return tuple(base), tuple(positionals), frozenset(options)


def _expanded_short_options(token: str) -> tuple[str, ...]:
    if token in MORPHEUS_SHORT_OPTION_ALIASES:
        return (MORPHEUS_SHORT_OPTION_ALIASES[token],)
    if not token.startswith("-") or token.startswith("--") or len(token) < 3:
        return ()
    aliases = tuple(f"-{letter}" for letter in token[1:])
    if not all(alias in MORPHEUS_SHORT_OPTION_ALIASES for alias in aliases):
        return ()
    return tuple(MORPHEUS_SHORT_OPTION_ALIASES[alias] for alias in aliases)


def _clean_command_token(token: str) -> str:
    folded = token.casefold().strip().strip("`'\"")
    if folded in {".", ".."}:
        return folded
    return folded.rstrip(",.!?;:)]}")


def _looks_like_bare_command_positional(token: str) -> bool:
    return bool(
        token in {".", ".."}
        or token.startswith(("./", "../", "/"))
        or "/" in token
        or token.endswith((".json", ".jsonl", ".md", ".toml", ".yaml", ".yml"))
    )


def _command_requirements_satisfied(
    expected: list[CommandRequirement],
    answer: list[CommandRequirement],
) -> bool:
    return all(
        any(
            expected_base == answer_base
            and expected_positionals == answer_positionals
            and expected_options <= answer_options
            for answer_base, answer_positionals, answer_options in answer
        )
        for expected_base, expected_positionals, expected_options in expected
    )
