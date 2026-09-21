#!/usr/bin/env python3
"""Generate `input.default.toml` from `entity.schema.json`.

The JSON Schema is the single source of truth for the input file: it drives editor
validation/completion (tombi) *and* the annotated reference input that ships with the
code.  This script renders the second from the first, so the two can never drift.

The schema splits into two halves:

  * Standard JSON Schema keywords -- `type`, `enum`, `minimum`, `items`, `required`,
    `default`, ... -- carry everything a validator can check.
  * An `x-entity` object per node carries what JSON Schema cannot express, verbatim from
    the template's comment annotations: `type` (the literal `@type:` string, e.g.
    "array<uint> [size 1 :->: 3]"), `default` (for non-JSON defaults such as
    "1 [no MPI]; MPI_SIZE [MPI]"), `notes`, `examples`, `enum` (an illustrative,
    NON-exhaustive list -- never validated), and `deprecated`.

`x-entity.inferred` on a table lists quantities the code derives rather than reads.  They
are deliberately absent from `properties` (so `additionalProperties: false` rejects them
as input keys) and are emitted here as an `@inferred:` comment block, after that table's
own keys and before its sub-tables.

Layout rules, matching the hand-written template:

  * a table at depth d gets indent 2*d, its keys 2*(d+1)
  * within a table: scalar keys, then the `@inferred:` block, then sub-tables
  * a blank line precedes every sub-table (matching `tombi format`)
  * every key gets a value: its default under `--defaults`, otherwise `""` -- a blank
    form to fill in

Usage:

    python scripts/generate_template.py -d -o input.default.toml   # the reference input
    python scripts/generate_template.py -d                         # ... to stdout
    python scripts/generate_template.py                            # blank form, values ""
    diff <(python scripts/generate_template.py -d) input.default.toml

With `--defaults` every key carries a value instead of `""`: the literal `x-entity.default`
where it is one, else the JSON `default`, else a stand-in derived from the schema's own
constraints (first enum value, `minimum`, `minItems`, ...).  That last case covers the
required keys, which the user must supply anyway, and the handful whose documented
default the code computes at runtime -- `N_GHOSTS`, "1% of the domain size", "box centre
pushed back ~1.7 box-diagonals".  Those are listed on stderr, and their `@default:` or
`@required` annotation still spells out the real behaviour.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import tomllib
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SCHEMA = REPO / "entity.schema.json"

INDENT = "  "

# order of the `@`-annotations inside a key's comment block
ANNOTATION_ORDER = ("required", "type", "default", "deprecated", "enum", "note", "example")


# ---------------------------------------------------------------------------
# schema helpers
# ---------------------------------------------------------------------------


def resolve(node: dict, defs: dict) -> dict:
    """Follow `$ref`, keeping any sibling keywords (2020-12 allows them)."""
    while "$ref" in node:
        target = defs[node["$ref"].rsplit("/", 1)[-1]]
        merged = dict(target)
        merged.update({k: v for k, v in node.items() if k != "$ref"})
        node = merged
    return node


def kind(node: dict, defs: dict) -> str:
    """'table' (TOML table), 'aot' (array of tables), or 'key' (scalar/array value)."""
    if node.get("type") == "object" or "properties" in node:
        return "table"
    items = node.get("items")
    if node.get("type") == "array" and isinstance(items, dict):
        if resolve(items, defs).get("type") == "object":
            return "aot"
    return "key"


def schema_enum(node: dict, defs: dict) -> list | None:
    """First real `enum` reachable through anyOf/oneOf/items (arrays of enums)."""
    if "enum" in node:
        return node["enum"]
    for branch in ("anyOf", "oneOf"):
        for sub in node.get(branch, []):
            found = schema_enum(resolve(sub, defs), defs)
            if found:
                return found
    items = node.get("items")
    if isinstance(items, dict):
        return schema_enum(resolve(items, defs), defs)
    return None


def own_enum(node: dict, defs: dict) -> list | None:
    """The node's own `enum`, including through anyOf/oneOf -- but NOT through `items`.

    Unlike `schema_enum`, this does not descend into array elements: it answers "what
    values may THIS node take", which is what a placeholder needs.
    """
    if "enum" in node:
        return node["enum"]
    for branch in ("anyOf", "oneOf"):
        for sub in node.get(branch, []):
            found = own_enum(resolve(sub, defs), defs)
            if found:
                return found
    return None


def node_type(node: dict, defs: dict) -> str | None:
    """First concrete JSON type of the node, looking into anyOf/oneOf branches."""
    t = node.get("type")
    if isinstance(t, list):
        return t[0]
    if t:
        return t
    for branch in ("anyOf", "oneOf"):
        for sub in node.get(branch, []):
            found = node_type(resolve(sub, defs), defs)
            if found:
                return found
    return None


def placeholder(node: dict, defs: dict) -> Any:
    """A constraint-respecting stand-in for a key the schema gives no default for.

    Used for required keys (which the user must fill in anyway) and for keys whose
    documented default is computed at runtime and so has no literal form -- N_GHOSTS,
    "1% of the domain size", "box center pushed back ~1.7 box-diagonals", ...
    """
    node = resolve(node, defs)

    values = own_enum(node, defs)
    if values:
        return values[0]

    # an anyOf/oneOf node keeps its constraints inside the branches, so pick the first
    # branch that names a type and derive the stand-in from that
    if "type" not in node:
        for branch in ("anyOf", "oneOf"):
            for sub in node.get(branch, []):
                sub = resolve(sub, defs)
                if node_type(sub, defs):
                    return placeholder(sub, defs)

    t = node_type(node, defs)
    if t == "boolean":
        return False
    if t in ("integer", "number"):
        lo = node.get("minimum")
        if lo is None and "exclusiveMinimum" in node:
            lo = node["exclusiveMinimum"] + 1
        value = lo if lo is not None else 0
        hi = node.get("maximum")
        if hi is not None:
            value = min(value, hi)
        return int(value) if t == "integer" else float(value)
    if t == "string":
        return ""
    if t == "array":
        if "prefixItems" in node:
            return [placeholder(i, defs) for i in node["prefixItems"]]
        items = node.get("items")
        count = node.get("minItems", 0)
        if count and isinstance(items, dict):
            return [placeholder(items, defs) for _ in range(count)]
        return []
    if t == "object":
        return {}
    return ""


def derive_type(node: dict, defs: dict) -> str:
    """Fallback `@type` when x-entity.type is absent (it should never be)."""
    t = node.get("type")
    if isinstance(t, list):
        return " | ".join(t)
    if t == "array":
        items = node.get("items")
        inner = derive_type(resolve(items, defs), defs) if isinstance(items, dict) else "any"
        return f"array<{inner}>"
    if t:
        return t
    for branch in ("anyOf", "oneOf"):
        if branch in node:
            return " | ".join(derive_type(resolve(s, defs), defs) for s in node[branch])
    return "any"


# ---------------------------------------------------------------------------
# value / annotation formatting
# ---------------------------------------------------------------------------


def toml_value(value: Any) -> str:
    """Render a JSON default as the TOML literal it corresponds to."""
    if isinstance(value, bool):  # before int -- bool is an int subclass
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    if value is None:
        return '""'
    return str(value)


def as_toml_literal(text: str) -> str | None:
    """Return `text` if it is already a standalone TOML value, else None.

    `x-entity.default` is prose more often than not ("N_GHOSTS", "1 [no MPI]; MPI_SIZE
    [MPI]"), but when it *is* a literal it is the better source than the JSON `default`,
    because it preserves the notation the docs use -- 1e-4 rather than 0.0001.  Anything
    carrying a comment marker is rejected so trailing asides do not leak into the value.
    """
    if "#" in text:
        return None
    try:
        tomllib.loads(f"x = {text}")
    except (tomllib.TOMLDecodeError, ValueError):
        return None
    return text


def format_enum(values: list) -> str:
    """Join enum values for an `@enum:` line.

    An `x-entity.enum` entry that already carries quotes or spaces is documentation prose
    (e.g. the CMasher colormap aside) and is passed through untouched; a bare token is
    quoted so it reads as the literal you would type.
    """
    out = []
    for v in values:
        if isinstance(v, str) and ('"' in v or " " in v):
            out.append(v)
        else:
            out.append(toml_value(v))
    return ", ".join(out)


def wrap(text: str, initial: str, subsequent: str, width: int) -> list[str]:
    """Wrap `text` to `width`, honouring embedded newlines as hard breaks."""
    lines: list[str] = []
    for i, chunk in enumerate(text.split("\n")):
        prefix = initial if i == 0 else subsequent
        chunk = chunk.rstrip()
        if not chunk:
            lines.append(prefix.rstrip())
            continue
        lines.extend(
            textwrap.wrap(
                chunk,
                width=width,
                initial_indent=prefix,
                subsequent_indent=subsequent,
                break_long_words=False,
                break_on_hyphens=False,
            )
            or [prefix.rstrip()]
        )
    return lines


def describe(text: str, indent: str, width: int) -> list[str]:
    """A plain `# ...` description block."""
    return wrap(text, f"{indent}# ", f"{indent}# ", width)


def annotate(tag: str, text: str | None, indent: str, width: int) -> list[str]:
    """A `#   @tag: ...` line, continuations aligned under the text."""
    if text is None:
        return [f"{indent}#   @{tag}"]
    initial = f"{indent}#   @{tag}: "
    return wrap(text, initial, f"{indent}#   " + " " * (len(tag) + 3), width)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


class Renderer:
    def __init__(self, schema: dict, width: int, defaults: bool = False) -> None:
        self.defs = schema.get("$defs", {})
        self.width = width
        self.defaults = defaults
        self.out: list[str] = []
        # keys we had to invent a stand-in for, reported at the end
        self.synthesized: list[str] = []

    def render(self, schema: dict) -> str:
        # the root behaves like a table at depth -1: no keys of its own, and its
        # sub-tables land at depth 0
        self.render_body(schema, "", -1)
        return "\n".join(self.out) + "\n"

    def render_body(self, node: dict, path: str, depth: int) -> None:
        indent = INDENT * (depth + 1)
        props = node.get("properties") or {}
        required = set(node.get("required") or [])

        keys, subtables = [], []
        for name, raw in props.items():
            resolved = resolve(raw, self.defs)
            entry = (name, raw, resolved)
            (keys if kind(resolved, self.defs) == "key" else subtables).append(entry)

        wrote = False
        for name, raw, resolved in keys:
            self.emit_key(
                name, raw, resolved, indent, name in required, f"{path}.{name}" if path else name
            )
            wrote = True

        inferred = (node.get("x-entity") or {}).get("inferred") or []
        if inferred:
            if wrote:
                self.out.append("")
            self.emit_inferred(inferred, indent)
            wrote = True

        for name, raw, resolved in subtables:
            # tombi puts a blank line before every sub-table, including the first one in
            # a parent that has no keys of its own ([radiation] -> [radiation.drag]);
            # `self.out` being non-empty is just "not the very first line of the file"
            if wrote or self.out:
                self.out.append("")
            self.emit_table(name, raw, resolved, path, depth + 1)
            wrote = True

    def key_value(self, name: str, node: dict, path: str, required: bool) -> str:
        """The right-hand side of `name = ...`.

        In template mode every key is an empty string -- a form to fill in.  In defaults
        mode the precedence is: the literal `x-entity.default`, then the JSON `default`,
        then a constraint-derived placeholder (recorded, since it is not a real default).
        """
        if not self.defaults:
            return '""'

        xe = node.get("x-entity") or {}
        documented = xe.get("default")
        if isinstance(documented, str):
            literal = as_toml_literal(documented)
            if literal is not None:
                return literal
        if "default" in node:
            return toml_value(node["default"])

        reason = "required" if required else (documented or "no documented default")
        self.synthesized.append(f"{path} ({reason})")
        return toml_value(placeholder(node, self.defs))

    def emit_key(
        self, name: str, raw: dict, node: dict, indent: str, required: bool, path: str = ""
    ) -> None:
        xe = node.get("x-entity") or {}
        description = raw.get("description") or node.get("description")
        if description:
            self.out += describe(description, indent, self.width)

        if required:
            self.out += annotate("required", None, indent, self.width)

        self.out += annotate(
            "type", xe.get("type") or derive_type(node, self.defs), indent, self.width
        )

        default = xe.get("default")
        if default is None and "default" in node:
            default = toml_value(node["default"])
        if default is not None:
            self.out += annotate("default", default, indent, self.width)

        if xe.get("deprecated"):
            self.out += annotate("deprecated", xe["deprecated"], indent, self.width)

        values = xe.get("enum") or schema_enum(node, self.defs)
        if values:
            self.out += annotate("enum", format_enum(values), indent, self.width)

        for note in xe.get("notes", []):
            self.out += annotate("note", note, indent, self.width)
        for example in xe.get("examples", []):
            self.out += annotate("example", example, indent, self.width)

        value = self.key_value(name, node, path, required)
        self.out.append(f"{indent}{name} = {value}")

    def emit_table(self, name: str, raw: dict, node: dict, parent: str, depth: int) -> None:
        indent = INDENT * depth
        path = f"{parent}.{name}" if parent else name
        is_aot = kind(node, self.defs) == "aot"

        description = raw.get("description") or node.get("description")
        if description:
            self.out += describe(description, indent, self.width)
        for note in (node.get("x-entity") or {}).get("notes", []):
            self.out += annotate("note", note, indent, self.width)

        self.out.append(f"{indent}[[{path}]]" if is_aot else f"{indent}[{path}]")

        body = resolve(node["items"], self.defs) if is_aot else node
        self.render_body(body, path, depth)

    def emit_inferred(self, entries: list[dict], indent: str) -> None:
        self.out.append(f"{indent}# @inferred:")
        cont = f"{indent}#     " + " " * 8
        for entry in entries:
            self.out.append(f"{indent}# - {entry['name']}")
            for tag in ("brief", "type", "enum", "from", "value"):
                if tag not in entry:
                    continue
                text = format_enum(entry[tag]) if tag == "enum" else str(entry[tag])
                self.out += wrap(text, f"{indent}#     @{tag}: ", cont, self.width)


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate the annotated input template from entity.schema.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "schema",
        nargs="?",
        type=Path,
        default=DEFAULT_SCHEMA,
        help=f"JSON Schema to render (default: {DEFAULT_SCHEMA.name} at the repo root)",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="write here instead of stdout",
    )
    ap.add_argument(
        "-d",
        "--defaults",
        action="store_true",
        help="fill each key with its default value instead of an empty string",
    )
    ap.add_argument(
        "-w",
        "--width",
        type=int,
        default=80,
        help="column at which comments wrap (default: 80)",
    )
    args = ap.parse_args()

    try:
        schema = json.loads(args.schema.read_text())
    except FileNotFoundError:
        print(f"error: no such schema: {args.schema}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as err:
        print(f"error: {args.schema} is not valid JSON: {err}", file=sys.stderr)
        return 1

    renderer = Renderer(schema, args.width, defaults=args.defaults)
    text = renderer.render(schema)

    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.write_text(text)
        print(
            f"wrote {args.output} ({text.count(chr(10))} lines) from {args.schema.name}",
            file=sys.stderr,
        )

    if renderer.synthesized:
        print(
            f"note: {len(renderer.synthesized)} key(s) have no literal default; "
            "a constraint-derived stand-in was used (the @default/@required annotation "
            "above each one still documents the real behaviour):",
            file=sys.stderr,
        )
        for entry in renderer.synthesized:
            print(f"  {entry}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
