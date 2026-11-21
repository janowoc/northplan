# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The limitations register and the code that cites it point at each other correctly.

``docs/limitations.md`` numbers its entries ``L1``, ``L2`` and so on, and the
code cites them. This file owns four checks over the tracked tree:

1. Every ``L<n>`` cited outside the register names a live entry.
2. Every "Lives in" pointer in the register resolves: the path is a tracked
   file or a directory with tracked files under it, and a ``::symbol`` is
   defined in that Python file.
3. Every file a "Lives in" sentence names cites its own entry back.
4. Every entry is cited somewhere, unless it is in ``UNCITED_ALLOWED``.

Each check is a pure function over in-memory inputs (the register text, the set
of tracked paths, the file contents by path); the tests over the real tree call
them on the repository and the synthetic tests call them on made-up inputs.

Tracked files are enumerated with ``git ls-files``, as in
``tests/test_licence_headers.py``, not by walking the filesystem. This file is
not scanned, so its synthetic cases may spell numbers the register does not
define; the register itself and ``docs/archive/`` are not scanned either.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTER = "docs/limitations.md"
THIS_FILE = "tests/test_limitation_citations.py"

SCANNED_SUFFIXES = frozenset({".py", ".yaml", ".yml", ".md", ".js", ".html", ".toml"})

#: Entries for something not modelled at all, so no code exists to cite them. Each
#: number has a reason; an entry that gains a citation must leave this set, and the
#: stale-entry check enforces it.
UNCITED_ALLOWED = frozenset(
    {
        13,  # alternative minimum tax: not modelled
        29,  # FHSA: not modelled
    }
)

_ENTRY_START = re.compile(r"^\*\*L(\d+)\.")
_CITATION = re.compile(r"\bL(\d+)\b")
_LIVES_IN = re.compile(r"Lives in (.*?)(?:\.(?=\s|$)|$)")
_BACKTICKED = re.compile(r"`([^`]+)`")
_POINTER = re.compile(
    r"(?P<path>[\w./{},-]*\.(?:py|yaml|yml|md|js|html|toml)|[\w./-]+/)(?:::(?P<sym>[\w.]+))?"
)
_BRACES = re.compile(r"\{([^{}]*)\}")


def parse_entries(register_text: str) -> dict[int, str]:
    """Entry number to entry text. An entry ends at the next entry or a ``#`` heading."""
    entries: dict[int, list[str]] = {}
    current: int | None = None
    for line in register_text.splitlines():
        start = _ENTRY_START.match(line)
        if start:
            current = int(start.group(1))
            entries[current] = [line]
        elif line.startswith("#"):
            current = None
        elif current is not None:
            entries[current].append(line)
    return {n: "\n".join(lines) for n, lines in entries.items()}


def citations(files: dict[str, str]) -> list[tuple[str, int, int]]:
    """Every ``(path, line number, n)`` for an ``L<n>`` in the given files."""
    found = []
    for path, text in files.items():
        for lineno, line in enumerate(text.splitlines(), start=1):
            for match in _CITATION.finditer(line):
                found.append((path, lineno, int(match.group(1))))
    return found


def check_citations_defined(defined: set[int], files: dict[str, str]) -> list[str]:
    """Problems for each citation of a number the register does not define."""
    return [
        f"{path}:{lineno}: L{n} is not a defined entry"
        for path, lineno, n in citations(files)
        if n not in defined
    ]


def lives_in_sentences(entry_text: str) -> list[str]:
    """Each "Lives in" sentence of an entry, whitespace normalised, without its full stop."""
    flat = re.sub(r"\s+", " ", entry_text)
    return [match.group(1) for match in _LIVES_IN.finditer(flat)]


def _expand_braces(path: str) -> list[str]:
    match = _BRACES.search(path)
    if match is None:
        return [path]
    head, tail = path[: match.start()], path[match.end() :]
    return [head + alt + tail for alt in match.group(1).split(",")]


def pointers(sentence: str) -> list[tuple[str, str | None]]:
    """Resolved ``(path, symbol)`` pointers of a sentence, in order.

    A backticked span that is not a path is skipped. A bare file name resolves
    against the directory of the last path with a slash in the same sentence,
    or the repo root if there is none; a brace group expands to one path each.
    """
    found: list[tuple[str, str | None]] = []
    current_dir = ""
    for span in _BACKTICKED.findall(sentence):
        match = _POINTER.fullmatch(span)
        if match is None:
            continue
        symbol = match.group("sym")
        for raw in _expand_braces(match.group("path")):
            if "/" in raw:
                path = raw
                current_dir = raw if raw.endswith("/") else raw.rsplit("/", 1)[0] + "/"
            else:
                path = current_dir + raw
            found.append((path, symbol))
    return found


def _exists(path: str, tracked: set[str]) -> bool:
    if path.endswith("/"):
        return any(t.startswith(path) for t in tracked)
    return path in tracked


def _symbol_problem(tree: ast.Module, symbol: str) -> str | None:
    """Why ``symbol`` (dotted) is not defined in the module, or ``None`` if it is."""
    body: list[ast.stmt] = tree.body
    parts = symbol.split(".")
    for index, name in enumerate(parts):
        matches = [node for node in body if name in _defined_names(node)]
        if not matches:
            return f"{name!r} is not defined"
        if index == len(parts) - 1:
            return None
        classes = [node for node in matches if isinstance(node, ast.ClassDef)]
        if not classes:
            return f"{name!r} is not a class, so {parts[index + 1]!r} cannot be inside it"
        body = classes[0].body
    return None


def _defined_names(node: ast.stmt) -> list[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    return []


def check_pointers(entries: dict[int, str], tracked: set[str], files: dict[str, str]) -> list[str]:
    """Problems for each "Lives in" pointer that names no tracked path or no symbol."""
    problems = []
    for n, text in sorted(entries.items()):
        for sentence in lives_in_sentences(text):
            for path, symbol in pointers(sentence):
                token = path if symbol is None else f"{path}::{symbol}"
                if not _exists(path, tracked):
                    problems.append(f"L{n}: {token}: path is not tracked")
                elif symbol is not None:
                    problems.extend(_symbol_problems(n, token, path, symbol, files))
    return problems


def _symbol_problems(
    n: int, token: str, path: str, symbol: str, files: dict[str, str]
) -> list[str]:
    if not path.endswith(".py"):
        return [f"L{n}: {token}: symbol on a non-Python file"]
    if path not in files:
        return [f"L{n}: {token}: file was not read"]
    try:
        tree = ast.parse(files[path])
    except SyntaxError as error:
        return [f"L{n}: {token}: file does not parse ({error})"]
    why = _symbol_problem(tree, symbol)
    return [] if why is None else [f"L{n}: {token}: {why}"]


def check_back_citations(
    entries: dict[int, str], tracked: set[str], files: dict[str, str]
) -> list[str]:
    """Problems for each file or directory a "Lives in" sentence names.

    A file is a problem when it never cites its entry's ``L<n>``; a directory, when no
    file under it does.
    """
    problems = []
    for n, text in sorted(entries.items()):
        cite = re.compile(rf"\bL{n}\b")
        for sentence in lives_in_sentences(text):
            for path, _symbol in pointers(sentence):
                if not _exists(path, tracked):
                    continue
                if path.endswith("/"):
                    under = [p for p in files if p.startswith(path)]
                    if not any(cite.search(files[p]) for p in under):
                        problems.append(f"L{n}: no file under {path} cites it")
                elif path not in files or not cite.search(files[path]):
                    problems.append(f"L{n}: {path} does not cite it")
    return problems


def check_every_entry_cited(
    defined: set[int], files: dict[str, str], allowed: frozenset[int]
) -> list[str]:
    """Problems for an uncited entry, a stale allow-list number, or an undefined one."""
    cited = {n for _path, _lineno, n in citations(files)}
    problems = [f"L{n} is cited nowhere" for n in sorted(defined - allowed - cited)]
    problems += [f"L{n} is allow-listed but cited" for n in sorted(allowed & cited)]
    problems += [f"L{n} is allow-listed but not defined" for n in sorted(allowed - defined)]
    return problems


# --- the real tree ---------------------------------------------------------------------------


def _tracked_paths() -> set[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, check=True
    ).stdout
    return {p.decode("utf-8") for p in out.split(b"\0") if p}


def _scanned(tracked: set[str]) -> dict[str, str]:
    return {
        path: (REPO_ROOT / path).read_text(encoding="utf-8")
        for path in sorted(tracked)
        if Path(path).suffix in SCANNED_SUFFIXES
        and path not in (REGISTER, THIS_FILE)
        and not path.startswith("docs/archive/")
        and (REPO_ROOT / path).is_file()
    }


@pytest.fixture(scope="module")
def tree() -> tuple[dict[int, str], set[str], dict[str, str]]:
    tracked = _tracked_paths()
    entries = parse_entries((REPO_ROOT / REGISTER).read_text(encoding="utf-8"))
    return entries, tracked, _scanned(tracked)


def _fail_if(problems: list[str]) -> None:
    assert not problems, "\n" + "\n".join(problems)


def test_the_register_defines_entries(tree) -> None:
    entries, _tracked, _files = tree
    assert len(entries) > 40


def test_every_citation_names_a_defined_entry(tree) -> None:
    entries, _tracked, files = tree
    _fail_if(check_citations_defined(set(entries), files))


def test_every_lives_in_pointer_resolves(tree) -> None:
    entries, tracked, files = tree
    _fail_if(check_pointers(entries, tracked, files))


def test_every_file_a_lives_in_sentence_names_cites_its_entry_back(tree) -> None:
    entries, tracked, files = tree
    _fail_if(check_back_citations(entries, tracked, files))


def test_every_entry_is_cited_unless_allow_listed(tree) -> None:
    entries, _tracked, files = tree
    _fail_if(check_every_entry_cited(set(entries), files, UNCITED_ALLOWED))


# --- synthetic cases: made-up registers and trees, never the real one ------------------------


def test_check_one_reports_an_undefined_number_and_not_a_defined_one() -> None:
    files = {"a.py": "x = 1  # L5\ny = 2  # L9\n"}
    problems = check_citations_defined({5}, files)
    assert problems == ["a.py:2: L9 is not a defined entry"]


def test_check_one_word_boundary_keeps_l1_apart_from_l11() -> None:
    assert check_citations_defined({11}, {"a.py": "# L1\n"}) == [
        "a.py:1: L1 is not a defined entry"
    ]
    assert check_citations_defined({1}, {"a.py": "# L11\n"}) == [
        "a.py:1: L11 is not a defined entry"
    ]
    assert check_citations_defined({1, 11}, {"a.py": "# L1 and L11\n"}) == []
    assert check_citations_defined(set(), {"a.py": "# XL7 and L8x\n"}) == []


def test_an_entry_is_found_in_its_second_paragraph() -> None:
    register = "**L1. First.** Text.\n\nMore text. Lives in `a.py`.\n\n**L2. Second.** Other.\n"
    entries = parse_entries(register)
    assert set(entries) == {1, 2}
    assert lives_in_sentences(entries[1]) == ["`a.py`"]
    assert lives_in_sentences(entries[2]) == []


def test_a_heading_ends_an_entry() -> None:
    register = "**L1. First.** Text.\n\n## Heading\n\nLives in `a.py`.\n"
    entries = parse_entries(register)
    assert lives_in_sentences(entries[1]) == []


def test_a_top_level_heading_ends_an_entry() -> None:
    register = "**L1. First.** Text.\n\n# Heading\n\nLives in `a.py`.\n"
    entries = parse_entries(register)
    assert lives_in_sentences(entries[1]) == []


def test_a_lives_in_sentence_split_across_lines_is_joined() -> None:
    text = "**L1. T.** Body. Lives in `x/a.py`,\n`x/b.py`\nand `c.py`. Then `d.py`."
    (sentence,) = lives_in_sentences(text)
    assert [p for p, _ in pointers(sentence)] == ["x/a.py", "x/b.py", "x/c.py"]


def test_a_dot_inside_a_file_name_does_not_end_the_sentence() -> None:
    (sentence,) = lives_in_sentences("**L1. T.** Lives in `engine/x.py`, `engine/y.py`. After.")
    assert [p for p, _ in pointers(sentence)] == ["engine/x.py", "engine/y.py"]


def test_a_pointer_after_the_closing_full_stop_is_not_part_of_the_sentence() -> None:
    (sentence,) = lives_in_sentences("**L1. T.** Lives in `a.py`. See also `b.py`.")
    assert [p for p, _ in pointers(sentence)] == ["a.py"]


def test_a_backticked_span_that_is_not_a_pointer_is_skipped() -> None:
    entries = {1: "**L1. T.** Lives in `SomeClass`, `a_name`, `Accounts.lif`."}
    assert check_pointers(entries, set(), {}) == []
    assert check_back_citations(entries, set(), {}) == []


def test_check_two_reports_a_path_that_is_not_tracked() -> None:
    entries = {1: "**L1. T.** Lives in `d/a.py`, `d/b.py`."}
    problems = check_pointers(entries, {"d/a.py"}, {"d/a.py": ""})
    assert problems == ["L1: d/b.py: path is not tracked"]


def test_check_two_reports_a_missing_symbol_and_not_a_present_one() -> None:
    files = {"x.py": "def present():\n    pass\n"}
    present = {1: "**L1. T.** Lives in `x.py::present`."}
    absent = {1: "**L1. T.** Lives in `x.py::absent`."}
    assert check_pointers(present, {"x.py"}, files) == []
    problems = check_pointers(absent, {"x.py"}, files)
    assert len(problems) == 1 and "x.py::absent" in problems[0]


def test_every_kind_of_top_level_name_is_a_symbol() -> None:
    files = {"x.py": "A: int = 1\nB = 2\nclass C:\n    pass\nasync def D():\n    pass\n"}
    entries = {1: "**L1. T.** Lives in `x.py::A`, `x.py::B`, `x.py::C`, `x.py::D`."}
    assert check_pointers(entries, {"x.py"}, files) == []


def test_a_dotted_symbol_must_be_inside_the_class() -> None:
    source = (
        "class C:\n    def f(self):\n        pass\n    n: int = 0\n\n"
        "def g():\n    def f():\n        pass\n"
    )
    files = {"x.py": source}
    ok = {1: "**L1. T.** Lives in `x.py::C.f`, `x.py::C.n`."}
    outside = {1: "**L1. T.** Lives in `x.py::C.g`."}
    in_function = {1: "**L1. T.** Lives in `x.py::g.f`."}
    assert check_pointers(ok, {"x.py"}, files) == []
    assert len(check_pointers(outside, {"x.py"}, files)) == 1
    (problem,) = check_pointers(in_function, {"x.py"}, files)
    assert "is not a class" in problem
    assert "is not defined" not in problem


def test_a_symbol_on_a_non_python_file_is_reported() -> None:
    entries = {1: "**L1. T.** Lives in `p/a.yaml::key`."}
    problems = check_pointers(entries, {"p/a.yaml"}, {"p/a.yaml": ""})
    assert problems == ["L1: p/a.yaml::key: symbol on a non-Python file"]


def test_a_brace_group_checks_each_alternative() -> None:
    entries = {1: "**L1. T.** Lives in `d/{a,b}.py`."}
    assert check_pointers(entries, {"d/a.py", "d/b.py"}, {}) == []
    assert check_pointers(entries, {"d/a.py"}, {}) == ["L1: d/b.py: path is not tracked"]


def test_a_bare_name_resolves_in_the_directory_of_the_last_slashed_path() -> None:
    entries = {1: "**L1. T.** Lives in `d/x.py`, `y.py`."}
    assert check_pointers(entries, {"d/x.py", "d/y.py"}, {}) == []
    assert check_pointers(entries, {"d/x.py", "y.py"}, {}) == ["L1: d/y.py: path is not tracked"]


def test_a_bare_name_with_no_earlier_slashed_path_resolves_from_the_root() -> None:
    entries = {1: "**L1. T.** Lives in `y.py`."}
    assert check_pointers(entries, {"y.py"}, {}) == []
    assert check_pointers(entries, {"d/y.py"}, {}) == ["L1: y.py: path is not tracked"]


def test_a_directory_exists_when_a_tracked_file_is_under_it() -> None:
    entries = {1: "**L1. T.** Lives in `d/`."}
    assert check_pointers(entries, {"d/a.py"}, {}) == []
    assert check_pointers(entries, {"e/a.py"}, {}) == ["L1: d/: path is not tracked"]


def test_a_directory_back_cites_when_any_file_under_it_does() -> None:
    entries = {1: "**L1. T.** Lives in `d/`."}
    tracked = {"d/a.py", "d/b.py"}
    assert check_back_citations(entries, tracked, {"d/a.py": "", "d/b.py": "# L1\n"}) == []
    problems = check_back_citations(entries, tracked, {"d/a.py": "", "d/b.py": "# L11\n"})
    assert problems == ["L1: no file under d/ cites it"]


def test_check_three_reports_a_file_without_the_number_and_not_one_with_it() -> None:
    entries = {1: "**L1. T.** Lives in `a.py`, `b.py`."}
    files = {"a.py": "# L1\n", "b.py": "# L11 only\n"}
    problems = check_back_citations(entries, {"a.py", "b.py"}, files)
    assert problems == ["L1: b.py does not cite it"]


def test_check_four_reports_an_uncited_entry_that_is_not_allow_listed() -> None:
    problems = check_every_entry_cited({1, 2}, {"a.py": "# L1\n"}, frozenset())
    assert problems == ["L2 is cited nowhere"]


def test_check_four_reports_a_stale_allow_list_entry() -> None:
    problems = check_every_entry_cited({1, 2}, {"a.py": "# L1\n"}, frozenset({1, 2}))
    assert problems == ["L1 is allow-listed but cited"]


def test_check_four_reports_an_allow_listed_number_that_is_not_defined() -> None:
    problems = check_every_entry_cited({1}, {"a.py": "# L1\n"}, frozenset({9}))
    assert problems == ["L9 is allow-listed but not defined"]


def test_the_last_slashed_path_fixes_the_directory_of_a_bare_name() -> None:
    entries = {1: "**L1. T.** Lives in `d/x.py`, `e/z.py`, `y.py`."}
    tracked = {"d/x.py", "e/z.py", "e/y.py"}
    assert check_pointers(entries, tracked, {}) == []
    assert check_pointers(entries, {"d/x.py", "e/z.py", "d/y.py"}, {}) == [
        "L1: e/y.py: path is not tracked"
    ]


def test_a_directory_sets_the_directory_of_a_bare_name() -> None:
    entries = {1: "**L1. T.** Lives in `d/`, `y.py`."}
    assert check_pointers(entries, {"d/y.py"}, {}) == []
    assert check_pointers(entries, {"d/a.py", "y.py"}, {}) == ["L1: d/y.py: path is not tracked"]


def test_check_three_covers_a_pointer_that_carries_a_symbol() -> None:
    entries = {1: "**L1. T.** Lives in `a.py::f`."}
    without = {"a.py": "def f():\n    pass\n"}
    with_number = {"a.py": "def f():\n    pass  # L1\n"}
    assert check_back_citations(entries, {"a.py"}, without) == ["L1: a.py does not cite it"]
    assert check_back_citations(entries, {"a.py"}, with_number) == []


def test_check_three_does_not_count_a_tracked_file_that_was_not_read_as_cited() -> None:
    entries = {1: "**L1. T.** Lives in `a.py`."}
    assert check_back_citations(entries, {"a.py"}, {}) == ["L1: a.py does not cite it"]


def test_check_two_reports_a_python_file_that_does_not_parse() -> None:
    entries = {1: "**L1. T.** Lives in `x.py::f`."}
    (problem,) = check_pointers(entries, {"x.py"}, {"x.py": "def f(:\n"})
    assert problem.startswith("L1: x.py::f: file does not parse")


def test_check_two_reports_a_symbol_on_a_tracked_python_file_that_was_not_read() -> None:
    entries = {1: "**L1. T.** Lives in `x.py::f`."}
    assert check_pointers(entries, {"x.py"}, {}) == ["L1: x.py::f: file was not read"]
