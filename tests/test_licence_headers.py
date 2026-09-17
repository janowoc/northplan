# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Every source file states who owns it and under what terms it may be used.

A file with no SPDX header is, to any automated licence scanner and to anyone
who receives it on its own, unlicensed and of unknown provenance. The AGPL's
copyleft only reaches a recipient who can tell the licence applies, so a header
missing from a file is a licence that does not travel with it.

``CPY001`` in ruff enforces the same rule, but only over ``.py`` files and only
over their first 4096 bytes. The parameter YAML, the Markdown, the Dockerfile,
the compose file and the HTML are covered here and nowhere else.

Tracked files are enumerated with ``git ls-files`` rather than a filesystem
walk: a walk descends into ``.venv/`` and cannot distinguish a file that is
under version control from one that merely exists.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: File extensions that must carry the header.
IN_SCOPE_SUFFIXES = frozenset({".py", ".yaml", ".yml", ".md", ".toml", ".html"})

#: Files that must carry the header but have no extension to recognise them by.
IN_SCOPE_NAMES = frozenset({"Dockerfile"})

#: Paths, relative to the repository root, that deliberately carry no header.
#:
#: ``LICENSE`` is the FSF's own text, which carries its own copyright notice
#: and must not be annotated. The ``.gitkeep`` files are empty by definition
#: and a header would give them content, which is the one thing they must not
#: have. The two ignore files do support ``#`` comments — that is not the
#: reason — but they are build and tooling configuration rather than source,
#: and the human decided in issue 26 that they carry no notice. The golden
#: source workbook is an ODF package — a zip — and any text prepended to it
#: stops it being one; its licence travels with the repository it is only
#: ever read from.
EXEMPT = frozenset(
    {
        "LICENSE",
        ".gitignore",
        ".dockerignore",
        "scenarios/.gitkeep",
        "tests/characterization/.gitkeep",
        "tests/golden/cases/.gitkeep",
        "tests/golden/sources/2026.ods",
    }
)

#: The two header lines, each anchored to a comment opener.
#:
#: The comment prefix is part of the pattern, not decoration. Without it the
#: same text uncommented would satisfy this test — and in a YAML file under
#: ``params/`` an uncommented header is not a comment at all but two spurious
#: top-level string keys, which ``yaml.safe_load`` accepts, the loader accepts,
#: and no structural test rejects, because they carry none of the ``_rate``,
#: ``_month`` or ``_annual`` stems those tests key on. That is exactly the
#: silent mutation of a parameter file this sweep exists to make impossible.
#:
#: The year is left open, matching the ``notice-rgx`` given to ruff, so that a
#: file first written in a later year is not required to claim 2026. The holder
#: and the licence are pinned: a header naming a different licence is a worse
#: failure than a header that is absent, because it looks deliberate. Ruff
#: checks neither, so these two patterns are the binding constraint on both.
_OPENER = r"^\s*(?:#|<!--)\s*"
COPYRIGHT_RE = re.compile(_OPENER + r"SPDX-FileCopyrightText:\s*\d{4} Jan Owoc")
LICENCE_RE = re.compile(_OPENER + r"SPDX-License-Identifier:\s*AGPL-3\.0-or-later")


def tracked_files() -> list[str]:
    """Every file under version control, as a path relative to the repository root."""
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted(entry for entry in completed.stdout.split("\0") if entry)


def is_in_scope(relative_path: str) -> bool:
    """Whether ``relative_path`` is a file kind that must carry the header."""
    path = Path(relative_path)
    return path.suffix in IN_SCOPE_SUFFIXES or path.name in IN_SCOPE_NAMES


def header_position(text: str) -> list[str]:
    """The two lines a header must occupy in ``text``: the first two non-blank.

    Anchored rather than searched within a window. The point of a header is
    that it is the first thing in the file, and a fixed scan depth enforces
    something weaker while giving a wrong diagnostic at the boundary: a file
    whose header sat one line past the window would be reported as *missing* a
    line it plainly carries, and the reader would be told to add a duplicate.

    The one exception is YAML frontmatter, which must keep its opening ``---``
    on line 1 — ``.claude/agents/*.md`` would stop being read as agent
    definitions otherwise. There the header is the first two non-blank lines
    after the closing ``---``. An unterminated ``---`` is treated as ordinary
    text rather than swallowing the file.
    """
    lines = text.split("\n")
    start = 0
    if lines and lines[0].strip() == "---":
        for index, line in enumerate(lines[1:], 1):
            if line.strip() == "---":
                start = index + 1
                break
    return [line for line in lines[start:] if line.strip()][:2]


def header_fields_missing_from(text: str) -> list[str]:
    """The SPDX fields absent from where a header must be in ``text``.

    The two lines are checked by position and in order, not searched for among
    the opening lines. "Somewhere near the top" would let a line of code sit
    above the copyright line and still count the copyright line as present,
    which is not what a header is.

    Takes the text rather than a path so that the anchoring and the patterns
    can be exercised directly against contrived input — a test that
    reimplemented either would go on passing if this function stopped doing it.
    """
    opening = header_position(text)
    first = opening[0] if opening else ""
    second = opening[1] if len(opening) > 1 else ""

    absent = []
    if not COPYRIGHT_RE.match(first):
        absent.append("SPDX-FileCopyrightText")
    if not LICENCE_RE.match(second):
        absent.append("SPDX-License-Identifier")
    return absent


def missing_header_fields(relative_path: str) -> list[str]:
    """The SPDX fields absent from the opening of the file at ``relative_path``."""
    return header_fields_missing_from((REPO_ROOT / relative_path).read_text(encoding="utf-8"))


def test_the_file_enumeration_is_not_empty() -> None:
    """Guard against every assertion below passing because ``git ls-files`` found nothing.

    If the subprocess ran outside a work tree, or the repository were renamed
    out from under it, the header tests would report green over zero files.
    """
    in_scope = [path for path in tracked_files() if is_in_scope(path)]
    assert in_scope, f"No tracked in-scope files found under {REPO_ROOT}; the header tests are vacuous."


def test_every_in_scope_file_carries_the_licence_header() -> None:
    """No tracked source file ships without its copyright and licence lines.

    Reports every offender at once. Failing on the first would turn adding a
    batch of files into one round trip per file.
    """
    offenders = [
        f"{path}: missing {' and '.join(fields)}"
        for path in tracked_files()
        if is_in_scope(path)
        for fields in [missing_header_fields(path)]
        if fields
    ]

    assert not offenders, (
        "Files under version control are missing their SPDX header. Add, at the "
        "top of each (below the YAML frontmatter block, if the file opens with "
        "one):\n"
        "  SPDX-FileCopyrightText: 2026 Jan Owoc\n"
        "  SPDX-License-Identifier: AGPL-3.0-or-later\n"
        "commented with `#` or wrapped in `<!-- -->` as the format requires:\n"
        + "\n".join(offenders)
    )


def test_the_exemption_list_is_exactly_the_files_agreed_to_be_exempt() -> None:
    """The set of unlicensed files is a written decision, not a side effect.

    Spelled out against a literal so that quietly adding another exemption to
    silence the test above fails here instead. Each is also required to exist:
    an exemption for a deleted file is stale, and a stale list is how a real
    file later ends up matching one by accident.
    """
    agreed = {
        "LICENSE",
        ".gitignore",
        ".dockerignore",
        "scenarios/.gitkeep",
        "tests/characterization/.gitkeep",
        "tests/golden/cases/.gitkeep",
        "tests/golden/sources/2026.ods",
    }

    assert agreed == EXEMPT, (
        "The exemption list changed. Exempting a file from the licence header "
        "is the human's decision, not a way to make the sweep pass."
    )

    absent = sorted(path for path in EXEMPT if not (REPO_ROOT / path).exists())
    assert not absent, (
        "These files are listed as exempt from the licence header but no longer "
        "exist. Remove the exemption rather than leaving it to match something "
        "else later:\n" + "\n".join(absent)
    )

    # Read as bytes: one exemption is a binary ODF package, and decoding it as
    # text raises before the assertion it is here to satisfy.
    annotated = sorted(
        path for path in EXEMPT if b"SPDX-" in (REPO_ROOT / path).read_bytes()
    )
    assert not annotated, (
        "These files are listed as exempt but carry SPDX text anyway, so the "
        "exemption is recording something that is not true. Either the file "
        "should be in scope, or the text should come off it:\n"
        + "\n".join(annotated)
    )

    overlapping = sorted(path for path in EXEMPT if is_in_scope(path))
    assert not overlapping, (
        "These files are both exempt and in scope, so two tests here now give "
        "contradictory instructions about them — one demands the header, the "
        "other demands its absence. The two sets must stay disjoint:\n"
        + "\n".join(overlapping)
    )


def test_every_tracked_file_is_either_in_scope_or_exempt() -> None:
    """A new kind of file cannot enter the repository without a header decision.

    Adding, say, a shell script or a JSON fixture would otherwise be neither
    checked nor deliberately exempted: it would simply fall outside every
    assertion here and ship unlicensed.
    """
    unclassified = [path for path in tracked_files() if not is_in_scope(path) and path not in EXEMPT]

    assert not unclassified, (
        "These tracked files are neither covered by the licence-header check nor "
        "listed as exempt. Decide which: add the extension to IN_SCOPE_SUFFIXES "
        "(or the name to IN_SCOPE_NAMES) and give the file a header, or add it to "
        "EXEMPT with a reason:\n" + "\n".join(unclassified)
    )


def test_the_header_detector_rejects_a_file_without_one() -> None:
    """The scan can fail, so the sweep above is not passing on a broken predicate.

    Without this, a regex that matched everything would make every file look
    compliant.
    """
    hash_form = "# SPDX-FileCopyrightText: 2026 Jan Owoc\n# SPDX-License-Identifier: AGPL-3.0-or-later\n"
    html_form = (
        "<!-- SPDX-FileCopyrightText: 2026 Jan Owoc -->\n"
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->\n"
    )
    both = ["SPDX-FileCopyrightText", "SPDX-License-Identifier"]

    assert header_fields_missing_from(hash_form) == []
    assert header_fields_missing_from(html_form) == []
    assert header_fields_missing_from("") == both
    assert header_fields_missing_from("# Alberta provincial income tax parameters — 2026\n") == both

    wrong_licence = "# SPDX-FileCopyrightText: 2026 Jan Owoc\n# SPDX-License-Identifier: MIT\n"
    assert header_fields_missing_from(wrong_licence) == ["SPDX-License-Identifier"], (
        "A header relicensing the file must fail, not pass for carrying two SPDX lines."
    )

    wrong_holder = "# SPDX-FileCopyrightText: 2026 Somebody Else\n# SPDX-License-Identifier: AGPL-3.0-or-later\n"
    assert header_fields_missing_from(wrong_holder) == ["SPDX-FileCopyrightText"], (
        "ruff's notice-rgx accepts any holder, so this test is the only thing checking it."
    )


def test_the_header_must_be_the_first_thing_in_the_file() -> None:
    """A licence line further down does not count. Blank lines and frontmatter do not count against it.

    The point of the header is that someone opening the file sees it first, so
    the check is anchored to the top rather than searching a window. Both
    directions matter: a header pushed down by other content must fail, and a
    header sitting correctly below a frontmatter block must pass however long
    that block grows.
    """
    header = "# SPDX-FileCopyrightText: 2026 Jan Owoc\n# SPDX-License-Identifier: AGPL-3.0-or-later\n"
    both = ["SPDX-FileCopyrightText", "SPDX-License-Identifier"]

    assert header_fields_missing_from(header) == []
    assert header_fields_missing_from("\n\n\n" + header) == [], "Leading blank lines are not content."
    assert header_fields_missing_from("import os\n" + header) == both, "A header below code is not a header."
    assert header_fields_missing_from("# A note.\n" + header) == both, (
        "A header below another comment is not a header."
    )

    frontmatter = "---\n" + "".join(f"key{n}: value\n" for n in range(40)) + "---\n\n"
    assert header_fields_missing_from(frontmatter + header) == [], (
        "A header below YAML frontmatter must pass no matter how long the block is."
    )
    assert header_fields_missing_from("---\nname: x\n" + header) == both, (
        "Unterminated frontmatter must not swallow the file and excuse a missing header."
    )


def test_the_header_must_actually_be_a_comment() -> None:
    """SPDX text that is not commented out is not a header, and in YAML is data.

    ``params/2026/federal.yaml`` with the two lines uncommented parses to a
    mapping carrying two extra top-level string keys. ``yaml.safe_load``
    accepts it, the loader accepts it, and no structural test rejects it — the
    units, month and period-suffix checks all key on stems these keys do not
    have. Matching the comment opener is what keeps this sweep from certifying
    a parameter file that has been silently changed.
    """
    uncommented = "SPDX-FileCopyrightText: 2026 Jan Owoc\nSPDX-License-Identifier: AGPL-3.0-or-later\n"
    assert header_fields_missing_from(uncommented) == [
        "SPDX-FileCopyrightText",
        "SPDX-License-Identifier",
    ]
