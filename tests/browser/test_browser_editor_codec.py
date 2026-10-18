# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The form's JSON codec and number rule, run in Chromium on the API page: Python's own JSON
output comes back token for token, text that is not one JSON value is refused, and the number
control's text becomes JSON number text only where the engine would read it as a number.
"""

from __future__ import annotations

import json
from typing import Any

from offline import Offline
from playwright.sync_api import Page


def loads_text(text: str) -> Any:
    return json.loads(
        text,
        parse_float=lambda t: ("f", t),
        parse_int=lambda t: ("i", t),
        object_pairs_hook=lambda pairs: ("o", pairs),
    )


CORPUS = [
    {
        "seed": 2**64 + 1,
        "x": 1.0,
        "y": 95000,
        "z": 1e-9,
        "w": 1e20,
        "v": 0.1 + 0.2,
        "u": -0.0,
        "t": 1e16,
    },
    {"b": {"real_mean": 0.0}, "2": 1, "10": 2, "a": 3, "1": [1.5, 2]},
    {
        "s": "é " + chr(0x2028) + ' \x85 \x01 "q" \\ / \t\n',
        "lone": "\ud800x",
        "emoji": "😀",
        "nul": "\x00",
    },
    [],
    {},
    [[]],
    [{}],
    "str",
    0,
    -1,
    1.5e-300,
    True,
    False,
    None,
]
NOT_JSON = [
    "",
    "{",
    "[1,]",
    '{"a":1,}',
    "NaN",
    "Infinity",
    "01",
    "1.",
    ".5",
    "+1",
    '"\\x"',
    "[1] 2",
    "tru",
    '"a\nb"',
    "{a:1}",
    '{"a" 1}',
    "-",
    "1e",
    '{"a":1,"a":2}',
]
TIDIED = {
    "+5": "5",
    " 5 ": "5",
    ".5": "0.5",
    "1.": "1.0",
    "007": "7",
    "00.5": "0.5",
    "-0": "-0",
    "1.e5": "1.0e5",
}
KEPT = ["1E+05", "95000", "1e-9", "18446744073709551617", "1e308", "1" + "0" * 400]
REFUSED = [
    "",
    "-",
    ".",
    "e5",
    "1,000",
    "0x10",
    "inf",
    "1_000",
    chr(0x661),
    "1e400",
    "-1e400",
    "1.0e309",
]


def test_python_json_comes_back_token_for_token(api_form_page: tuple[Page, Offline]) -> None:
    page, _ = api_form_page
    texts = []
    for value in CORPUS:
        texts.append(json.dumps(value))
        texts.append(json.dumps(value, indent=2, ensure_ascii=False))
    written = page.evaluate(
        "(texts) => texts.map((t) => NorthplanJson.stringify(NorthplanJson.parse(t)))", texts
    )
    assert len(written) == len(texts) == 2 * len(CORPUS)
    for text, again in zip(texts, written, strict=True):
        assert loads_text(again) == loads_text(text), text


def test_text_that_is_not_json_is_refused(api_form_page: tuple[Page, Offline]) -> None:
    page, _ = api_form_page
    names = page.evaluate(
        """(texts) => texts.map((t) => {
            try { NorthplanJson.parse(t); return 'accepted'; } catch (e) { return e.name; }
        })""",
        NOT_JSON,
    )
    assert names == ["SyntaxError"] * len(NOT_JSON)


def test_objects_keep_their_key_order_and_numbers_their_text(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    found = page.evaluate(
        """() => {
            const text = '{"b":1,"2":2.0,"10":3}';
            const parsed = NorthplanJson.parse(text);
            return [
                Array.from(parsed.keys()),
                NorthplanJson.stringify(parsed),
                NorthplanJson.toNumber(NorthplanJson.parse('42')),
            ];
        }"""
    )
    assert found == [["b", "2", "10"], '{"b":1,"2":2.0,"10":3}', 42]


def test_stringify_refuses_what_it_cannot_write_exactly(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    found = page.evaluate(
        """() => {
            const out = [];
            for (const v of [1.5, NaN, 2 ** 53, undefined, {}, () => 1]) {
                try { NorthplanJson.stringify(v); out.push('written'); }
                catch (e) { out.push(e.name); }
            }
            out.push(NorthplanJson.stringify(7));
            return out;
        }"""
    )
    assert found == ["TypeError"] * 6 + ["7"]


def number_text(page: Page, texts: list[str]) -> list[str | None]:
    return page.evaluate("(texts) => texts.map((t) => northplanNumberText(t))", texts)


def test_number_text_tidies_what_the_engine_reads_as_a_number(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    assert number_text(page, list(TIDIED)) == list(TIDIED.values())
    assert number_text(page, KEPT) == KEPT


def test_number_text_refuses_other_text(api_form_page: tuple[Page, Offline]) -> None:
    page, _ = api_form_page
    assert number_text(page, REFUSED) == [None] * len(REFUSED)
