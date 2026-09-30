"""Coding tasks for action-flow study B (docs/validation/action-flow.md). Each has visible files, a prompt, and a
hidden unittest file the agent never sees; a run passes when the hidden tests pass on its final work directory."""

TASKS = {
    "chunk_bug": {
        "prompt": "chunk() in chunks.py drops items. Fix it so the tests in test_chunks.py pass "
        "(run them with `python3 -m unittest`).",
        "files": {
            "chunks.py": '''def chunk(items, size):
    """Split items into consecutive lists of at most `size` elements. Raises ValueError if size < 1."""
    return [items[i : i + size] for i in range(0, len(items) - size, size)]
''',
            "test_chunks.py": """import unittest
from chunks import chunk

class T(unittest.TestCase):
    def test_tail(self):
        self.assertEqual(chunk([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]])

if __name__ == "__main__":
    unittest.main()
""",
        },
        "hidden": """import unittest
from chunks import chunk

class H(unittest.TestCase):
    def test_all(self):
        self.assertEqual(chunk([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]])
        self.assertEqual(chunk([], 3), [])
        self.assertEqual(chunk([1, 2], 5), [[1, 2]])
        self.assertEqual(chunk([1, 2, 3, 4], 2), [[1, 2], [3, 4]])
        with self.assertRaises(ValueError):
            chunk([1], 0)

unittest.main()
""",
    },
    "cart_total": {
        "prompt": "The cart total is wrong for prices with thousands separators. Fix the bug so "
        "`python3 -m unittest` passes.",
        "files": {
            "prices.py": '''def parse_price(text):
    """Parse a price like '1,299.50' or '12.5' (comma = thousands separator) into a float."""
    return float(text.replace(",", "."))  if text.count(",") and "." not in text else float(text.replace(",", ""))
''',
            "cart.py": """from prices import parse_price

def total(lines):
    \"\"\"lines: list of (price_text, quantity). Returns the total rounded to 2 decimals.\"\"\"
    return round(sum(parse_price(p) * q for p, q in lines), 2)
""",
            "test_cart.py": """import unittest
from cart import total

class T(unittest.TestCase):
    def test_total(self):
        self.assertEqual(total([("1,299", 1), ("0.50", 2)]), 1300.0)

if __name__ == "__main__":
    unittest.main()
""",
        },
        "hidden": """import unittest
from cart import total
from prices import parse_price

class H(unittest.TestCase):
    def test_all(self):
        self.assertEqual(total([("1,299", 1), ("0.50", 2)]), 1300.0)
        self.assertEqual(parse_price("1,299.50"), 1299.5)
        self.assertEqual(parse_price("12,000"), 12000.0)
        self.assertEqual(parse_price("12.5"), 12.5)
        self.assertEqual(total([("2,000", 2), ("1,000.25", 1)]), 5000.25)

unittest.main()
""",
    },
    "missing_module": {
        "prompt": "Make `python3 report.py sample.json` work. It should print one line: "
        "`records=<number of records> total=<sum of the amount fields>`.",
        "files": {
            "report.py": """import sys
import orjsonx as json

def main(path):
    data = json.loads(open(path, "rb").read())
    total = sum(r["amount"] for r in data)
    print(f"records={len(data)} total={total}")

if __name__ == "__main__":
    main(sys.argv[1])
""",
            "sample.json": '[{"amount": 10}, {"amount": 2.5}, {"amount": 30}]',
        },
        "hidden": """import json, subprocess, sys, tempfile, unittest, os

class H(unittest.TestCase):
    def test_all(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump([{"amount": 1}, {"amount": 2}, {"amount": 3.5}, {"amount": 4}], f)
        out = subprocess.run([sys.executable, "report.py", f.name], capture_output=True, text=True).stdout.strip()
        os.unlink(f.name)
        self.assertEqual(out, "records=4 total=10.5")

unittest.main()
""",
    },
    "round_money": {
        "prompt": "Implement round_money() in money.py as its docstring says, so `python3 -m unittest` passes.",
        "files": {
            "money.py": '''def round_money(value):
    """Round a number to 2 decimal places, halves away from zero, as it is written in decimal
    (2.675 -> "2.68", 1.005 -> "1.01", -1.005 -> "-1.01"). Return a string with exactly 2 decimals."""
    raise NotImplementedError
''',
            "test_money.py": """import unittest
from money import round_money

class T(unittest.TestCase):
    def test_examples(self):
        self.assertEqual(round_money(2.675), "2.68")
        self.assertEqual(round_money(1.005), "1.01")
        self.assertEqual(round_money(-1.005), "-1.01")
        self.assertEqual(round_money(3), "3.00")

if __name__ == "__main__":
    unittest.main()
""",
        },
        "hidden": """import unittest
from money import round_money

class H(unittest.TestCase):
    def test_all(self):
        cases = {2.675: "2.68", 1.005: "1.01", -1.005: "-1.01", 3: "3.00", 0.125: "0.13", 1.2345: "1.23",
                 -2.5: "-2.50", 0.005: "0.01", 10.0049: "10.00", 8.345: "8.35"}
        for value, expected in cases.items():
            self.assertEqual(round_money(value), expected, value)

unittest.main()
""",
    },
    "deep_flatten": {
        "prompt": "flatten() in flat.py crashes on deeply nested input. Fix it so `python3 -m unittest` passes.",
        "files": {
            "flat.py": '''def flatten(value):
    """Flatten nested lists into one list, left to right."""
    if not isinstance(value, list):
        return [value]
    out = []
    for item in value:
        out.extend(flatten(item))
    return out
''',
            "test_flat.py": """import unittest
from flat import flatten

class T(unittest.TestCase):
    def test_deep(self):
        value = 1
        for _ in range(5000):
            value = [value]
        self.assertEqual(flatten(value), [1])

    def test_order(self):
        self.assertEqual(flatten([1, [2, [3, 4]], 5]), [1, 2, 3, 4, 5])

if __name__ == "__main__":
    unittest.main()
""",
        },
        "hidden": """import unittest
from flat import flatten

class H(unittest.TestCase):
    def test_all(self):
        value = [1]
        for _ in range(20000):
            value = [value, 2]
        out = flatten(value)
        self.assertEqual(out[0], 1)
        self.assertEqual(len(out), 20001)
        self.assertEqual(flatten([1, [2, [3, 4]], 5]), [1, 2, 3, 4, 5])
        self.assertEqual(flatten([]), [])
        self.assertEqual(flatten([[], [[]], 7]), [7])

unittest.main()
""",
    },
    "log_parse": {
        "prompt": "Implement parse_line() in logs.py as documented, so `python3 -m unittest` passes.",
        "files": {
            "logs.py": '''def parse_line(line):
    """Parse '2026-09-30T10:00:00Z [WARN] (svc=api,id=7) disk almost full' into
    {"time": "2026-09-30T10:00:00Z", "level": "WARN", "fields": {"svc": "api", "id": "7"}, "message": "disk almost full"}.
    The level is upper-cased. The (key=value,...) part is optional; without it fields is {}.
    The message may itself contain brackets or parentheses. Return None for lines that do not match."""
    raise NotImplementedError
''',
            "test_logs.py": """import unittest
from logs import parse_line

class T(unittest.TestCase):
    def test_full(self):
        self.assertEqual(parse_line("2026-09-30T10:00:00Z [WARN] (svc=api,id=7) disk almost full"),
            {"time": "2026-09-30T10:00:00Z", "level": "WARN", "fields": {"svc": "api", "id": "7"},
             "message": "disk almost full"})

    def test_no_fields(self):
        self.assertEqual(parse_line("2026-09-30T10:00:01Z [info] started")["fields"], {})

if __name__ == "__main__":
    unittest.main()
""",
        },
        "hidden": """import unittest
from logs import parse_line

class H(unittest.TestCase):
    def test_all(self):
        r = parse_line("2026-09-30T10:00:00Z [WARN] (svc=api,id=7) disk almost full")
        self.assertEqual(r, {"time": "2026-09-30T10:00:00Z", "level": "WARN", "fields": {"svc": "api", "id": "7"},
                             "message": "disk almost full"})
        r = parse_line("2026-09-30T10:00:01Z [info] started [ok] (retry 2)")
        self.assertEqual((r["level"], r["fields"], r["message"]), ("INFO", {}, "started [ok] (retry 2)"))
        r = parse_line("2026-09-30T10:00:02Z [Error] (svc=db) failed (code=5)")
        self.assertEqual((r["level"], r["fields"], r["message"]), ("ERROR", {"svc": "db"}, "failed (code=5)"))
        self.assertIsNone(parse_line("not a log line"))

unittest.main()
""",
    },
    "business_days": {
        "prompt": "Implement business_days() in days.py as documented, so `python3 -m unittest` passes.",
        "files": {
            "days.py": '''def business_days(start, end, holidays=()):
    """Count weekdays (Mon-Fri) d with start < d <= end that are not in holidays.
    start and end are datetime.date; if end <= start the result is 0."""
    raise NotImplementedError
''',
            "test_days.py": """import unittest
from datetime import date
from days import business_days

class T(unittest.TestCase):
    def test_week(self):
        self.assertEqual(business_days(date(2026, 9, 25), date(2026, 10, 2)), 5)

if __name__ == "__main__":
    unittest.main()
""",
        },
        "hidden": """import unittest
from datetime import date
from days import business_days

class H(unittest.TestCase):
    def test_all(self):
        self.assertEqual(business_days(date(2026, 9, 25), date(2026, 10, 2)), 5)
        self.assertEqual(business_days(date(2026, 9, 26), date(2026, 9, 28)), 1)
        self.assertEqual(business_days(date(2026, 10, 2), date(2026, 9, 25)), 0)
        self.assertEqual(business_days(date(2026, 9, 25), date(2026, 10, 2), [date(2026, 9, 30)]), 4)
        self.assertEqual(business_days(date(2026, 1, 1), date(2026, 12, 31)), 260)
        self.assertEqual(business_days(date(2026, 9, 30), date(2026, 9, 30)), 0)

unittest.main()
""",
    },
    "pytest_style": {
        "prompt": "Implement slugify() in slug.py so the tests in test_slug.py pass.",
        "files": {
            "slug.py": '''def slugify(text):
    """Lower-case, replace every run of non-alphanumeric characters with a single '-', and strip '-' from
    both ends. Letters with accents keep their base letter (e.g. 'é' -> 'e')."""
    raise NotImplementedError
''',
            "test_slug.py": """import pytest
from slug import slugify

@pytest.mark.parametrize("text,expected", [
    ("Hello, World!", "hello-world"),
    ("  Café  au lait ", "cafe-au-lait"),
    ("--a--b--", "a-b"),
])
def test_slugify(text, expected):
    assert slugify(text) == expected
""",
        },
        "hidden": """import unittest
from slug import slugify

class H(unittest.TestCase):
    def test_all(self):
        cases = {"Hello, World!": "hello-world", "  Café  au lait ": "cafe-au-lait", "--a--b--": "a-b",
                 "Crème Brûlée 2": "creme-brulee-2", "": "", "ÀÉÎÕÜ": "aeiou"}
        for text, expected in cases.items():
            self.assertEqual(slugify(text), expected, text)

unittest.main()
""",
    },
}
