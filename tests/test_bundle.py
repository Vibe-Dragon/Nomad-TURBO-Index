"""Small plain-text files share a pass. Other files stay alone."""

import unittest

from nomad_turbo.bundle import GAP_BYTES, BundleMember, group_small_text, members_stored


class _File:
    def __init__(self, source: str, kind: str, size_bytes: int, text: str):
        self.source = source
        self.kind = kind
        self.size_bytes = size_bytes
        self.text = text


class SmallFileBundleTest(unittest.TestCase):
    def test_tiny_text_shares_a_group_and_only_stored_text_completes(self):
        rows = [
            _File("a.md", "text", 100, "alpha"),
            _File("big.md", "text", GAP_BYTES + 1, "large body"),
            _File("b.md", "text", 200, "beta"),
            _File("scan.pdf", "pdf", 50, "pages"),
            _File("book.zim", "zim", 80, "zim"),
            _File("empty.md", "text", 10, ""),
        ]
        groups = group_small_text(rows)
        shared = [group for group in groups if len(group) > 1]
        self.assertEqual(len(shared), 1)
        self.assertEqual([item.source for item in shared[0]], ["a.md", "b.md", "empty.md"])
        alone = [group[0].source for group in groups if len(group) == 1]
        self.assertEqual(alone, ["big.md", "scan.pdf", "book.zim"])

        members = [BundleMember(item.source, item.text) for item in shared[0]]
        stored = {"a.md", "b.md", "empty.md"}
        self.assertEqual(members_stored(members, stored), ["a.md", "b.md"])
        self.assertEqual(members_stored(members, {"a.md"}), ["a.md"])
        self.assertEqual(members_stored(members, set()), [])


if __name__ == "__main__":
    unittest.main()
