import unittest

from src.document.chunker import fixed_size_chunk, recursive_chunk


class ChunkerTests(unittest.TestCase):
    def test_rejects_invalid_budget(self):
        with self.assertRaises(ValueError):
            fixed_size_chunk("text", chunk_size=0)
        with self.assertRaises(ValueError):
            recursive_chunk("text", chunk_size=4, chunk_overlap=4)

    def test_token_length_function_is_respected(self):
        text = "abcdefghij" * 20
        token_count = lambda value: (len(value) + 3) // 4
        chunks = fixed_size_chunk(
            text, chunk_size=10, chunk_overlap=2, length_fn=token_count
        )
        self.assertTrue(chunks)
        self.assertTrue(all(token_count(chunk.text) <= 10 for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
