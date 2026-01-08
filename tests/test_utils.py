import unittest
import sys
import os

# Add src to path so we can import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils import clean_text

class TestTextCleaning(unittest.TestCase):

    def test_basic_cleaning(self):
        raw = "I am VERY   angry!!!"
        expected = "i am very angry!!!"
        self.assertEqual(clean_text(raw), expected)

    def test_redaction_removal(self):
        raw = "My account number is XXXX XXXX."
        expected = "my account number is ."
        self.assertEqual(clean_text(raw), expected)

    def test_empty_input(self):
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(""), "")

    def test_special_chars(self):
        raw = "Hello @#% World"
        expected = "hello world"
        self.assertEqual(clean_text(raw), expected)

if __name__ == '__main__':
    unittest.main()