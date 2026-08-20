import unittest

from neurotrain.celebrations import celebrate


class CelebrationsTests(unittest.TestCase):
    def test_returns_html_display_object(self):
        from IPython.display import HTML

        result = celebrate("Great job!", "You finished the notebook.")
        self.assertIsInstance(result, HTML)

    def test_message_and_subtext_are_present(self):
        result = celebrate("Great job!", "You finished the notebook.")
        self.assertIn("Great job!", result.data)
        self.assertIn("You finished the notebook.", result.data)

    def test_stays_offline_no_external_requests(self):
        result = celebrate("Great job!")
        lowered = result.data.lower()
        self.assertNotIn("http://", lowered)
        self.assertNotIn("https://", lowered)
        self.assertNotIn("cdn.", lowered)


if __name__ == "__main__":
    unittest.main()
