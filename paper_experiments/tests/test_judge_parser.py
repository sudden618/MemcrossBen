import unittest

from evaluation.judge import parse_judge_output


class JudgeParserTests(unittest.TestCase):
    def test_fenced_json(self):
        parsed = parse_judge_output(
            '```json\n{"coverage":5,"accuracy":4,"attribution":3,"alignment":2}\n```'
        )
        self.assertEqual(parsed["coverage"], 5)

    def test_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            parse_judge_output('{"coverage":6,"accuracy":4,"attribution":3,"alignment":2}')


if __name__ == "__main__":
    unittest.main()
