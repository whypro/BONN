import unittest
from processor import Processor

class ProcessorTest(unittest.TestCase):

    def test_dec2bin(self):
        data = [
            (-1.5, '1111 1111 1111 1110 1000 0000 0000 0000'.replace(' ', '')),
            (-6.3125, '1111 1111 1111 1001 1011 0000 0000 0000'.replace(' ', '')),
            (-3.01171875, '1111 1111 1111 1100 1111 1101 0000 0000'.replace(' ', ''))
        ]
        for d, b in data:
            self.assertEqual(b, self.processor.dec2bin(d))

    def setUp(self):
        self.processor = Processor()


if __name__ == '__main__':
    unittest.main()