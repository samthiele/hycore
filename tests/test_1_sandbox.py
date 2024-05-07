import unittest
import os

import hycore as hc

class MyTestCase(unittest.TestCase):
    def test_sandbox(self):

        sbox = hc.get_sandbox()
        print(sbox)
        self.assertTrue(os.path.exists(sbox))
        hc.empty_sandbox()
        self.assertFalse(os.path.exists(sbox))


if __name__ == '__main__':
    unittest.main()
