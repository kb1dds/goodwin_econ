#Unit tests for goodwin.py

import unittest
import goodwinsheaf

class RadiiTestCase(unittest.TestCase):
    """Tests for differebt consistency radii."""

    def test_constructed_is_small(self):
        """the radii of data made by Euler sufficiently small?"""
        self.assertTrue(all(elt<10 for elt in goodwinsheaf.checkradii()))#check all entries have small radii

if __name__ == '__main__':
    unittest.main()