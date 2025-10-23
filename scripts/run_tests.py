#!/usr/bin/env python3
import os
import sys
import unittest


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)

    tests_dir = os.path.join(repo_root, "tests")
    suite = unittest.defaultTestLoader.discover(start_dir=tests_dir, pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
