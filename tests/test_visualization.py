import os
import sys
import unittest

# Ensure the repository root is on sys.path so tests can import the `mylibs` package
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# Import the function to test
import libs.visualization as viz
from config import app_config


class TestVisualization(unittest.TestCase):

    def test_create_dataset(self):
        config = app_config
        viz.create_dataset(
            config.dataset,
            "train",
            config.radius,
            config.method,
            save_path="resources/train_dataset.h5",
        )

    def test_load_dataset(self):
        X, y = viz.load_dataset_h5("resources/train_dataset.h5")
        assert X is not None
        assert y is not None
        assert len(X) == len(y)


if __name__ == "__main__":
    unittest.main()
