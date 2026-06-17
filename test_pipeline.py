import unittest
import os
import sqlite3
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from neuralnet import PyTorchModel
import sys

sys.path.append(os.path.abspath("data_scraping"))

class TestPipeline(unittest.TestCase):

    def test_database_schema(self):
        """Verify the database has been updated with the lag_delay column."""
        db_path = "data_scraping/bus_data.db"
        self.assertTrue(os.path.exists(db_path), f"Database not found at {db_path}")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(vehicle_locations)")
        columns = [col[1] for col in cursor.fetchall()]
        conn.close()

        print("Schema columns:", columns)
        self.assertIn("lag_delay", columns, "Column 'lag_delay' should exist in the vehicle_locations table")

    def test_scraper_tracking_and_edge_cases(self):
        """Verify tracking of delay and date changes in scraper/inference style."""
        last_known_delays = {"0.0": 0.0, "1.0": 0.0}
        last_seen_date = None

        pings = [
            ("2026-06-16", 1000.0, 1.0, 120.0),
            ("2026-06-16", 1100.0, 1.0, 150.0),
            ("2026-06-16", 1200.0, 0.0, 80.0),
            ("2026-06-17", 500.0, 1.0, 45.0),
        ]

        results = []
        for date, time, direction, delay in pings:
            if last_seen_date is not None and date != last_seen_date:
                last_known_delays["0.0"] = 0.0
                last_known_delays["1.0"] = 0.0
            last_seen_date = date

            dir_key = str(direction)
            lag_delay = last_known_delays.get(dir_key, 0.0)

            results.append(lag_delay)

            current_delay = float(delay) if delay is not None else 0.0
            last_known_delays[dir_key] = current_delay

        self.assertEqual(results[0], 0.0)
        self.assertEqual(results[1], 120.0)
        self.assertEqual(results[2], 0.0)
        self.assertEqual(results[3], 0.0)
        print("Scraper delay tracking logic tests passed.")

    def test_pytorch_model_dimensions(self):
        """Test if the PyTorch model correctly processes inputs with the new shapes."""
        num_unique_buses = 10
        model = PyTorchModel(num_unique_buses=num_unique_buses)

        batch_size = 5
        X = torch.zeros((batch_size, 7), dtype=torch.float32)
        X[:, 0] = 0.5
        X[:, 1] = 0.8
        X[:, 2] = 60.1
        X[:, 3] = 24.9
        X[:, 4] = 120.0
        X[:, 5] = 1.0
        X[:, 6] = 3.0

        output = model(X)

        self.assertEqual(output.shape, (batch_size, 1), f"Expected shape (5, 1), got {output.shape}")
        print("PyTorch model forward pass dimensions test passed.")

if __name__ == "__main__":
    unittest.main()
