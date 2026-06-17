# THIS FILE IS ONLY USED ONCE TO BACKFILL THE LAG DELAY DATA INTO THE DATABASE
from posixpath import abspath
import sqlite3
import time
import os

def backfill_lag_delay():
    current_dir = os.path.dirname(abspath(__file__))
    conn = sqlite3.connect(f'{current_dir}/bus_data.db')
    cursor = conn.cursor()

    print("Fetching distinct routes to process...")
    routes = []
    with open(f'{current_dir}/routes.txt', 'r') as file:
        for row in file:
            row = row.strip()
            routes.append(row)

    total_updated = 0
    start_time = time.time()
    print(f'routes count: {len(routes)}')

    for route in routes:
        print(f"\nProcessing Route {route}...")

        cursor.execute('''
            SELECT id, date, direction, delay
            FROM vehicle_locations
            WHERE route = ? AND delay IS NOT NULL AND time IS NOT NULL
            ORDER BY date ASC, time ASC
        ''', (route,))

        rows = cursor.fetchall()
        print(f"Found {len(rows)} rows for route {route}. Calculating EWMA...")

        updates = []
        last_known_delays = {"0.0": 0.0, "1.0": 0.0}
        last_seen_date = None

        for row in rows:
            row_id = row[0]
            current_date = row[1]

            raw_dir = float(row[2])
            direction = "1.0" if raw_dir == 2.0 else "0.0"

            current_delay = float(row[3])

            if last_seen_date is not None and current_date != last_seen_date:
                last_known_delays = {"0.0": 0.0, "1.0": 0.0}
            last_seen_date = current_date

            lag_delay_for_this_row = last_known_delays[direction]

            updates.append((lag_delay_for_this_row, row_id))

            if lag_delay_for_this_row == 0.0:
                last_known_delays[direction] = current_delay
            else:
                last_known_delays[direction] = (0.9 * lag_delay_for_this_row) + (0.1 * current_delay)

        print(f"Writing {len(updates)} updates to the database for Route {route}...")

        cursor.executemany('''
            UPDATE vehicle_locations
            SET lag_delay = ?
            WHERE id = ?
        ''', updates)

        conn.commit()
        total_updated += len(updates)
        print(f"Route {route} complete.")

    end_time = time.time()
    print(f"\n--- Backfill Complete ---")
    print(f"Successfully updated {total_updated} rows in {end_time - start_time:.2f} seconds.")

    conn.close()

if __name__ == '__main__':
    backfill_lag_delay()