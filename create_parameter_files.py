import subprocess

def main():
    routes = []
    with open ("data_scraping/routes.txt", "r") as file:
        for row in file:
            row = row.strip()
            routes.append(row)

    subprocess.run(["cd", "parameters"])
    for route in routes:
        subprocess.run(["touch", f"model_route_{route}.pth"])
        subprocess.run(["touch", f"scaler_route_{route}.joblib"])
        subprocess.run(["touch", f"encoder_route_{route}.joblib"])
        subprocess.run(["touch", f"y_scaler_route_{route}.joblib"])

if __name__ == '__main__':
    main()
