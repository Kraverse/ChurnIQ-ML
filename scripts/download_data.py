from pathlib import Path
from urllib.request import urlretrieve

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
OUT = Path("data/raw/Telco-Customer-Churn.csv")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(URL, OUT)
    print(f"Downloaded dataset to {OUT}")


if __name__ == "__main__":
    main()
