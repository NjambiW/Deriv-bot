import os
from dotenv import load_dotenv

load_dotenv()

PATAPI = os.getenv("PATAPI")
APP_ID = os.getenv("APP_ID")
CLIENTid = os.getenv("CLIENTid")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join("deriv_r100_data.csv")