import os
from enum import Enum

DB_PATH = os.path.join(os.path.dirname(__file__), "../db/harel.db")

COLLECTION_NAME = "harel"

class InsuranceType(Enum):
    TRAVEL = 1
    HEALTH = 2
    CAR = 3
    APARTMENT = 4
    LIFE = 5
    BUSINESS = 6
    DENTAL = 7
    MORTGAGE = 8
