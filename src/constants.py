import sys
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "../db/harel.db")

COLLECTION_NAME = "harel"

INSURANCE_TYPES = [
    "Travel",
    "Health",
    "Car",
    "Apartment",
    "Life",
    "Business",
    "Dental",
    "Mortgage",
]
