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


# https://github.com/nebius/token-factory-cookbook/blob/main/api/api_native.ipynb
NEBIUS_API_KEY = "v1.CmQKHHN0YXRpY2tleS1lMDB0OGUzZDNzeDhkeXI4bnESIXNlcnZpY2VhY2NvdW50LWUwMHA1emhzNDg1bjRuZmFwMjIMCLqfyMwGEOSgybUCOgwIuaLglwcQgMOX2gJAAloDZTAw.AAAAAAAAAAGemZGoeFt6Ku8C0uYiN4JtJhgL1bUgdwSSkAgACu5DcC-3WAETfyToGkbFnGvIB3B-sVJTZQDH7nqtBV5S2XUB"
