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

    @classmethod
    def from_string(cls, name):
        if name == "travel":
            return cls.TRAVEL
        if name == "health":
            return cls.HEALTH
        if name == "car":
            return cls.CAR
        if name == "apartment":
            return cls.APARTMENT
        if name == "life":
            return cls.LIFE
        if name == "business":
            return cls.BUSINESS
        if name == "dental":
            return cls.DENTAL
        if name == "mortgage":
            return cls.MORTGAGE


class SourceType(Enum):
    FAQ = 1
    PDF = 2
    POLICY = 3
    BLOG = 4
