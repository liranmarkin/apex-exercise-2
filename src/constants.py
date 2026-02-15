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

# TODO: User must replace this with their actual key
# https://github.com/nebius/token-factory-cookbook/blob/main/api/api_native.ipynb
NEBIUS_API_KEY = "v1.CmMKHHN0YXRpY2tleS1lMDBldHBiMzYyY3JuMngxcXYSIXNlcnZpY2VhY2NvdW50LWUwMGtieTJqN2p6ajljYXJuczILCKeFo8wGEL26q1s6DAiliLuXBxDA0NfVA0ACWgNlMDA.AAAAAAAAAAGNSitzi_mVnjLQCBIM0OeiIYDXqXQJwYLBqfTkFWqTVMAo_oZW5fhZCxCmfkh7rz9-U72xMILMxWQ7a8fAxkYG"
