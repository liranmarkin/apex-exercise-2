"""
Mock RAG — implements the same interface as RAG but returns hardcoded data.

Matches the real RAG interface (query_collection)
so it can be swapped in anywhere the real one is used.
"""

from constants import InsuranceType


class MockRAG:
    def __init__(self, reset_collection: bool = True):
        """Initialize mock RAG with hardcoded data."""
        self.hardcoded_data = {
            InsuranceType.CAR.value: [
                {
                    "entity": {
                        "full_doc": "ביטוח רכב - כיסוי נזקים. כיסוי מלא לרכב המבוטח כולל נזקים הנגרמים בשימוש חלק שלו. למעט אם השימוש היה לצורך הנעת כלי חיצוני אחר.",
                        "url": "https://www.harel.co.il/car-insurance/coverage",
                        "page_index": 12
                    }
                },
                {
                    "entity": {
                        "full_doc": "ביטוח רכב - שיפור וחידושים. התביעה תכוסה רק אם הנזק נגרם במקרה הביטוח שהוגדר בפוליסה.",
                        "url": "https://www.harel.co.il/car-insurance/terms",
                        "page_index": 25
                    }
                }
            ],
            InsuranceType.LIFE.value: [
                {
                    "entity": {
                        "full_doc": "ביטוח חיים - קצבה שנתית. קצבה חודשית המשולמת במהלך חיי המבוטח או המוטבים.",
                        "url": "https://www.harel.co.il/life-insurance/benefits",
                        "page_index": 5
                    }
                },
                {
                    "entity": {
                        "full_doc": "ביטוח חיים - תנאים כלליים. התביעה תביא לתשלום התגמול המיוחד ליורשים של המבוטח.",
                        "url": "https://www.harel.co.il/life-insurance/conditions",
                        "page_index": 18
                    }
                }
            ],
            InsuranceType.HEALTH.value: [
                {
                    "entity": {
                        "full_doc": "ביטוח בריאות - כיסוי אשפוז. כיסוי מלא לעלויות אשפוז בבית חולים מורשה.",
                        "url": "https://www.harel.co.il/health-insurance/hospitalization",
                        "page_index": 8
                    }
                },
                {
                    "entity": {
                        "full_doc": "ביטוח בריאות - טיפול רופא. כיסוי ביקורי רופא מומחה בטל-אביב ותל-אביב",
                        "url": "https://www.harel.co.il/health-insurance/medical",
                        "page_index": 15
                    }
                }
            ],
            InsuranceType.APARTMENT.value: [
                {
                    "entity": {
                        "full_doc": "ביטוח דירה - כיסוי תוכן. כיסוי לתוכן הדירה כנגד סיכונים של שריפה, גניבה וחבלה.",
                        "url": "https://www.harel.co.il/apartment-insurance/contents",
                        "page_index": 30
                    }
                },
                {
                    "entity": {
                        "full_doc": "ביטוח דירה - אחריות צד שלישי. כיסוי אחריות בגין נזקים שנגרמו לצדדים שלישיים.",
                        "url": "https://www.harel.co.il/apartment-insurance/liability",
                        "page_index": 42
                    }
                }
            ],
            InsuranceType.TRAVEL.value: [
                {
                    "entity": {
                        "full_doc": "ביטוח נסיעות - ביטול נסיעה. כיסוי הוצאות ביטול בעקבות אירוע בלתי צפוי.",
                        "url": "https://www.harel.co.il/travel-insurance/cancellation",
                        "page_index": 3
                    }
                },
                {
                    "entity": {
                        "full_doc": "ביטוח נסיעות - כיסוי רפואי בחו״ל. כיסוי עלויות רפואיות במהלך הנסיעה בחו״ל.",
                        "url": "https://www.harel.co.il/travel-insurance/medical",
                        "page_index": 11
                    }
                }
            ],
            InsuranceType.BUSINESS.value: [
                {
                    "entity": {
                        "full_doc": "ביטוח עסק - כיסוי רכוש. כיסוי לרכוש העסק כנגד סיכונים שונים.",
                        "url": "https://www.harel.co.il/business-insurance/property",
                        "page_index": 7
                    }
                },
                {
                    "entity": {
                        "full_doc": "ביטוח עסק - אחריות מעבידים. כיסוי אחריות כלפי עובדים של המעבידים.",
                        "url": "https://www.harel.co.il/business-insurance/employer-liability",
                        "page_index": 22
                    }
                }
            ],
            InsuranceType.DENTAL.value: [
                {
                    "entity": {
                        "full_doc": "ביטוח שיניים - טיפולי שיניים שגרתיים. כיסוי לניקיון שיניים ופטום דופן בשנה.",
                        "url": "https://www.harel.co.il/dental-insurance/routine",
                        "page_index": 4
                    }
                },
                {
                    "entity": {
                        "full_doc": "ביטוח שיניים - טיפולי שיניים מקיפים. כיסוי לטיפולים שונים כולל שיקום שיניים.",
                        "url": "https://www.harel.co.il/dental-insurance/comprehensive",
                        "page_index": 16
                    }
                }
            ],
            InsuranceType.MORTGAGE.value: [
                {
                    "entity": {
                        "full_doc": "ביטוח משכנתא - פיצוי בפקדון. כיסוי בגין הפסד הפקדון במקרה פדיון מוקדם.",
                        "url": "https://www.harel.co.il/mortgage-insurance/deposit",
                        "page_index": 6
                    }
                },
                {
                    "entity": {
                        "full_doc": "ביטוח משכנתא - חיזוק נתינויות. כיסוי בגין הנתינויות של הביטוח.",
                        "url": "https://www.harel.co.il/mortgage-insurance/terms",
                        "page_index": 19
                    }
                }
            ]
        }

    def query_collection(self, insurance_type, query: str, maximal_docs: int = 2):
        """Query the mock database and return hardcoded results."""
        # Handle InsuranceType enum
        if hasattr(insurance_type, 'value'):
            type_value = insurance_type.value
        else:
            type_value = insurance_type
        
        # Return hardcoded data for the insurance type
        if type_value in self.hardcoded_data:
            results = self.hardcoded_data[type_value][:maximal_docs]
            return results
        
        # Default fallback
        return []
