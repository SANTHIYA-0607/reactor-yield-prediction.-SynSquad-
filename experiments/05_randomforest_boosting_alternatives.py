"""Documentation entry point for the reported Random Forest + Boosting alternatives.

The supplied experiment report summarizes the best result in this category as
17.0097 ± 2.3465 RMSE. It is a category summary, not one combined algorithm.
"""

BEST_REPORTED = {
    "category": "Random Forest + Boosting Alternatives",
    "best_validation_rmse": 17.0097,
    "std": 2.3465,
    "note": "Category-level summary from supplied report; not a single combined model.",
}

if __name__ == "__main__":
    print(BEST_REPORTED)
