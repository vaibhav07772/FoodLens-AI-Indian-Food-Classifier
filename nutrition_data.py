"""
nutrition_data.py
------------------
Nutrition + metadata lookup for the Indian Food dataset classes
(see dataset/README.md for the dataset source).

For each dish, per-100g approximate values are given for: calories
(kcal), protein (g), carbs (g), fat (g), calcium (mg) — plus two
extra fields that make the result more genuinely useful for an
Indian-food app:

    diet        -> "Vegetarian" or "Non-Vegetarian" (shown as a
                   green/red dot, like the mandatory FSSAI mark on
                   Indian packaged food)
    name_hindi  -> the dish name in Hindi (Devanagari), for a
                   bilingual result card

IMPORTANT: these are ballpark/typical values compiled from general
nutrition knowledge, NOT lab-measured figures for your exact photo —
real values vary with recipe, oil quantity, and portion size. Good
enough for a college project's "nutrition estimate" feature; for
sourced figures, swap get_nutrition() to call a real API (IFCT 2017 /
USDA FoodData Central / Nutritionix) using the label as the query.
"""

# label (matches model/labels.txt) -> nutrition + metadata, per 100g
FOOD_NUTRITION = {
    "biryani": {"calories": 200, "protein_g": 8, "carbs_g": 25, "fat_g": 7, "calcium_mg": 20,
                "diet": "Non-Vegetarian", "name_hindi": "बिरयानी"},
    "butter_chicken": {"calories": 280, "protein_g": 18, "carbs_g": 8, "fat_g": 20, "calcium_mg": 40,
                        "diet": "Non-Vegetarian", "name_hindi": "बटर चिकन"},
    "chappati": {"calories": 297, "protein_g": 8, "carbs_g": 50, "fat_g": 6, "calcium_mg": 20,
                 "diet": "Vegetarian", "name_hindi": "चपाती"},
    "chicken_tandoori": {"calories": 150, "protein_g": 25, "carbs_g": 2, "fat_g": 5, "calcium_mg": 20,
                          "diet": "Non-Vegetarian", "name_hindi": "तंदूरी चिकन"},
    "chole_bhature": {"calories": 350, "protein_g": 10, "carbs_g": 45, "fat_g": 15, "calcium_mg": 60,
                       "diet": "Vegetarian", "name_hindi": "छोले भटूरे"},
    "dal_makhani": {"calories": 150, "protein_g": 7, "carbs_g": 15, "fat_g": 8, "calcium_mg": 50,
                     "diet": "Vegetarian", "name_hindi": "दाल मखनी"},
    "dhokla": {"calories": 160, "protein_g": 6, "carbs_g": 25, "fat_g": 4, "calcium_mg": 30,
               "diet": "Vegetarian", "name_hindi": "ढोकला"},
    "dosa": {"calories": 168, "protein_g": 4, "carbs_g": 28, "fat_g": 4, "calcium_mg": 15,
             "diet": "Vegetarian", "name_hindi": "डोसा"},
    "gajar_halwa": {"calories": 300, "protein_g": 4, "carbs_g": 40, "fat_g": 14, "calcium_mg": 100,
                     "diet": "Vegetarian", "name_hindi": "गाजर का हलवा"},
    "ghevar": {"calories": 400, "protein_g": 4, "carbs_g": 55, "fat_g": 18, "calcium_mg": 60,
               "diet": "Vegetarian", "name_hindi": "घेवर"},
    "gulab_jamun": {"calories": 387, "protein_g": 5, "carbs_g": 52, "fat_g": 18, "calcium_mg": 90,
                     "diet": "Vegetarian", "name_hindi": "गुलाब जामुन"},
    "idli": {"calories": 132, "protein_g": 4, "carbs_g": 27, "fat_g": 1, "calcium_mg": 15,
             "diet": "Vegetarian", "name_hindi": "इडली"},
    "jalebi": {"calories": 375, "protein_g": 2, "carbs_g": 60, "fat_g": 14, "calcium_mg": 20,
               "diet": "Vegetarian", "name_hindi": "जलेबी"},
    "kadai_paneer": {"calories": 220, "protein_g": 10, "carbs_g": 10, "fat_g": 16, "calcium_mg": 250,
                      "diet": "Vegetarian", "name_hindi": "कड़ाही पनीर"},
    "kathi_roll": {"calories": 250, "protein_g": 12, "carbs_g": 28, "fat_g": 10, "calcium_mg": 50,
                    "diet": "Non-Vegetarian", "name_hindi": "काठी रोल"},
    "kofta": {"calories": 220, "protein_g": 6, "carbs_g": 18, "fat_g": 14, "calcium_mg": 80,
              "diet": "Vegetarian", "name_hindi": "कोफ्ता"},
    "masala_bhindi": {"calories": 120, "protein_g": 3, "carbs_g": 10, "fat_g": 8, "calcium_mg": 60,
                       "diet": "Vegetarian", "name_hindi": "मसाला भिंडी"},
    "meduvada": {"calories": 180, "protein_g": 6, "carbs_g": 20, "fat_g": 8, "calcium_mg": 30,
                 "diet": "Vegetarian", "name_hindi": "मेदु वड़ा"},
    "pani_puri": {"calories": 150, "protein_g": 3, "carbs_g": 28, "fat_g": 3, "calcium_mg": 15,
                  "diet": "Vegetarian", "name_hindi": "पानी पूरी"},
    "pav_bhaji": {"calories": 200, "protein_g": 5, "carbs_g": 28, "fat_g": 8, "calcium_mg": 40,
                  "diet": "Vegetarian", "name_hindi": "पाव भाजी"},
    "poori": {"calories": 330, "protein_g": 5, "carbs_g": 35, "fat_g": 18, "calcium_mg": 15,
              "diet": "Vegetarian", "name_hindi": "पूरी"},
    "rasgulla": {"calories": 186, "protein_g": 4, "carbs_g": 33, "fat_g": 4, "calcium_mg": 60,
                 "diet": "Vegetarian", "name_hindi": "रसगुल्ला"},
    "samosa": {"calories": 260, "protein_g": 5, "carbs_g": 30, "fat_g": 13, "calcium_mg": 20,
               "diet": "Vegetarian", "name_hindi": "समोसा"},
    "toor_dal": {"calories": 120, "protein_g": 7, "carbs_g": 18, "fat_g": 2, "calcium_mg": 25,
                 "diet": "Vegetarian", "name_hindi": "तूर दाल"},
    "vada_pav": {"calories": 280, "protein_g": 6, "carbs_g": 40, "fat_g": 10, "calcium_mg": 30,
                 "diet": "Vegetarian", "name_hindi": "वड़ा पाव"},
}

DEFAULT_NUTRITION = {
    "calories": None, "protein_g": None, "carbs_g": None, "fat_g": None, "calcium_mg": None,
    "diet": None, "name_hindi": None,
}


def get_nutrition(label: str) -> dict:
    """Looks up nutrition + metadata for a predicted food label.
    Matching is case/format-tolerant (spaces or hyphens -> underscore)."""
    key = label.strip().lower().replace(" ", "_").replace("-", "_")
    return FOOD_NUTRITION.get(key, DEFAULT_NUTRITION)


def calories_to_walk_minutes(calories_per_100g: int, portion_g: int = 150) -> int:
    """Rough 'burn it off' estimate: brisk walking burns ~280 kcal/hour
    for an average adult (~4.7 kcal/min). portion_g is an assumed
    typical serving size (150g) since nutrition values above are per
    100g. Purely illustrative — varies a lot by body weight and pace."""
    if calories_per_100g is None:
        return None
    total_calories = calories_per_100g * (portion_g / 100)
    return round(total_calories / 4.7)
