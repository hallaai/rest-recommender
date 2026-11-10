"""Cuisine vocabulary for the synthetic world.

Dayparts (Spanish meal times): 0 breakfast 07-12, 1 lunch 12-17, 2 afternoon 17-20,
3 dinner 20-23, 4 late 23-02. `fit` is an additive utility bonus per daypart: it drives
when a cuisine is ordered and is part of the hidden ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass

DAYPARTS = ("breakfast", "lunch", "afternoon", "dinner", "late")
CATEGORIES = ("main", "side", "drink", "dessert")


def daypart_of_hour(hour: int) -> int:
    h = hour % 24
    if 7 <= h < 12:
        return 0
    if 12 <= h < 17:
        return 1
    if 17 <= h < 20:
        return 2
    if 20 <= h < 23:
        return 3
    return 4


@dataclass(frozen=True)
class Cuisine:
    key: str
    label: str
    popularity: float
    main_price: tuple[float, float]       # EUR range for mains at price tier 2
    fit: tuple[float, float, float, float, float]
    city_mult: tuple[float, float]        # (BCN, MAD)
    veg_share: float
    spicy_share: float
    rain: float                            # utility bonus on rainy days (comfort food)
    heat: float                            # utility bonus on hot days (> 30 C)
    hours: tuple[int, int]                 # typical opening window (open, close), close may be > 24
    examples: tuple[str, ...]              # fallback dish names when Gemini is not used


CUISINES: tuple[Cuisine, ...] = (
    Cuisine("tapas", "Spanish & tapas", 1.2, (8, 18), (-1.2, 0.4, -0.2, 0.5, -0.3), (1.0, 1.25),
            0.3, 0.1, 0.1, 0.0, (12, 24), ("Patatas bravas", "Tortilla española", "Croquetas de jamón",
                                           "Paella valenciana", "Pulpo a la gallega", "Calamares")),
    Cuisine("pizza", "Pizza", 1.4, (10, 17), (-2.0, 0.1, -0.3, 0.5, 0.5), (1.0, 1.0),
            0.35, 0.1, 0.4, -0.2, (12, 25), ("Margherita", "Diavola", "Quattro formaggi",
                                           "Prosciutto e funghi", "Capricciosa", "Tartufo")),
    Cuisine("burgers", "Burgers", 1.4, (9, 15), (-2.0, 0.2, -0.2, 0.4, 0.6), (1.0, 1.05),
            0.15, 0.15, 0.35, -0.1, (12, 25), ("Classic cheeseburger", "Bacon BBQ burger",
                                             "Smash double", "Chicken burger", "Veggie burger")),
    Cuisine("sushi", "Sushi", 1.1, (12, 28), (-2.0, 0.3, -0.5, 0.4, -0.5), (1.1, 1.0),
            0.15, 0.1, 0.0, 0.1, (12, 24), ("Salmon nigiri set", "California roll", "Spicy tuna roll",
                                          "Sashimi moriawase", "Dragon roll")),
    Cuisine("chinese", "Chinese", 0.8, (8, 14), (-1.5, 0.3, -0.2, 0.4, 0.2), (1.0, 1.0),
            0.25, 0.25, 0.25, -0.1, (12, 24), ("Kung pao chicken", "Cantonese rice",
                                             "Sweet and sour pork", "Dumplings", "Beef chow mein")),
    Cuisine("ramen", "Ramen & Japanese", 0.6, (12, 16), (-2.0, 0.4, -0.3, 0.4, 0.0), (1.1, 1.0),
            0.15, 0.3, 0.6, -0.6, (12, 24), ("Tonkotsu ramen", "Shoyu ramen", "Spicy miso ramen",
                                           "Chicken katsu curry", "Gyoza")),
    Cuisine("indian", "Indian", 0.7, (10, 16), (-1.8, 0.3, -0.3, 0.5, 0.0), (1.0, 1.0),
            0.45, 0.55, 0.35, -0.3, (12, 24), ("Butter chicken", "Chicken tikka masala",
                                             "Lamb rogan josh", "Palak paneer", "Chana masala")),
    Cuisine("mexican", "Mexican", 0.8, (9, 15), (-1.8, 0.3, 0.0, 0.5, 0.3), (1.05, 1.0),
            0.3, 0.4, 0.1, 0.1, (13, 24), ("Tacos al pastor", "Burrito de pollo", "Quesadilla",
                                         "Nachos supremos", "Enchiladas verdes")),
    Cuisine("kebab", "Kebab & Turkish", 0.8, (6, 11), (-2.0, 0.2, -0.1, 0.3, 0.9), (1.0, 1.0),
            0.2, 0.25, 0.2, -0.1, (12, 26), ("Döner kebab", "Dürüm mixto", "Lahmacun",
                                           "Falafel plate", "Shawarma box")),
    Cuisine("poke", "Poke & healthy bowls", 0.8, (11, 15), (-0.8, 0.7, 0.0, -0.1, -1.0), (1.3, 0.9),
            0.4, 0.1, -0.3, 0.6, (11, 23), ("Salmon poke", "Tuna poke", "Tofu teriyaki bowl",
                                          "Chicken quinoa bowl", "Avocado power bowl")),
    Cuisine("italian", "Italian pasta", 0.9, (10, 18), (-1.8, 0.4, -0.3, 0.4, 0.0), (1.0, 1.0),
            0.4, 0.1, 0.3, -0.2, (12, 24), ("Spaghetti carbonara", "Lasagna", "Penne arrabbiata",
                                          "Gnocchi al pesto", "Risotto ai funghi")),
    Cuisine("thai", "Thai", 0.5, (11, 16), (-1.8, 0.4, -0.2, 0.4, 0.0), (1.1, 1.0),
            0.3, 0.6, 0.2, 0.0, (12, 24), ("Pad thai", "Green curry", "Tom yum", "Massaman curry")),
    Cuisine("vietnamese", "Vietnamese", 0.4, (10, 14), (-1.5, 0.5, -0.2, 0.3, -0.2), (1.1, 1.0),
            0.3, 0.3, 0.4, -0.1, (12, 23), ("Pho bo", "Banh mi", "Bun cha", "Fresh spring rolls")),
    Cuisine("korean", "Korean", 0.4, (12, 18), (-1.8, 0.4, -0.2, 0.4, 0.2), (1.0, 1.0),
            0.2, 0.5, 0.2, 0.0, (12, 24), ("Bibimbap", "Korean fried chicken", "Bulgogi", "Kimchi jjigae")),
    Cuisine("lebanese", "Lebanese & Middle Eastern", 0.5, (9, 15), (-1.2, 0.5, 0.0, 0.3, 0.1), (1.0, 1.0),
            0.55, 0.15, 0.0, 0.2, (12, 24), ("Mixed mezze", "Chicken shawarma plate", "Falafel wrap",
                                          "Kafta skewers", "Hummus bowl")),
    Cuisine("peruvian", "Peruvian", 0.4, (14, 22), (-2.0, 0.5, -0.3, 0.4, -0.3), (0.8, 1.4),
            0.15, 0.35, 0.0, 0.2, (13, 24), ("Ceviche clásico", "Lomo saltado", "Ají de gallina",
                                           "Arroz chaufa")),
    Cuisine("chicken", "Fried chicken", 0.7, (8, 13), (-2.0, 0.2, 0.0, 0.3, 0.7), (1.0, 1.0),
            0.05, 0.3, 0.3, -0.2, (12, 26), ("Crispy chicken bucket", "Chicken tenders",
                                           "Spicy wings", "Chicken sandwich")),
    Cuisine("bakery", "Bakery & breakfast", 0.6, (4, 10), (1.8, -0.4, 0.7, -1.5, -2.5), (1.0, 1.0),
            0.6, 0.0, 0.1, 0.0, (7, 20), ("Croissant & coffee combo", "Avocado toast", "Eggs benedict",
                                        "Pancake stack", "Bocadillo de jamón")),
    Cuisine("desserts", "Desserts & ice cream", 0.5, (4, 9), (-1.0, -0.4, 1.0, 0.2, 0.3), (1.0, 1.0),
            0.8, 0.0, -0.1, 0.9, (12, 25), ("Gelato tub", "Churros con chocolate", "Cheesecake",
                                          "Brownie sundae", "Crepes")),
    Cuisine("vegan", "Vegan", 0.3, (10, 15), (-1.0, 0.6, -0.1, 0.2, -0.4), (1.2, 0.9),
            1.0, 0.2, 0.0, 0.2, (12, 23), ("Vegan burger", "Buddha bowl", "Seitan kebab",
                                         "Vegan lasagna", "Tofu curry")),
    Cuisine("arepas", "Venezuelan & arepas", 0.3, (8, 12), (-0.8, 0.4, 0.0, 0.3, 0.2), (0.7, 1.6),
            0.25, 0.1, 0.1, 0.0, (10, 24), ("Arepa reina pepiada", "Arepa pabellón",
                                          "Tequeños", "Cachapa")),
    Cuisine("grill", "Grill & Argentinian", 0.4, (15, 28), (-2.2, 0.3, -0.5, 0.6, -0.3), (1.0, 1.1),
            0.05, 0.05, 0.1, 0.0, (13, 24), ("Entraña", "Bife de chorizo", "Choripán",
                                           "Empanadas criollas", "Parrillada")),
)

CUISINE_INDEX = {c.key: i for i, c in enumerate(CUISINES)}

# Generic add-ons used by the template fallback.
FALLBACK_ADDONS = {
    "side": (("Fries", 3.5), ("Mixed salad", 4.5), ("Garlic bread", 3.0), ("Onion rings", 4.0)),
    "drink": (("Coca-Cola 33cl", 2.5), ("Still water 50cl", 1.8), ("Craft beer", 3.5),
              ("Fresh lemonade", 3.2), ("Iced tea", 2.6)),
    "dessert": (("Chocolate brownie", 4.5), ("Cheesecake slice", 5.0), ("Fruit salad", 4.0)),
}


@dataclass(frozen=True)
class Chain:
    key: str
    name: str
    cuisine: str
    price_tier: int
    outlets: tuple[int, int]   # (BCN, MAD)


# Fictional chains with outlets in both cities: the same menu (same dish names) in two
# cities lets content models carry a traveller's taste across cities.
CHAINS: tuple[Chain, ...] = (
    Chain("ch_pizza", "Forno Rosso", "pizza", 2, (6, 8)),
    Chain("ch_burger", "Smash Republic", "burgers", 2, (6, 8)),
    Chain("ch_sushi", "Koi Koi Sushi", "sushi", 2, (4, 5)),
    Chain("ch_poke", "Aloha Bowls", "poke", 2, (5, 3)),
    Chain("ch_kebab", "Anatolia Express", "kebab", 1, (3, 5)),
    Chain("ch_chicken", "Cluck & Crunch", "chicken", 1, (3, 4)),
    Chain("ch_bakery", "Pan de Casa", "bakery", 1, (3, 4)),
)

# Seven cuisine families for colour on maps and charts (eight categorical colours at most).
FAMILIES: dict[str, tuple[str, ...]] = {
    "Fast food": ("pizza", "burgers", "chicken", "kebab"),
    "Spanish & Latin": ("tapas", "mexican", "peruvian", "arepas", "grill"),
    "Asian": ("sushi", "chinese", "ramen", "thai", "vietnamese", "korean"),
    "Italian": ("italian",),
    "Indian & Middle East": ("indian", "lebanese"),
    "Healthy & vegan": ("poke", "vegan"),
    "Sweet & breakfast": ("bakery", "desserts"),
}
FAMILY_OF = {c: f for f, cs in FAMILIES.items() for c in cs}
