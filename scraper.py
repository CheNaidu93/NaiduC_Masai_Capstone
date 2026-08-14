import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin


BASE_URL = "http://books.toscrape.com/"
GBP_TO_INR = 105.50

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; DataPipelineAssignment/1.0)"
}


def get_soup(url):
    """
    Download a webpage and return a BeautifulSoup object.
    Raises an exception if the request fails.
    """

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=20
    )

    response.raise_for_status()

    return BeautifulSoup(response.text, "html.parser")


def get_categories():
    """
    Extract category names and URLs from the left navigation menu.
    """

    soup = get_soup(BASE_URL)

    categories = {}

    category_links = soup.select(
        "div.side_categories ul li ul li a"
    )

    for link in category_links:

        category_name = link.get_text(strip=True)

        category_url = urljoin(
            BASE_URL,
            link.get("href")
        )

        categories[category_name] = category_url

    return categories


def parse_price(price_text):
    """
    Convert price such as '£51.77' into float 51.77.
    """

    try:
        cleaned = re.sub(
            r"[^0-9.]",
            "",
            price_text
        )

        return float(cleaned)

    except (ValueError, TypeError):

        return None


def parse_rating(rating_text):
    """
    Convert textual rating into integer.
    """

    if rating_text is None:
        return None

    rating_text = rating_text.strip()

    return RATING_MAP.get(rating_text)


def parse_availability(availability_text):
    """
    Convert availability text into boolean.

    'In stock' -> True
    anything else -> False
    """

    if not availability_text:
        return False

    return "in stock" in availability_text.lower()


def scrape_category(category_name, category_url):
    """
    Scrape every paginated page belonging to one category.
    """

    records = []

    current_url = category_url

    while current_url:

        soup = get_soup(current_url)

        products = soup.select(
            "article.product_pod"
        )

        for product in products:

            # Title
            title_element = product.select_one("h3 a")

            title = (
                title_element.get("title", "").strip()
                if title_element
                else None
            )

            # Price
            price_element = product.select_one(
                "p.price_color"
            )

            price_text = (
                price_element.get_text(strip=True)
                if price_element
                else None
            )

            # Rating
            rating_element = product.select_one(
                "p.star-rating"
            )

            star_rating = None

            if rating_element:

                classes = rating_element.get("class", [])

                for class_name in classes:

                    if class_name in RATING_MAP:
                        star_rating = class_name
                        break

            # Availability
            availability_element = product.select_one(
                "p.instock.availability"
            )

            availability = (
                availability_element.get_text(
                    " ",
                    strip=True
                )
                if availability_element
                else None
            )

            records.append(
                {
                    "title": title,
                    "price": price_text,
                    "star_rating": star_rating,
                    "availability": availability,
                    "category": category_name
                }
            )

        # Find next page
        next_link = soup.select_one(
            "li.next a"
        )

        if next_link:

            current_url = urljoin(
                current_url,
                next_link.get("href")
            )

        else:

            current_url = None

    return records


def clean_data(df):
    """
    Clean scraped fields and create derived columns.
    """

    # -------------------------------------------------
    # PRICE
    # -------------------------------------------------

    df["price_gbp"] = df["price"].apply(
        parse_price
    )

    # -------------------------------------------------
    # RATING
    # -------------------------------------------------

    df["rating"] = df["star_rating"].apply(
        parse_rating
    )

    # -------------------------------------------------
    # AVAILABILITY
    # -------------------------------------------------

    df["in_stock"] = df["availability"].apply(
        parse_availability
    )

    # -------------------------------------------------
    # NUMERIC IMPUTATION
    # -------------------------------------------------

    if df["price_gbp"].isna().any():

        median_price = df["price_gbp"].median()

        df["price_gbp"] = df["price_gbp"].fillna(
            median_price
        )

    if df["rating"].isna().any():

        median_rating = df["rating"].median()

        df["rating"] = df["rating"].fillna(
            round(median_rating)
        )

    # -------------------------------------------------
    # DROP ROWS WHERE ESSENTIAL TEXT DATA IS MISSING
    # -------------------------------------------------

    df = df.dropna(
        subset=[
            "title",
            "category"
        ]
    )

    # -------------------------------------------------
    # TYPES
    # -------------------------------------------------

    df["price_gbp"] = df["price_gbp"].astype(float)

    df["rating"] = df["rating"].astype(int)

    df["in_stock"] = df["in_stock"].astype(bool)

    # -------------------------------------------------
    # FIXED PROJECT CURRENCY RATE
    # -------------------------------------------------

    df["price_inr"] = (
        df["price_gbp"] * GBP_TO_INR
    ).round(2)

    # -------------------------------------------------
    # SELECT FINAL COLUMNS
    # -------------------------------------------------

    df = df[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category",
            "star_rating",
            "availability"
        ]
    ]

    return df


def scrape_books(
    minimum_books=60,
    minimum_categories=3
):

    categories = get_categories()

    print(
        f"Found {len(categories)} categories."
    )

    all_records = []

    # We scrape categories until both requirements
    # are satisfied.

    for category_name, category_url in categories.items():

        print(
            f"Scraping category: {category_name}"
        )

        records = scrape_category(
            category_name,
            category_url
        )

        all_records.extend(records)

        current_categories = len(
            set(
                record["category"]
                for record in all_records
            )
        )

        print(
            f"Books collected: {len(all_records)}"
        )

        if (
            len(all_records) >= minimum_books
            and
            current_categories >= minimum_categories
        ):
            break

    df = pd.DataFrame(all_records)

    df = clean_data(df)

    # Safety validation
    if len(df) < minimum_books:

        raise ValueError(
            f"Only {len(df)} books were collected. "
            f"At least {minimum_books} are required."
        )

    if df["category"].nunique() < minimum_categories:

        raise ValueError(
            "Fewer than "
            f"{minimum_categories} categories were collected."
        )

    return df


if __name__ == "__main__":

    df = scrape_books()

    print("\nFinal dataset:")
    print(df.head())

    print(
        f"\nTotal books: {len(df)}"
    )

    print(
        f"Categories: {df['category'].nunique()}"
    )

    print(
        "\nData types:"
    )

    print(
        df.dtypes
    )
