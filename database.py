import sqlite3
import pandas as pd


DATABASE_PATH = "books.db"


def create_database(connection):
    """
    Create normalized categories and books tables.
    """

    cursor = connection.cursor()

    # Enable foreign key enforcement
    cursor.execute(
        "PRAGMA foreign_keys = ON"
    )

    # Drop existing tables so the pipeline is reproducible
    cursor.execute(
        "DROP TABLE IF EXISTS books"
    )

    cursor.execute(
        "DROP TABLE IF EXISTS categories"
    )

    # ---------------------------------------------
    # CATEGORIES TABLE
    # ---------------------------------------------

    cursor.execute(
        """
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL UNIQUE
        )
        """
    )

    # ---------------------------------------------
    # BOOKS TABLE
    # ---------------------------------------------

    cursor.execute(
        """
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock INTEGER NOT NULL,
            category_id INTEGER NOT NULL,

            FOREIGN KEY (category_id)
                REFERENCES categories(category_id),

            CHECK (rating BETWEEN 1 AND 5),

            CHECK (in_stock IN (0, 1))
        )
        """
    )

    connection.commit()


def insert_categories(connection, df):

    categories = sorted(
        df["category"].unique()
    )

    cursor = connection.cursor()

    cursor.executemany(
        """
        INSERT INTO categories(category_name)
        VALUES (?)
        """,
        [
            (category,)
            for category in categories
        ]
    )

    connection.commit()


def insert_books(connection, df):

    # Build mapping:
    # category_name -> category_id

    category_df = pd.read_sql(
        """
        SELECT
            category_id,
            category_name
        FROM categories
        """,
        connection
    )

    category_map = dict(
        zip(
            category_df["category_name"],
            category_df["category_id"]
        )
    )

    cursor = connection.cursor()

    records = []

    for _, row in df.iterrows():

        records.append(
            (
                row["title"],
                float(row["price_gbp"]),
                float(row["price_inr"]),
                int(row["rating"]),
                int(row["in_stock"]),
                category_map[row["category"]]
            )
        )

    cursor.executemany(
        """
        INSERT INTO books (
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        records
    )

    connection.commit()


def load_dataframe(df):

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    try:

        create_database(connection)

        insert_categories(
            connection,
            df
        )

        insert_books(
            connection,
            df
        )

        return connection

    except Exception:

        connection.close()

        raise