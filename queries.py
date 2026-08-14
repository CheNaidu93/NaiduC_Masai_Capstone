import sqlite3
import pandas as pd


QUERIES = {

    "query_1_select_where": """
        SELECT
            title,
            price_gbp,
            rating,
            in_stock
        FROM books
        WHERE rating >= 4
        ORDER BY rating DESC
    """,

    "query_2_order_by_limit": """
        SELECT
            title,
            price_gbp,
            rating
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10
    """,

    "query_3_distinct_categories": """
        SELECT DISTINCT
            category_id
        FROM books
        ORDER BY category_id
    """,

    "query_4_between_price": """
        SELECT
            title,
            price_gbp,
            price_inr
        FROM books
        WHERE price_gbp BETWEEN 10 AND 30
        ORDER BY price_gbp
    """,

    "query_5_in_ratings": """
        SELECT
            title,
            rating,
            in_stock
        FROM books
        WHERE rating IN (4, 5)
        ORDER BY rating DESC, title
    """,

    "query_6_join": """
        SELECT
            c.category_name,
            b.title,
            b.price_gbp,
            b.price_inr,
            b.rating,
            b.in_stock
        FROM books b
        JOIN categories c
            ON b.category_id = c.category_id
        ORDER BY
            c.category_name,
            b.rating DESC,
            b.title
    """
}


def execute_queries(connection):

    results = {}

    for query_name, query in QUERIES.items():

        print("\n" + "=" * 80)

        print(query_name)

        print("=" * 80)

        print(query.strip())

        df = pd.read_sql(
            query,
            connection
        )

        results[query_name] = df

        print("\nOutput:")

        print(df.to_string(index=False))

    return results


def pandas_merge_equivalent(connection, raw_df):

    # ---------------------------------------------
    # Get the category table from SQLite
    # ---------------------------------------------

    categories_df = pd.read_sql(
        """
        SELECT
            category_id,
            category_name
        FROM categories
        """,
        connection
    )

    # ---------------------------------------------
    # Prepare books dataframe
    # ---------------------------------------------

    books_df = raw_df.copy()

    # Map category name to category ID
    category_map = dict(
        zip(
            categories_df["category_name"],
            categories_df["category_id"]
        )
    )

    books_df["category_id"] = (
        books_df["category"]
        .map(category_map)
    )

    # ---------------------------------------------
    # Reproduce JOIN using pandas.merge()
    # ---------------------------------------------

    merged_df = pd.merge(
        books_df,
        categories_df,
        on="category_id",
        how="inner"
    )

    merged_df = merged_df[
        [
            "category_name",
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock"
        ]
    ]

    merged_df = merged_df.sort_values(
        by=[
            "category_name",
            "rating",
            "title"
        ],
        ascending=[
            True,
            False,
            True
        ]
    ).reset_index(drop=True)

    return merged_df


def validate_join_equivalence(
    sql_join_df,
    pandas_join_df
):

    sql_result = sql_join_df.copy()

    pandas_result = pandas_join_df.copy()

    sql_result = sql_result.sort_values(
        by=[
            "category_name",
            "rating",
            "title"
        ],
        ascending=[
            True,
            False,
            True
        ]
    ).reset_index(drop=True)

    pandas_result = pandas_result.sort_values(
        by=[
            "category_name",
            "rating",
            "title"
        ],
        ascending=[
            True,
            False,
            True
        ]
    ).reset_index(drop=True)

    return sql_result.equals(
        pandas_result
    )
