import os
import pandas as pd

from scraper import scrape_books
from database import load_dataframe
from queries import (
    execute_queries,
    pandas_merge_equivalent,
    validate_join_equivalence
)


OUTPUT_FILE = "sql_outputs.txt"


def save_query_outputs(results):

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        for query_name, df in results.items():

            file.write(
                "\n"
                + "=" * 80
                + "\n"
            )

            file.write(
                query_name
                + "\n"
            )

            file.write(
                "=" * 80
                + "\n"
            )

            file.write(
                df.to_string(index=False)
            )

            file.write("\n\n")


def main():

    print(
        "=" * 80
    )

    print(
        "ZEpto Data Pipeline"
    )

    print(
        "=" * 80
    )

    # ------------------------------------------------
    # STEP 1: SCRAPE
    # ------------------------------------------------

    print(
        "\n[1/5] Scraping books..."
    )

    df = scrape_books()

    print(
        f"Collected {len(df)} books."
    )

    print(
        f"Categories: {df['category'].nunique()}"
    )

    # ------------------------------------------------
    # STEP 2: VALIDATE CLEANED DATA
    # ------------------------------------------------

    print(
        "\n[2/5] Validating cleaned dataset..."
    )

    print(
        df[
            [
                "title",
                "price_gbp",
                "price_inr",
                "rating",
                "in_stock",
                "category"
            ]
        ].head()
    )

    print(
        "\nData types:"
    )

    print(
        df.dtypes
    )

    # ------------------------------------------------
    # STEP 3: DATABASE
    # ------------------------------------------------

    print(
        "\n[3/5] Loading SQLite database..."
    )

    connection = load_dataframe(
        df
    )

    print(
        "Database created successfully."
    )

    # ------------------------------------------------
    # STEP 4: SQL QUERIES
    # ------------------------------------------------

    print(
        "\n[4/5] Executing SQL queries..."
    )

    results = execute_queries(
        connection
    )

    save_query_outputs(
        results
    )

    # ------------------------------------------------
    # STEP 5: PANDAS MERGE
    # ------------------------------------------------

    print(
        "\n[5/5] Validating SQL JOIN with pandas.merge..."
    )

    sql_join_result = results[
        "query_6_join"
    ]

    pandas_join_result = pandas_merge_equivalent(
        connection,
        df
    )

    print(
        "\nSQL JOIN result:"
    )

    print(
        sql_join_result.head(10).to_string(
            index=False
        )
    )

    print(
        "\nPandas merge result:"
    )

    print(
        pandas_join_result.head(10).to_string(
            index=False
        )
    )

    equivalent = validate_join_equivalence(
        sql_join_result,
        pandas_join_result
    )

    print(
        f"\nSQL JOIN == pandas.merge(): "
        f"{equivalent}"
    )

    # ------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "PIPELINE COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 80
    )

    print(
        f"Books loaded       : {len(df)}"
    )

    print(
        f"Categories         : {df['category'].nunique()}"
    )

    print(
        f"SQLite database    : books.db"
    )

    print(
        f"SQL output file    : {OUTPUT_FILE}"
    )

    print(
        f"JOIN validation    : {equivalent}"
    )

    connection.close()


if __name__ == "__main__":
    main()
