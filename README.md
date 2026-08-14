# Data Pipeline — Zepto Capstone

## Overview

This module implements an end-to-end data engineering pipeline for catalog-style product data.

The pipeline performs the following steps:

1. Scrapes book data from Books to Scrape.
2. Cleans and validates the scraped fields.
3. Converts GBP prices to INR using the required fixed project rate.
4. Stores the cleaned data in a normalized SQLite database.
5. Executes SQL queries demonstrating filtering, sorting, limiting, distinct values, range filtering and joins.
6. Reads SQL results into pandas DataFrames.
7. Reproduces the SQL JOIN using `pandas.merge()` and validates that both approaches produce equivalent results.

The project uses the same raw-to-relational workflow described in the capstone brief: raw scraped data is transformed into a clean relational store for downstream analytics.

---

## Project Structure

```text
data_pipeline/
│
├── scraper.py
├── database.py
├── queries.py
├── run_pipeline.py
├── requirements.txt
├── README.md
├── sql_outputs.txt
└── books.db
```

---

## Data Source

The data source is:

`http://books.toscrape.com/`

Books to Scrape is a public website intended for scraping practice.

No login, API key or paid service is required.

---

## Scraping Scope

The pipeline dynamically discovers book categories and scrapes complete category pagination until at least:

* 60 books
* 3 different categories

have been collected.

This ensures that the final dataset satisfies the assignment requirement while avoiding hard-coded product data.

---

## Fields Collected

The scraper initially collects:

| Field          | Description                                       |
| -------------- | ------------------------------------------------- |
| `title`        | Book title                                        |
| `price`        | Original price text in GBP                        |
| `star_rating`  | Rating text such as One, Two, Three, Four or Five |
| `availability` | Original availability text                        |
| `category`     | Book category                                     |

The cleaning stage creates the required normalized fields:

| Field       | Type    | Description                  |
| ----------- | ------- | ---------------------------- |
| `price_gbp` | float   | Numeric GBP price            |
| `price_inr` | float   | Converted INR price          |
| `rating`    | integer | Rating from 1 to 5           |
| `in_stock`  | boolean | Whether the book is in stock |
| `category`  | string  | Category name                |

---

## Currency Conversion

The required fixed project baseline is:

**1 GBP = 105.50 INR**

This is an artificial project-defined conversion rate.

No live currency API is required or used.

The INR price is calculated as:

```text
price_inr = price_gbp × 105.50
```

The result is rounded to two decimal places.

---

## Cleaning Decisions

### Price

The currency symbol and other non-numeric characters are removed before conversion to `float`.

Example:

```text
£51.77 → 51.77
```

If a price cannot be parsed, the numeric value is represented as missing initially and median imputation is applied.

### Rating

The website provides textual ratings:

```text
One
Two
Three
Four
Five
```

These are converted using:

```text
One   → 1
Two   → 2
Three → 3
Four  → 4
Five  → 5
```

If a numeric rating cannot be parsed, the median rating is used.

### Availability

Availability is converted to a Boolean value.

If the availability text contains `In stock`, the resulting value is:

```text
True
```

Otherwise:

```text
False
```

### Missing essential fields

Rows without an identifiable title or category are dropped because these fields are required to uniquely identify and classify a book.

For numeric fields, median imputation is used instead of dropping otherwise usable rows.

---

## Database Design

SQLite is used as the relational database.

The database contains two normalized tables.

### categories

```sql
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT NOT NULL UNIQUE
);
```

### books

```sql
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
);
```

The relationship is:

```text
categories
    │
    │ category_id
    │
    └──────────< books.category_id
```

This avoids storing the category name repeatedly in every database row.

---

## SQL Queries

The pipeline executes six SQL queries.

They collectively demonstrate:

* `SELECT`
* `WHERE`
* `ORDER BY`
* `LIMIT`
* `DISTINCT`
* `BETWEEN`
* `IN`
* `JOIN`

The executed outputs are stored in:

```text
sql_outputs.txt
```

The JOIN query combines the `books` and `categories` tables using:

```sql
JOIN categories c
    ON b.category_id = c.category_id
```

---

## Pandas Validation

At least two SQL query results are loaded using:

```python
pd.read_sql()
```

The JOIN result is independently reproduced using:

```python
pd.merge()
```

The SQL and pandas results are sorted using the same ordering columns and compared using:

```python
DataFrame.equals()
```

The pipeline prints whether both results are equivalent.

---

## Installation

From the `data_pipeline` directory:

```bash
python -m venv venv
```

Activate the environment.

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

Run:

```bash
python run_pipeline.py
```

The pipeline will:

1. Scrape the website.
2. Clean the data.
3. Convert GBP to INR.
4. Create `books.db`.
5. Populate the normalized tables.
6. Execute the SQL queries.
7. Save query outputs to `sql_outputs.txt`.
8. Reproduce the JOIN using pandas.
9. Compare the SQL JOIN and pandas JOIN results.

The pipeline is designed to run from scratch without manual copy-pasting of data.

---

## Reproducibility

The SQLite tables are recreated each time the pipeline runs.

Therefore:

```bash
python run_pipeline.py
```

is sufficient to regenerate the database from the source website.

---

## Design Summary

The design follows a simple ETL pattern:

```text
Books to Scrape
       │
       ▼
   Requests
       │
       ▼
 BeautifulSoup
       │
       ▼
 Raw DataFrame
       │
       ▼
 Cleaning / Validation
       │
       ▼
 GBP → INR
       │
       ▼
 Normalized SQLite
       │
       ├──────────────┐
       ▼              ▼
      SQL          pandas
       │              │
       └──────┬───────┘
              ▼
        Result Validation
```

The implementation deliberately uses a fixed conversion rate rather than an external currency API because the assignment defines 1 GBP = 105.50 INR as the required grading baseline.

## Currency Conversion

The project-defined fixed conversion rate is:

1 GBP = 105.50 INR

This is an artificial fixed baseline supplied by the assignment.
No live currency API is used.

price_inr = price_gbp × 105.50
