from dotenv import load_dotenv

from src.config import get_database_url


def main() -> None:
    load_dotenv()

    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: psycopg. Install project dependencies first."
        ) from exc

    db_url = get_database_url()

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            result = cur.fetchone()

    print("Neon DB connection successful.")
    print(f"SELECT 1 result: {result}")


if __name__ == "__main__":
    main()
