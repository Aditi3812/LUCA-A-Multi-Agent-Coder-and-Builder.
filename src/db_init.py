from src.db import init_db


def main() -> None:
    init_db()
    print("Database schema initialized successfully.")


if __name__ == "__main__":
    main()
