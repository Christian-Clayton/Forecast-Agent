import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(database_path: str | Path) -> sqlite3.Connection:
    database = str(database_path)
    if database != ":memory:":
        Path(database).parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(database)
    if database != ":memory:":
        journal_mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()
        if journal_mode is None or journal_mode[0].lower() != "wal":
            connection.close()
            raise sqlite3.OperationalError(
                f"Unable to enable WAL journal mode for {database!r}"
            )

    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    return connection
