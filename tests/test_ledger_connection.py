from forecast_agent.ledger.ledger import connect


def test_ledger_connection_enables_wal_and_initializes_schema(tmp_path) -> None:
    connection = connect(tmp_path / "ledger.sqlite3")
    try:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

        assert journal_mode == "wal"
        assert {"questions", "forecasts", "resolutions", "scores"} <= tables
    finally:
        connection.close()
