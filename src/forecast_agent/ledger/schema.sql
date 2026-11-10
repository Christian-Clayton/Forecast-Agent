PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS questions (
    id TEXT PRIMARY KEY,
    raw_text TEXT NOT NULL,
    falsifiable_text TEXT NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('binary', 'quantile', 'count_range')),
    resolution_criteria TEXT NOT NULL,
    resolution_source TEXT NOT NULL,
    resolves_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS forecasts (
    id TEXT PRIMARY KEY,
    question_id TEXT NOT NULL REFERENCES questions(id),
    created_at TEXT NOT NULL,
    as_of TEXT NOT NULL,
    domain_class TEXT NOT NULL CHECK (domain_class IN ('thin', 'fat', 'mixed')),
    classification_reasoning TEXT NOT NULL,
    toolkit TEXT NOT NULL CHECK (toolkit IN ('A', 'B')),
    probability REAL,
    quantiles_json TEXT,
    base_rate REAL,
    base_rate_source TEXT,
    ensemble_members_json TEXT NOT NULL DEFAULT '[]',
    ensemble_spread REAL,
    tool_call_ids_json TEXT NOT NULL DEFAULT '[]',
    skill_ids_loaded_json TEXT NOT NULL DEFAULT '[]',
    cost_usd REAL,
    latency_s REAL,
    trace_id TEXT
);

CREATE TRIGGER IF NOT EXISTS forecasts_no_update
BEFORE UPDATE ON forecasts
BEGIN
    SELECT RAISE(ABORT, 'forecasts are append-only');
END;

CREATE TRIGGER IF NOT EXISTS forecasts_no_delete
BEFORE DELETE ON forecasts
BEGIN
    SELECT RAISE(ABORT, 'forecasts are append-only');
END;

CREATE TABLE IF NOT EXISTS resolutions (
    question_id TEXT PRIMARY KEY REFERENCES questions(id),
    resolved_at TEXT NOT NULL,
    outcome_json TEXT NOT NULL,
    resolver_note TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scores (
    forecast_id TEXT PRIMARY KEY REFERENCES forecasts(id),
    brier REAL,
    log_score REAL,
    pinball REAL,
    baseline_brier REAL,
    skill_score REAL
);
