from __future__ import annotations

from pathlib import Path


def _sql_path(path: Path) -> str:
    # SQL string literal escaping; paths originate from config, not user row data.
    return str(path.resolve()).replace("'", "''")


def build_features(raw_dir: str | Path, processed_dir: str | Path, config: dict):
    import duckdb

    raw, out = Path(raw_dir), Path(processed_dir)
    out.mkdir(parents=True, exist_ok=True)
    if config["data"].get("source", "home_credit") == "uci_credit_card":
        # Public, credential-free benchmark used for a reproducible end-to-end run.
        # UCI distributes one XLS workbook; preserve its history columns as observed.
        import urllib.request
        import zipfile
        from .data import file_sha256

        archive = raw / "uci_default_credit_card.zip"
        workbook = raw / "default of credit card clients.xls"
        raw.mkdir(parents=True, exist_ok=True)
        if not workbook.exists():
            if not archive.exists():
                urllib.request.urlretrieve(config["data"].get("uci_url", "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"), archive)
            with zipfile.ZipFile(archive) as zf:
                member = next(n for n in zf.namelist() if n.lower().endswith(".xls"))
                workbook.write_bytes(zf.read(member))
        import pandas as pd
        frame = pd.read_excel(workbook, header=1)
        frame.columns = [str(c).strip().upper().replace(" ", "_") for c in frame.columns]
        frame = frame.rename(columns={"ID": "SK_ID_CURR", "DEFAULT_PAYMENT_NEXT_MONTH": "TARGET"})
        if "TARGET" not in frame:
            # UCI's original workbook uses this exact label; fail closed if schema changes.
            raise ValueError(f"UCI target column not found; columns={frame.columns.tolist()}")
        target = out / "application_features.parquet"
        frame.to_parquet(target, index=False)
        (out / "source_manifest.json").write_text(__import__("json").dumps({
            "source": "UCI Default of Credit Card Clients", "source_url": config["data"].get("uci_url"),
            "sha256": file_sha256(workbook), "rows": len(frame), "target": "default payment next month",
        }, indent=2))
        return target
    app = raw / config["data"]["application_train"]
    bureau = raw / config["data"]["bureau"]
    prior = raw / config["data"]["previous_application"]
    for path in (app, bureau, prior):
        if not path.exists():
            raise FileNotFoundError(f"Required competition data file missing: {path}")
    template_path = Path(__file__).parents[2] / "sql" / "aggregate_features.sql"
    if not template_path.exists():
        template_path = Path.cwd() / "sql" / "aggregate_features.sql"
    if not template_path.exists():
        raise FileNotFoundError("Could not locate sql/aggregate_features.sql")
    template = template_path.read_text()
    query = template.replace("{{application_csv}}", _sql_path(app))
    query = query.replace("{{bureau_csv}}", _sql_path(bureau))
    query = query.replace("{{previous_application_csv}}", _sql_path(prior))
    target = out / "application_features.parquet"
    con = duckdb.connect()
    try:
        con.execute(f"COPY ({query}) TO '{_sql_path(target)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    finally:
        con.close()
    return target


def read_features(path: str | Path):
    import duckdb
    con = duckdb.connect()
    try:
        return con.execute("SELECT * FROM read_parquet(?)", [str(Path(path).resolve())]).fetchdf()
    finally:
        con.close()
