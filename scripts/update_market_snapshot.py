"""Explicitly refresh and record real market data; never run during imports."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from portfolio.market_data import fetch_live_market, SNAPSHOT_DIR

if __name__ == "__main__":
    prices, (_, _, _, metadata) = fetch_live_market()
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    csv_path = SNAPSHOT_DIR/"cn_etf_prices.csv"
    prices.to_csv(csv_path, index_label="date", float_format="%.8f", encoding="utf-8", lineterminator="\n")
    metadata["csv_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    (SNAPSHOT_DIR/"cn_etf_snapshot.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"start": metadata["start_date"], "end": metadata["end_date"],
                      "observations": metadata["observations"], "provider": metadata["provider"]}, ensure_ascii=False))
