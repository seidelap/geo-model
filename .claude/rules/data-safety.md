# Data Safety Rules

- NEVER use ACLED data for any purpose. License explicitly prohibits ML training/testing.
- NEVER commit API keys, credentials, or tokens. Use environment variables.
- NEVER store raw Common Crawl WARC files in the repo. Process and discard.
- All data paths must be configurable (env vars or config file), never hardcoded.
- GDELT/POLECAT queries must be date-bounded. Never download full history in one call.
- Parquet for all tabular storage. Never commit CSV files to the repo.
