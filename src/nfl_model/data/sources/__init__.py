"""Per-source ingestors. Each module owns one upstream and writes to the
warehouse via :func:`nfl_model.data.warehouse.upsert_dataframe`.
"""
