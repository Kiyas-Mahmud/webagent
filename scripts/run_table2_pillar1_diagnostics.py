"""Run the source-attested offline P1 companion diagnostic package."""

try:
    from scripts.table2_companion_diagnostic_bootstrap import main
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from table2_companion_diagnostic_bootstrap import main


if __name__ == "__main__":
    main("P1")
