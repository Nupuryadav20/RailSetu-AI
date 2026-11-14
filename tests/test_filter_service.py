
import pandas as pd

from backend.app.services.filter_service import find_column


def test_find_column_not_found():
    df = pd.DataFrame({
        "department": ["Engineering"],
        "priority_level": ["High"]
    })

    result = find_column(df, ["unknown_column", "missing_column"])

    assert result is None
