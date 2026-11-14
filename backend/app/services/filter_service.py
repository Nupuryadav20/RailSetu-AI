
import re
import pandas as pd


def find_column(df, candidates):
    columns = {column.lower(): column for column in df.columns}

    for candidate in candidates:
        if candidate.lower() in columns:
            return columns[candidate.lower()]

    return None


def paginate_dataframe(
    df,
    page=1,
    page_size=10,
    search=None,
    department=None,
    priority=None,
    corridor=None
):
    filtered_df = df.copy()

    if search:
        search_pattern = re.escape(search)
        mask = filtered_df.astype(str).apply(
            lambda column: column.str.contains(
                search_pattern,
                case=False,
                na=False
            )
        ).any(axis=1)

        filtered_df = filtered_df[mask]

    if department:
        column = find_column(filtered_df, ["department"])
        if column:
            filtered_df = filtered_df[
                filtered_df[column].astype(str).str.lower()
                == department.lower()
            ]

    if priority:
        column = find_column(
            filtered_df,
            ["priority_level", "final_priority_level"]
        )
        if column:
            filtered_df = filtered_df[
                filtered_df[column].astype(str).str.lower()
                == priority.lower()
            ]

    if corridor:
        column = find_column(
            filtered_df,
            ["corridor_id", "corridor"]
        )
        if column:
            filtered_df = filtered_df[
                filtered_df[column].astype(str).str.lower()
                == corridor.lower()
            ]

    total_records = len(filtered_df)
    total_pages = (total_records + page_size - 1) // page_size

    start = (page - 1) * page_size
    end = start + page_size

    items = filtered_df.iloc[start:end].to_dict(orient="records")

    return {
        "items": items,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_records": total_records,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_previous": page > 1
        }
    }
