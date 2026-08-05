import pandas as pd


def clip_negative_first_row(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with negative values in the first row clipped to zero."""
    clipped_df = df.copy()
    clipped_df.iloc[0] = clipped_df.iloc[0].clip(lower=0)
    return clipped_df
