from numbers import Real
from typing import Optional

import pandas as pd


class VideoSyncMapper:
    def __init__(self, offset_ms: int = 0) -> None:
        self.offset_ms = int(offset_ms)
        self._origin_seconds: Optional[float] = None
        self._active_column: Optional[str] = None

    def reset(self) -> None:
        self._origin_seconds = None
        self._active_column = None

    def position_for_row(
        self,
        dataframe: pd.DataFrame,
        row_index: int,
        timestamp_column: Optional[str],
        fallback_interval_ms: int,
    ) -> Optional[int]:
        if dataframe is None or dataframe.empty:
            return None

        resolved_row = max(0, min(int(row_index), len(dataframe) - 1))

        if not timestamp_column or timestamp_column not in dataframe.columns:
            return max(0, resolved_row * int(fallback_interval_ms) + self.offset_ms)

        if self._active_column != timestamp_column:
            self.reset()
            self._active_column = timestamp_column

        first_value = dataframe.iloc[0][timestamp_column]
        current_value = dataframe.iloc[resolved_row][timestamp_column]

        if self._origin_seconds is None:
            self._origin_seconds = self._to_seconds(first_value, timestamp_column)

        current_seconds = self._to_seconds(current_value, timestamp_column)
        if self._origin_seconds is None or current_seconds is None:
            return None

        elapsed_seconds = max(0.0, current_seconds - self._origin_seconds)
        return max(0, int(round(elapsed_seconds * 1000.0)) + self.offset_ms)


    @staticmethod
    def _to_seconds(value, column_name: str) -> Optional[float]:
        if value is None or pd.isna(value):
            return None

        lower_name = column_name.lower()

        if isinstance(value, pd.Timedelta):
            return value.total_seconds()

        if isinstance(value, pd.Timestamp):
            return value.timestamp()

        if isinstance(value, Real):
            numeric = float(value)

            if any(token in lower_name for token in ("nanosecond", "_ns", " ns")):
                return numeric / 1_000_000_000.0
            if any(token in lower_name for token in ("microsecond", "_us", " us")):
                return numeric / 1_000_000.0
            if any(token in lower_name for token in ("millisecond", "_ms", " ms")):
                return numeric / 1000.0

            # Guess unit when it is not named
            magnitude = abs(numeric)
            if magnitude >= 1e17:
                return numeric / 1_000_000_000.0
            if magnitude >= 1e14:
                return numeric / 1_000_000.0
            if magnitude >= 1e11:
                return numeric / 1000.0

            # timestamps are treated as seconds
            return numeric

        try:
            parsed_delta = pd.to_timedelta(value)
            if not pd.isna(parsed_delta):
                return parsed_delta.total_seconds()
        except (TypeError, ValueError):
            pass

        try:
            parsed_datetime = pd.to_datetime(value)
            if not pd.isna(parsed_datetime):
                return parsed_datetime.timestamp()
        except (TypeError, ValueError, OverflowError):
            return None

        return None