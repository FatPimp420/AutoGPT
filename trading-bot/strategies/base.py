from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> str | None:
        """Return 'buy', 'sell', or None."""
        ...

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Override to add TA indicators to the dataframe before signal logic."""
        return df
