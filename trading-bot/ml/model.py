"""MLSignalModel: wraps any BaseStrategy and gates signals through a trained
RandomForestClassifier.

Workflow
--------
1. Call ``fit(df)`` on historical OHLCV data (indicators are added internally
   using the base_strategy).
2. Call ``generate_signal(df)`` at runtime.  The raw signal from the wrapped
   strategy is only forwarded when the model's predicted confidence for that
   direction meets ``min_confidence``.
"""

import os
import pickle
import sys

import pandas as pd

# Provide a clear error if sklearn is missing
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
except ImportError as _exc:
    raise ImportError(
        "scikit-learn is required for MLSignalModel.  "
        "Install it with:  pip install scikit-learn>=1.3.0"
    ) from _exc

# Make sure the trading-bot root is importable regardless of how the module is
# invoked (e.g. `python -m ml.model` from outside the package root).
_BOT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BOT_ROOT not in sys.path:
    sys.path.insert(0, _BOT_ROOT)

from strategies.base import BaseStrategy
from ml.features import build_features, label_data


# Label → column name mapping used in predict_proba
_LABEL_TO_NAME = {-1: "sell", 0: "hold", 1: "buy"}
# Signal string → label integer used to look up the confidence column
_SIGNAL_TO_LABEL = {"buy": 1, "sell": -1}


class MLSignalModel:
    """Supervised classifier wrapper for a BaseStrategy.

    Parameters
    ----------
    base_strategy:
        Any concrete ``BaseStrategy``.  Its ``add_indicators()`` and
        ``generate_signal()`` methods are used during both training and
        inference.
    min_confidence:
        Minimum predicted probability for the direction signalled by the
        base strategy.  If the model's confidence is below this threshold the
        signal is vetoed and ``generate_signal`` returns ``None``.
    """

    def __init__(
        self,
        base_strategy: BaseStrategy,
        min_confidence: float = 0.6,
    ) -> None:
        self.base_strategy = base_strategy
        self.min_confidence = min_confidence

        self._clf: RandomForestClassifier | None = None
        self._scaler: StandardScaler | None = None
        # Classes seen during training (needed for consistent probability cols)
        self._classes: list[int] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        sentiment_score: float = 0.0,
        forward_periods: int = 6,
        threshold: float = 0.01,
    ) -> "MLSignalModel":
        """Train the RandomForest on historical OHLCV data.

        Args:
            df: Raw OHLCV DataFrame (indicators will be computed internally).
            sentiment_score: Constant sentiment value broadcast to all rows.
            forward_periods: Passed to ``label_data()``.
            threshold: Passed to ``label_data()``.

        Returns:
            self (for method chaining).

        Raises:
            ValueError: If there are fewer than 2 distinct classes in the
                training labels (e.g., data is too short or too uniform).
        """
        # Add indicators in-place (work on a copy to avoid mutating caller's df)
        enriched = self.base_strategy.add_indicators(df.copy())

        X = build_features(enriched, sentiment_score=sentiment_score)
        y = label_data(enriched, forward_periods=forward_periods, threshold=threshold)

        # Align and drop rows where either X or y has no information
        mask = pd.notna(y)
        X = X.loc[mask]
        y = y.loc[mask]

        if len(X) == 0:
            raise ValueError("No valid training samples after NaN removal.")

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X.values)

        self._clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self._clf.fit(X_scaled, y.values)
        self._classes = list(self._clf.classes_)

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        df: pd.DataFrame,
        sentiment_score: float = 0.0,
    ) -> dict[str, float]:
        """Predict class probabilities for the *last row* of ``df``.

        Args:
            df: OHLCV DataFrame with indicators already computed.
            sentiment_score: Sentiment value used when building features.

        Returns:
            Dict with keys ``"buy"``, ``"sell"``, ``"hold"`` and float values
            that sum to 1.0.  Returns equal weights if the model is not fitted
            or if the feature matrix is empty.
        """
        if self._clf is None or self._scaler is None:
            return {"buy": 1 / 3, "sell": 1 / 3, "hold": 1 / 3}

        X = build_features(df, sentiment_score=sentiment_score)
        if X.empty:
            return {"buy": 1 / 3, "sell": 1 / 3, "hold": 1 / 3}

        # Use only the last row for prediction
        last_row = X.iloc[[-1]].values
        last_row_scaled = self._scaler.transform(last_row)
        proba = self._clf.predict_proba(last_row_scaled)[0]

        result: dict[str, float] = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
        for label_int, prob in zip(self._classes, proba):
            name = _LABEL_TO_NAME.get(label_int, "hold")
            result[name] = float(prob)

        return result

    def generate_signal(
        self,
        df: pd.DataFrame,
        sentiment_score: float = 0.0,
    ) -> str | None:
        """Generate a ML-gated trading signal.

        Process:
        1. Delegate to ``base_strategy.generate_signal(df)`` for the raw
           directional signal.
        2. If the raw signal is ``None``, return ``None`` immediately.
        3. Compute ML probabilities on the current bar.
        4. If the model's confidence for the signalled direction is at least
           ``min_confidence``, confirm and return the signal.
        5. Otherwise return ``None`` (ML veto).

        Args:
            df: OHLCV DataFrame with indicators already computed (or raw
                OHLCV — the base_strategy will add its indicators if absent).
            sentiment_score: Sentiment value forwarded to ``predict_proba``.

        Returns:
            ``"buy"``, ``"sell"``, or ``None``.
        """
        if self._clf is None or self._scaler is None:
            # Model not trained yet — fall back to raw strategy signal
            return self.base_strategy.generate_signal(df)

        # --- Step 1: raw signal ---
        raw_signal = self.base_strategy.generate_signal(df)
        if raw_signal is None:
            return None

        # --- Step 2: ML probabilities ---
        # Enrich the DataFrame with indicators before building features
        enriched = self.base_strategy.add_indicators(df.copy())
        if len(enriched) == 0:
            return None

        proba = self.predict_proba(enriched, sentiment_score=sentiment_score)

        # --- Step 3: confidence gate ---
        confidence = proba.get(raw_signal, 0.0)
        if confidence >= self.min_confidence:
            return raw_signal

        return None  # ML vetoes the strategy signal

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Pickle the fitted model and scaler to ``path``.

        Args:
            path: Filesystem path for the output file.
        """
        payload = {
            "clf": self._clf,
            "scaler": self._scaler,
            "classes": self._classes,
            "min_confidence": self.min_confidence,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)

    def load(self, path: str) -> "MLSignalModel":
        """Load a previously saved model from ``path``.

        Args:
            path: Filesystem path of a file written by ``save()``.

        Returns:
            self (for method chaining).
        """
        with open(path, "rb") as fh:
            payload = pickle.load(fh)

        self._clf = payload["clf"]
        self._scaler = payload["scaler"]
        self._classes = payload.get("classes", [])
        self.min_confidence = payload.get("min_confidence", self.min_confidence)
        return self
