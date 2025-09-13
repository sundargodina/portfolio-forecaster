# backend/schemas.py
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

# ---- Put your valid asset universe here ----
VALID_ASSETS = [
    "US Equity",
    "International Equity",
    "Global Equity",
    "US Small Cap",
    "International Small Cap",
    "EM Debt",
    "Inflation-Protected Bond",
    "Cash"
]

# ---- Request payload used by all portfolio endpoints ----
class PortfolioWeights(BaseModel):
    weights: Dict[str, float] = Field(
        ...,
        example={
            "US Equity": 0.6,
            "Cash": 0.2,
            "Global Equity": 0.2
        },
        description=f"Mapping of asset_name -> weight. Valid assets: {VALID_ASSETS}"
    )
    normalise: bool = Field(
        True,
        example=True,
        description="If true, weights will be normalised to sum to 1 before calculations."
    )

# Keep names compatible with existing app.py usage
class WeightsPayload(PortfolioWeights):
    pass

class AnnualisedReturnPayload(WeightsPayload):
    pass

class AnnualisedVolPayload(WeightsPayload):
    pass

class SharpePayload(WeightsPayload):
    risk_free: float = Field(0.0, example=0.02, description="Risk-free rate used in Sharpe calculation")

class TrackingErrorPayload(WeightsPayload):
    benchmark_weights: Dict[str, float] = Field(
        ...,
        example={"US Equity": 0.5, "Global Equity": 0.5},
        description="Weights dict for the benchmark (same valid asset names)."
    )
    normalise_benchmark: bool = Field(True, example=True)

class DownsideDevPayload(WeightsPayload):
    mar: float = Field(0.0, example=0.0, description="Minimum acceptable return (MAR) for downside deviation")

class VaRPayload(WeightsPayload):
    alpha: float = Field(0.05, gt=0, lt=1, example=0.05, description="VaR tail probability (e.g., 0.05 for 5%)")

class CVaRPayload(VaRPayload):
    pass

class MaxDrawdownPayload(WeightsPayload):
    pass

class ProjectionPayload(WeightsPayload):
    start_value: float = Field(100.0, gt=0, example=100000.0, description="Starting portfolio value")
    percentiles: Optional[List[float]] = Field(
        default_factory=lambda: [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
        example=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
        description="List of percentiles to return for projection chart"
    )

# ---- Response models to make docs explicit and neat ----
class ForecastResponse(BaseModel):
    """Generic single-value forecast response (keeps legacy key names below)."""
    # Many endpoints return different keys; we'll make endpoint-specific classes below.

class AnnualisedReturnResponse(BaseModel):
    annualised_return: float = Field(..., example=0.062)

class AnnualisedVolatilityResponse(BaseModel):
    annualised_volatility: float = Field(..., example=0.12)

class SharpeResponse(BaseModel):
    annualised_return: float = Field(..., example=0.062)
    annualised_volatility: float = Field(..., example=0.12)
    risk_free: float = Field(..., example=0.02)
    sharpe_ratio: float = Field(..., example=0.35)

class TrackingErrorResponse(BaseModel):
    tracking_error: float = Field(..., example=0.04)

class DownsideDevResponse(BaseModel):
    downside_deviation: float = Field(..., example=0.07)
    mar: float = Field(..., example=0.0)

class VaRResponse(BaseModel):
    value_at_risk: float = Field(..., example=-0.05)
    alpha: float = Field(..., example=0.05)

class CVaRResponse(BaseModel):
    conditional_value_at_risk: float = Field(..., example=-0.07)
    alpha: float = Field(..., example=0.05)

class MaxDrawdownResponse(BaseModel):
    maximum_drawdown: float = Field(..., example=-0.25)

class ProjectionResponse(BaseModel):
    years: List[int] = Field(..., example=[0,1,2,3,4,5])
    percentiles: Dict[str, List[float]] = Field(
        ...,
        example={
            "0.01": [90000, 92000, 94000],
            "0.5":  [100000,110000,120000]
        }
    )

class AssetMetric(BaseModel):
    asset_name: str
    annualised_return: float
    annualised_volatility: float

class AssetsMetricsResponse(BaseModel):
    metrics: List[AssetMetric]

class CorrelationResponse(BaseModel):
    correlation: Dict[str, Dict[str, float]] = Field(
        ...,
        example={
            "US Equity": {"US Equity": 1.0, "Cash": 0.05},
            "Cash": {"US Equity": 0.05, "Cash": 1.0}
        }
    )

class AssetsListResponse(BaseModel):
    assets: List[str] = Field(..., example=VALID_ASSETS)
    shape: List[int] = Field(..., example=[8,30,1000], description="(assets, years, paths)")
    categories: List[Dict[str, Any]] = Field(..., example=[{"asset_name":"US Equity","category":"Equity"}])
