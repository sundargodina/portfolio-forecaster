# backend/app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import numpy as np

from backend.core import (
    SimulationData, validate_weights, portfolio_returns, annualised_return,
    annualised_volatility, sharpe_ratio, tracking_error, downside_deviation,
    value_at_risk, conditional_value_at_risk, max_drawdown_from_returns,
    project_values, per_asset_stats, asset_correlation
)
# Import new models
from backend.schemas import (
    AnnualisedReturnPayload, AnnualisedReturnResponse,
    AnnualisedVolPayload, AnnualisedVolatilityResponse,
    SharpePayload, SharpeResponse,
    TrackingErrorPayload, TrackingErrorResponse,
    DownsideDevPayload, DownsideDevResponse,
    VaRPayload, VaRResponse,
    CVaRPayload, CVaRResponse,
    MaxDrawdownPayload, MaxDrawdownResponse,
    ProjectionPayload, ProjectionResponse,
    AssetsListResponse, AssetsMetricsResponse, CorrelationResponse
)

DATA_DIR = Path(__file__).resolve().parent / "Data"
HDF5_PATH = DATA_DIR / "base_simulation.hdf5"
CATEGORIES_PATH = DATA_DIR / "asset_categories.csv"

app = FastAPI(
    title="Forecast Statistics API - Clean",
    version="1.1.0",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1}
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data once
SIM = None
LOAD_ERROR = None
try:
    SIM = SimulationData(HDF5_PATH, CATEGORIES_PATH)
except Exception as e:
    LOAD_ERROR = str(e)

@app.get("/health")
def health():
    return {"status": "ok", "data_loaded": SIM is not None, "load_error": LOAD_ERROR}

@app.get("/assets", response_model=AssetsListResponse)
def assets():
    if SIM is None:
        raise HTTPException(status_code=500, detail=f"Data not loaded: {LOAD_ERROR}")
    return {"assets": SIM.asset_names, "shape": list(SIM.shape), "categories": SIM.categories.to_dict(orient='records')}

def compute_portfolio_returns(payload) -> np.ndarray:
    if SIM is None:
        raise HTTPException(status_code=500, detail=f"Data not loaded: {LOAD_ERROR}")
    try:
        w = validate_weights(payload.weights, SIM.asset_names, normalise=payload.normalise)
    except KeyError as e:
        # KeyError contains message and recommended valid examples
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return portfolio_returns(SIM.sim_returns, w)

@app.post("/forecast_statistic/annualised_return", response_model=AnnualisedReturnResponse)
def endpoint_ann_return(payload: AnnualisedReturnPayload):
    R = compute_portfolio_returns(payload)
    val = annualised_return(R)
    return {"annualised_return": float(val)}

@app.post("/forecast_statistic/annualised_volatility", response_model=AnnualisedVolatilityResponse)
def endpoint_ann_vol(payload: AnnualisedVolPayload):
    R = compute_portfolio_returns(payload)
    val = annualised_volatility(R)
    return {"annualised_volatility": float(val)}

@app.post("/forecast_statistic/sharpe_ratio", response_model=SharpeResponse)
def endpoint_sharpe(payload: SharpePayload):
    R = compute_portfolio_returns(payload)
    ann_r = annualised_return(R)
    ann_v = annualised_volatility(R)
    sr = sharpe_ratio(ann_r, ann_v, rf=payload.risk_free)
    return {
        "annualised_return": float(ann_r),
        "annualised_volatility": float(ann_v),
        "risk_free": float(payload.risk_free),
        "sharpe_ratio": float(sr)
    }

@app.post("/forecast_statistic/tracking_error", response_model=TrackingErrorResponse)
def endpoint_te(payload: TrackingErrorPayload):
    if SIM is None:
        raise HTTPException(status_code=500, detail=f"Data not loaded: {LOAD_ERROR}")
    R = compute_portfolio_returns(payload)
    try:
        wb = validate_weights(payload.benchmark_weights, SIM.asset_names, normalise=payload.normalise_benchmark)
    except KeyError as e:
        raise HTTPException(status_code=400, detail="Benchmark: " + str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail="Benchmark: " + str(e))
    Rb = portfolio_returns(SIM.sim_returns, wb)
    te = tracking_error(R, Rb)
    return {"tracking_error": float(te)}

@app.post("/forecast_statistic/downside_deviation", response_model=DownsideDevResponse)
def endpoint_dd(payload: DownsideDevPayload):
    R = compute_portfolio_returns(payload)
    dd = downside_deviation(R, mar=payload.mar)
    return {"downside_deviation": float(dd), "mar": float(payload.mar)}

@app.post("/forecast_statistic/value_at_risk", response_model=VaRResponse)
def endpoint_var(payload: VaRPayload):
    R = compute_portfolio_returns(payload)
    var = value_at_risk(R, alpha=payload.alpha)
    return {"value_at_risk": float(var), "alpha": float(payload.alpha)}

@app.post("/forecast_statistic/conditional_value_at_risk", response_model=CVaRResponse)
def endpoint_cvar(payload: CVaRPayload):
    R = compute_portfolio_returns(payload)
    cvar = conditional_value_at_risk(R, alpha=payload.alpha)
    return {"conditional_value_at_risk": float(cvar), "alpha": float(payload.alpha)}

@app.post("/forecast_statistic/maximum_drawdown", response_model=MaxDrawdownResponse)
def endpoint_mdd(payload: MaxDrawdownPayload):
    R = compute_portfolio_returns(payload)
    mdd = max_drawdown_from_returns(R)
    return {"maximum_drawdown": float(mdd)}

@app.post("/projection/percentiles", response_model=ProjectionResponse)
def endpoint_projection(payload: ProjectionPayload):
    R = compute_portfolio_returns(payload)
    V = project_values(R, start_value=payload.start_value)
    years = list(range(V.shape[0]))
    pcts = payload.percentiles or [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    pct_curves = {}
    for q in pcts:
        pct_curves[str(q)] = np.quantile(V, q, axis=1).tolist()
    return {"years": years, "percentiles": pct_curves}

@app.get("/assets/metrics", response_model=AssetsMetricsResponse)
def endpoint_asset_metrics():
    if SIM is None:
        raise HTTPException(status_code=500, detail=f"Data not loaded: {LOAD_ERROR}")
    rows = per_asset_stats(SIM.sim_returns, SIM.asset_names)
    return {"metrics": rows}

@app.get("/assets/correlation", response_model=CorrelationResponse)
def endpoint_asset_correlation():
    if SIM is None:
        raise HTTPException(status_code=500, detail=f"Data not loaded: {LOAD_ERROR}")
    corr = asset_correlation(SIM.sim_returns, SIM.asset_names)
    corr = corr.fillna(0)
    # convert DataFrame to nested dict (string keys)
    corr_dict = corr.to_dict()
    return {"correlation": corr_dict}
