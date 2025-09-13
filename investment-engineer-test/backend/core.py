#core.py
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import h5py

class SimulationData:
    def __init__(self, hdf5_path: Path, categories_csv: Path):
        self.hdf5_path = Path(hdf5_path)
        self.categories_csv = Path(categories_csv)
        self.asset_names: List[str] = []
        self.sim_returns: Optional[np.ndarray] = None
        self.categories: Optional[pd.DataFrame] = None
        self._load()

    def _load(self):
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")
        if not self.categories_csv.exists():
            raise FileNotFoundError(f"Categories CSV not found: {self.categories_csv}")
        with h5py.File(self.hdf5_path, "r") as f:
            raw = f["asset_names"][()]
            if isinstance(raw, np.ndarray) and raw.dtype.kind in ("S", "O"):
                self.asset_names = [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in raw]
            else:
                self.asset_names = list(map(str, raw.tolist()))
            self.sim_returns = f["asset_class_projections/simulated_return"][()]
        if self.sim_returns.ndim != 3:
            raise ValueError("simulated_return must be 3D: (assets, years, paths)")
        if len(self.asset_names) != self.sim_returns.shape[0]:
            raise ValueError("asset_names length must equal first dim of simulated_return")
        df = pd.read_csv(self.categories_csv)
        if not set(["asset_name", "category"]).issubset(df.columns):
            raise ValueError("asset_categories.csv must have columns: asset_name, category")
        all_assets = pd.DataFrame({"asset_name": self.asset_names})
        self.categories = all_assets.merge(df, how="left", on="asset_name")
        self.categories["category"] = self.categories["category"].fillna("Uncategorised")

    def asset_index(self, name: str) -> int:
        try:
            return self.asset_names.index(name)
        except ValueError:
            raise KeyError(f"Unknown asset: {name}")

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.sim_returns.shape)

def validate_weights(weights: Dict[str, float], asset_names: List[str], normalise: bool = True) -> np.ndarray:
    if not isinstance(weights, dict) or len(weights) == 0:
        raise ValueError("weights must be a non-empty dict mapping asset_name -> weight")
    unknown = [k for k in weights.keys() if k not in asset_names]
    if unknown:
        raise KeyError(f"Unknown assets in weights: {unknown}. Valid example names: {asset_names[:8]}{'...' if len(asset_names)>8 else ''}")
    w = np.zeros(len(asset_names), dtype=float)
    for i, a in enumerate(asset_names):
        if a in weights:
            w[i] = float(weights[a])
    if normalise:
        s = w.sum()
        if s <= 0:
            raise ValueError("Sum of provided weights must be > 0 when normalise=True")
        w = w / s
    else:
        s = w.sum()
        if not np.isclose(s, 1.0, atol=1e-8):
            raise ValueError(f"Sum of weights must be 1.0 when normalise=False (got {s:.6f})")
    return w

def portfolio_returns(sim_returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.tensordot(weights, sim_returns, axes=(0,0))

def annualised_return(returns: np.ndarray) -> float:
    years, paths = returns.shape
    growth = (1 + returns).prod(axis=0) ** (1.0 / max(years,1)) - 1.0
    return float(np.mean(growth))

def annualised_volatility(returns: np.ndarray) -> float:
    stds = returns.std(axis=0, ddof=1)
    return float(np.mean(stds))

def sharpe_ratio(ann_ret: float, ann_vol: float, rf: float = 0.0) -> float:
    if ann_vol == 0:
        return float('nan')
    return (ann_ret - rf) / ann_vol

def tracking_error(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    if returns.shape != benchmark_returns.shape:
        raise ValueError("returns and benchmark_returns must have same shape (years, paths)")
    active = returns - benchmark_returns
    stds = active.std(axis=0, ddof=1)
    return float(np.mean(stds))

def downside_deviation(returns: np.ndarray, mar: float = 0.0) -> float:
    downside = np.minimum(0.0, returns - mar)
    var = (downside ** 2).mean(axis=0)
    return float(np.sqrt(np.mean(var)))

def value_at_risk(returns: np.ndarray, alpha: float = 0.05) -> float:
    flat = returns.flatten()
    return float(np.quantile(flat, alpha))

def conditional_value_at_risk(returns: np.ndarray, alpha: float = 0.05) -> float:
    flat = returns.flatten()
    var = np.quantile(flat, alpha)
    vals = flat[flat <= var]
    return float(vals.mean()) if vals.size>0 else float(var)

def max_drawdown_from_returns(returns: np.ndarray) -> float:
    years, paths = returns.shape
    mdds = []
    for p in range(paths):
        r = returns[:, p]
        cum = (1 + r).cumprod()
        peak = np.maximum.accumulate(cum)
        dd = (cum / peak) - 1.0
        mdds.append(float(dd.min()))
    return float(np.mean(mdds))

def project_values(returns: np.ndarray, start_value: float = 100.0) -> np.ndarray:
    years, paths = returns.shape
    cum = np.vstack([np.ones(paths), (1 + returns).cumprod(axis=0)])
    return start_value * cum

def per_asset_stats(sim_returns: np.ndarray, asset_names: List[str]):
    assets, years, paths = sim_returns.shape
    rows = []
    for i, name in enumerate(asset_names):
        r = sim_returns[i]
        growth = (1 + r).prod(axis=0) ** (1.0 / max(years,1)) - 1.0
        ann_ret = float(growth.mean())
        vol = float(r.std(axis=0, ddof=1).mean())
        rows.append({"asset_name": name, "annualised_return": ann_ret, "annualised_volatility": vol})
    return rows

def asset_correlation(sim_returns: np.ndarray, asset_names: List[str]):
    assets, years, paths = sim_returns.shape
    flat = sim_returns.reshape(assets, years*paths)
    corr = np.corrcoef(flat)
    import pandas as pd
    df = pd.DataFrame(corr, index=asset_names, columns=asset_names)
    return df
