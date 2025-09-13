import numpy as np
from backend.core import validate_weights, max_drawdown_from_returns, value_at_risk, conditional_value_at_risk

def test_validate_weights_normalise():
    names = ["A","B","C"]
    w = validate_weights({"A":1,"B":1}, names, normalise=True)
    assert abs(w.sum() - 1.0) < 1e-8
    assert w[0] == 0.5 and w[1] == 0.5 and w[2] == 0.0

def test_mdd_simple():
    r = np.array([[1.0],[-0.5]])
    mdd = max_drawdown_from_returns(r)
    assert abs(mdd - (-0.5)) < 1e-8

def test_var_cvar_ordering():
    r = np.array([[-0.10, 0.05, -0.02, 0.01]])
    var = value_at_risk(r, alpha=0.25)
    cvar = conditional_value_at_risk(r, alpha=0.25)
    assert cvar <= var + 1e-12
