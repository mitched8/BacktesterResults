import numpy as np
import pandas as pd

# Optional (only needed if you want SABR calibration). Imported lazily inside
# `calibrate_sabr_rho` so environments without scipy don't warn at import time.
least_squares = None


# -----------------------------
# Utilities
# -----------------------------
def _norm_ppf(p: float) -> float:
    """
    Inverse CDF of standard normal using a scipy-free rational approximation
    (Peter J. Acklam-style).

    Returns:
      x such that P(Z <= x) = p for Z ~ N(0,1).
    """
    import math

    if not np.isfinite(p):
        return np.nan
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf

    # Coefficients in rational approximations
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow:
        # Rational approximation for lower region
        q = math.sqrt(-2.0 * math.log(p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return num / den

    if p > phigh:
        # Rational approximation for upper region
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return -(num / den)

    # Rational approximation for central region
    q = p - 0.5
    r = q * q
    num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    return num / den


def bs_strike_from_delta(
    F: float,
    sigma: float,
    T: float,
    delta: float,
    is_call: bool = True,
    df: float = 1.0,
) -> float:
    """
    Black (forward) delta convention:
      Call delta = df * N(d1)
      Put  delta = -df * N(-d1)
    Solve for K given delta, F, sigma, T. Uses:
      d1 = [ln(F/K) + 0.5*sigma^2*T] / (sigma*sqrt(T))
      => ln(F/K) = d1*sigma*sqrt(T) - 0.5*sigma^2*T
    """
    if T <= 0 or sigma <= 0:
        return np.nan

    if is_call:
        # delta = df * N(d1)
        p = np.clip(delta / df, 1e-8, 1 - 1e-8)
        d1 = _norm_ppf(p)
    else:
        # delta = -df * N(-d1)  =>  -delta/df = N(-d1)
        p = np.clip((-delta) / df, 1e-8, 1 - 1e-8)
        minus_d1 = _norm_ppf(p)
        d1 = -minus_d1

    logFK = d1 * sigma * np.sqrt(T) - 0.5 * sigma * sigma * T
    K = F / np.exp(logFK)
    return K


# -----------------------------
# Bergomi-ish realised RR
# -----------------------------
def realised_spot_vol_stats(
    df: pd.DataFrame,
    spot_col: str = "spot",
    atm_vol_col: str = "atm_vol",
    window: int = 126,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Computes rolling realised:
      - spot log returns
      - ATM vol changes (in vol points)
      - correlation(returns, dvol)
      - vol-of-vol (annualised)
    """
    out = df.copy()

    spot = out[spot_col].astype(float)
    vol = out[atm_vol_col].astype(float)

    # daily increments
    r = np.log(spot).diff()                          # ~ daily log return
    dvol = vol.diff()                                # daily change in ATM vol (vol points)

    # rolling correlation and vol-of-vol
    rho = r.rolling(window).corr(dvol)

    dt = 1.0 / trading_days
    volvol = dvol.rolling(window).std() / np.sqrt(dt)  # annualised vol-of-vol in "vol points per year^0.5"

    out["logret"] = r
    out["d_atm_vol"] = dvol
    out["realised_rho_sv"] = rho
    out["realised_volvol"] = volvol
    return out


def bergomi_atm_skew(
    rho_sv: pd.Series,
    volvol: pd.Series,
    sigma_atm: pd.Series,
    T: float,
    kappa: float = 0.0,
) -> pd.Series:
    """
    Map realised (rho, volvol) into an ATM skew proxy.
    This is the ONE place to adjust if your desk uses a different Bergomi mapping.

    A common practical proxy is:
        skew_ATM ~ (rho * volvol / sigma_atm) * g(T,kappa)

    where g(T,kappa) is an exposure/decay factor. Here:
        g = 0.5 * sqrt(T) * (1 - exp(-kappa*T)) / (kappa*T)    if kappa>0
        g = 0.5 * sqrt(T)                                      if kappa=0

    Units:
      - volvol is annualised (vol points / sqrt(year))
      - sigma_atm is vol points (e.g. 0.10 for 10%)
      => skew comes out as "vol points per 1 log-moneyness"
    """
    sigma = sigma_atm.replace(0.0, np.nan)
    if kappa and kappa > 0:
        g = 0.5 * np.sqrt(T) * (1.0 - np.exp(-kappa * T)) / (kappa * T)
    else:
        g = 0.5 * np.sqrt(T)

    return (rho_sv * volvol / sigma) * g


def realised_rr_25d(
    df: pd.DataFrame,
    T: float,
    delta: float = 0.25,
    window: int = 126,
    spot_col: str = "spot",
    atm_vol_col: str = "atm_vol",
    trading_days: int = 252,
    kappa: float = 0.0,
) -> pd.DataFrame:
    """
    Produces a realised RR series in "vol points" comparable to quoted implied RR:
      RR_25d = vol(K_25c) - vol(K_25p)

    We approximate the smile locally as:
      vol(K) ≈ vol_ATM + skew_ATM * ln(K/F)

    So:
      RR_25d ≈ skew_ATM * [ln(Kc/F) - ln(Kp/F)]
    """
    x = realised_spot_vol_stats(
        df=df,
        spot_col=spot_col,
        atm_vol_col=atm_vol_col,
        window=window,
        trading_days=trading_days,
    )

    F = x[spot_col].astype(float)          # if you have forwards, replace spot with forward
    sigma = x[atm_vol_col].astype(float)

    skew = bergomi_atm_skew(
        rho_sv=x["realised_rho_sv"],
        volvol=x["realised_volvol"],
        sigma_atm=sigma,
        T=T,
        kappa=kappa,
    )

    # Convert delta->strikes each day using Black delta with F as forward
    Kc = []
    Kp = []
    for f, s in zip(F.values, sigma.values):
        if not np.isfinite(f) or not np.isfinite(s) or s <= 0:
            Kc.append(np.nan); Kp.append(np.nan); continue
        kc = bs_strike_from_delta(F=f, sigma=s, T=T, delta=delta, is_call=True)
        kp = bs_strike_from_delta(F=f, sigma=s, T=T, delta=-delta, is_call=False)
        Kc.append(kc); Kp.append(kp)

    x["K_25c"] = Kc
    x["K_25p"] = Kp

    # log-moneyness difference
    x["dm_25"] = np.log(x["K_25c"] / x[spot_col]) - np.log(x["K_25p"] / x[spot_col])

    # realised RR in vol points
    x["realised_rr_25d"] = skew * x["dm_25"]
    x["bergomi_skew_atm"] = skew
    return x


# -----------------------------
# SABR (Hagan) + daily calibration to get implied rho
# -----------------------------
def sabr_hagan_iv(F, K, T, alpha, beta, rho, nu):
    """Hagan 2002 SABR Black implied vol approximation."""
    F = float(F); K = float(K)
    if F <= 0 or K <= 0 or T <= 0:
        return np.nan

    one_minus_beta = 1.0 - beta
    FK = F * K
    logFK = np.log(F / K)

    # ATM branch
    if abs(logFK) < 1e-12:
        # sigma_ATM ≈ alpha / F^(1-beta) * [1 + ... T]
        f_term = F ** one_minus_beta
        if f_term <= 0:
            return np.nan
        pref = alpha / f_term

        # time correction
        c1 = (one_minus_beta**2 / 24.0) * (alpha**2) / (F**(2*one_minus_beta))
        c2 = (rho * beta * nu * alpha) / (4.0 * (F**one_minus_beta))
        c3 = (2.0 - 3.0 * rho**2) * (nu**2) / 24.0
        return pref * (1.0 + (c1 + c2 + c3) * T)

    # General strike
    z = (nu / alpha) * (FK ** (0.5 * one_minus_beta)) * logFK
    xz = np.log((np.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho))

    # Denominators
    denom = (FK ** (0.5 * one_minus_beta)) * (
        1.0
        + (one_minus_beta**2 / 24.0) * (logFK**2)
        + (one_minus_beta**4 / 1920.0) * (logFK**4)
    )

    # time correction
    c1 = (one_minus_beta**2 / 24.0) * (alpha**2) / (FK ** one_minus_beta)
    c2 = (rho * beta * nu * alpha) / (4.0 * (FK ** (0.5 * one_minus_beta)))
    c3 = (2.0 - 3.0 * rho**2) * (nu**2) / 24.0

    return (alpha / denom) * (z / xz) * (1.0 + (c1 + c2 + c3) * T)


def calibrate_sabr_rho(
    F: float,
    T: float,
    strikes: np.ndarray,
    vols: np.ndarray,
    beta: float = 0.5,
    bounds=((1e-6, -0.999, 1e-6), (5.0, 0.999, 5.0)),
    x0=(0.2, -0.3, 0.8),
):
    """
    Calibrate (alpha, rho, nu) with beta fixed.
    Returns dict with params. Needs scipy.
    """
    try:
        from scipy.optimize import least_squares as _least_squares  # type: ignore
    except Exception as e:
        raise ImportError("scipy is required for SABR calibration (pip install scipy).") from e

    strikes = np.asarray(strikes, dtype=float)
    vols = np.asarray(vols, dtype=float)

    def resid(x):
        alpha, rho, nu = x
        model = np.array([sabr_hagan_iv(F, k, T, alpha, beta, rho, nu) for k in strikes])
        return (model - vols)

    res = _least_squares(resid, x0=np.array(x0, dtype=float), bounds=bounds, method="trf")
    alpha, rho, nu = res.x
    return {"alpha": alpha, "beta": beta, "rho": rho, "nu": nu, "ok": res.success}


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Your input df must have at least: date, spot, atm_vol
    # atm_vol should be annualised in decimal (e.g., 0.10 = 10 vol)
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=400, freq="B"),
        "spot": 1.20 + 0.02 * np.cumsum(np.random.randn(400) / 100.0),
        "atm_vol": 0.10 + 0.01 * np.cumsum(np.random.randn(400) / 100.0),
    }).set_index("date")

    T = 0.5          # 6m option maturity in years
    window = 126     # ~6m rolling window

    out = realised_rr_25d(
        df=df,
        T=T,
        delta=0.25,
        window=window,
        spot_col="spot",
        atm_vol_col="atm_vol",
        kappa=0.0,     # set >0 if you want mean-reversion damping
    )

    # 'realised_rr_25d' is your comparable number (vol points)
    print(out[["realised_rr_25d", "realised_rho_sv", "realised_volvol", "bergomi_skew_atm"]].tail())

    # If you also have daily SABR smiles, calibrate rho per day and compare:
    # strikes_today = np.array([...])
    # vols_today    = np.array([...])  # Black vols for same maturity T
    # sabr = calibrate_sabr_rho(F=float(df["spot"].iloc[-1]), T=T, strikes=strikes_today, vols=vols_today, beta=0.5)
    # print("SABR rho:", sabr["rho"])