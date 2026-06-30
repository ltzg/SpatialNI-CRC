from __future__ import annotations

"""
Step 04 - Progression and survival analysis.

Run this after patient-level SpatialNI tables are built. It provides the
statistical backbone for stage association, survival, and treatment-stratified
analyses.
"""

import math
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.stats import chi2, mannwhitneyu, spearmanr

def compare_binary_endpoint(feature: pd.Series, endpoint: pd.Series) -> dict[str, float]:
    frame = pd.DataFrame({"feature": feature, "endpoint": endpoint}).dropna()
    a = frame.loc[frame["endpoint"] == 1, "feature"].to_numpy(float)
    b = frame.loc[frame["endpoint"] == 0, "feature"].to_numpy(float)
    if len(a) == 0 or len(b) == 0:
        return {"n": float(len(frame)), "effect": np.nan, "p_value": np.nan}
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    return {"n": float(len(frame)), "effect": float(2.0 * (u / (len(a) * len(b))) - 1.0), "p_value": float(p)}

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))

def fit_logistic_effect(feature: pd.Series, endpoint: pd.Series) -> dict[str, float]:
    frame = pd.DataFrame({"x": feature, "y": endpoint}).dropna()
    x = pd.to_numeric(frame["x"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(frame["y"], errors="coerce").to_numpy(float)
    keep = np.isin(y, [0.0, 1.0])
    x, y = x[keep], y[keep]
    if len(x) < 10 or len(np.unique(y)) < 2 or float(np.nanstd(x)) < 1e-8:
        return {"n": float(len(x)), "beta": np.nan, "se": np.nan, "or": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}
    design = np.column_stack([np.ones(len(x)), x]); beta = np.zeros(2)
    for _ in range(100):
        eta = design @ beta; prob = np.clip(_sigmoid(eta), 1e-6, 1 - 1e-6); w = prob * (1 - prob)
        xtwx = design.T @ (w[:, None] * design); xtwz = design.T @ (w * (eta + (y - prob) / w))
        try: beta_new = np.linalg.solve(xtwx, xtwz)
        except np.linalg.LinAlgError: beta_new = None
        if beta_new is None: break
        if np.max(np.abs(beta_new - beta)) < 1e-8: beta = beta_new; break
        beta = beta_new
    try: cov = np.linalg.inv(xtwx)
    except Exception: cov = None
    if cov is None or not np.isfinite(cov[1, 1]) or cov[1, 1] <= 0:
        return {"n": float(len(x)), "beta": np.nan, "se": np.nan, "or": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}
    slope = float(beta[1]); se = float(np.sqrt(cov[1, 1])); z = slope / se; p = float(chi2.sf(z ** 2, 1))
    return {"n": float(len(x)), "beta": slope, "se": se, "or": float(np.exp(slope)), "ci_low": float(np.exp(slope - 1.96 * se)), "ci_high": float(np.exp(slope + 1.96 * se)), "p_value": p}

def fixed_effect_meta_analysis(beta: np.ndarray, se: np.ndarray) -> dict[str, float]:
    beta = np.asarray(beta, float); se = np.asarray(se, float); keep = np.isfinite(beta) & np.isfinite(se) & (se > 0); beta, se = beta[keep], se[keep]
    if len(beta) == 0:
        return {"k": 0.0, "beta": np.nan, "se": np.nan, "effect": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}
    w = 1.0 / np.square(se); pooled = float(np.sum(w * beta) / np.sum(w)); pooled_se = float(np.sqrt(1.0 / np.sum(w))); z = pooled / pooled_se
    return {"k": float(len(beta)), "beta": pooled, "se": pooled_se, "effect": float(np.exp(pooled)), "ci_low": float(np.exp(pooled - 1.96 * pooled_se)), "ci_high": float(np.exp(pooled + 1.96 * pooled_se)), "p_value": float(chi2.sf(z ** 2, 1))}

def binary_endpoint_meta_analysis(table: pd.DataFrame, feature_col: str, endpoint_col: str, center_col: str) -> pd.Series:
    per_center = table.groupby(center_col, sort=False).apply(lambda g: pd.Series(fit_logistic_effect(g[feature_col], g[endpoint_col]))).reset_index()
    pooled = fixed_effect_meta_analysis(per_center["beta"].to_numpy(float), per_center["se"].to_numpy(float))
    return pd.Series({"centers_used": float(per_center["beta"].notna().sum()), "combined_or": pooled["effect"], "ci_low": pooled["ci_low"], "ci_high": pooled["ci_high"], "p_value": pooled["p_value"]})

def stage_trend_spearman(feature: pd.Series, ordered_stage: pd.Series) -> dict[str, float]:
    frame = pd.DataFrame({"feature": feature, "stage": ordered_stage}).dropna()
    if len(frame) < 4: return {"n": float(len(frame)), "rho": np.nan, "p_value": np.nan}
    rho, p = spearmanr(frame["feature"], frame["stage"])
    return {"n": float(len(frame)), "rho": float(rho), "p_value": float(p)}

def cox_nll_grad_hess(beta: np.ndarray, x: np.ndarray, times: np.ndarray, events: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    p = x.shape[1]; loglik = 0.0; grad = np.zeros(p); hess = np.zeros((p, p))
    for t in np.sort(np.unique(times[events == 1])):
        risk = times >= t; event = (times == t) & (events == 1); d = int(event.sum())
        if d == 0: continue
        xr = x[risk]; eta = xr @ beta; eta_max = float(np.max(eta)); w = np.exp(eta - eta_max); denom = float(np.sum(w))
        s1 = np.sum(xr * w[:, None], axis=0); s2 = xr.T @ (xr * w[:, None]); xe = np.sum(x[event], axis=0)
        loglik += float(xe @ beta) - d * (math.log(denom) + eta_max); grad += xe - d * (s1 / denom); hess -= d * (s2 / denom - np.outer(s1 / denom, s1 / denom))
    return -loglik, -grad, -hess

def fit_cox_model(x: np.ndarray, times: np.ndarray, events: np.ndarray, feature_names: list[str] | None = None) -> dict[str, object]:
    x = np.asarray(x, float); times = np.asarray(times, float); events = np.asarray(events, int)
    keep = np.isfinite(times) & np.isfinite(events) & (times > 0) & np.isin(events, [0, 1]) & np.isfinite(x).all(axis=1)
    x, times, events = x[keep], times[keep], events[keep]
    if len(times) == 0 or int(events.sum()) == 0:
        p = x.shape[1]; summary = pd.DataFrame({"feature": feature_names or list(range(p)), "beta": np.nan, "hr": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan})
        return {"beta": np.full(p, np.nan), "se": np.full(p, np.nan), "hr": np.full(p, np.nan), "ci_low": np.full(p, np.nan), "ci_high": np.full(p, np.nan), "p_value": np.full(p, np.nan), "vcov": np.full((p, p), np.nan), "n": float(len(times)), "events": float(events.sum()), "summary": summary}
    result = minimize(fun=lambda b: cox_nll_grad_hess(b, x, times, events)[0], x0=np.zeros(x.shape[1]), jac=lambda b: cox_nll_grad_hess(b, x, times, events)[1], hess=lambda b: cox_nll_grad_hess(b, x, times, events)[2], method="trust-constr", options={"gtol": 1e-9, "xtol": 1e-9, "maxiter": 1000, "verbose": 0})
    beta = np.asarray(result.x, float); info = cox_nll_grad_hess(beta, x, times, events)[2]; vcov = np.linalg.inv(info); se = np.sqrt(np.diag(vcov)); z = beta / se; p = chi2.sf(z ** 2, 1)
    summary = pd.DataFrame({"feature": feature_names or list(range(len(beta))), "beta": beta, "hr": np.exp(beta), "ci_low": np.exp(beta - 1.96 * se), "ci_high": np.exp(beta + 1.96 * se), "p_value": p})
    return {"beta": beta, "se": se, "hr": np.exp(beta), "ci_low": np.exp(beta - 1.96 * se), "ci_high": np.exp(beta + 1.96 * se), "p_value": p, "vcov": vcov, "n": float(len(times)), "events": float(events.sum()), "summary": summary}

def logrank_test(times_a: np.ndarray, events_a: np.ndarray, times_b: np.ndarray, events_b: np.ndarray) -> dict[str, float]:
    times = np.concatenate([times_a, times_b]).astype(float); events = np.concatenate([events_a, events_b]).astype(int); groups = np.concatenate([np.zeros(len(times_a), int), np.ones(len(times_b), int)])
    obs = exp = var = 0.0
    for t in np.sort(np.unique(times[events == 1])):
        risk = times >= t; event = (times == t) & (events == 1); n = float(risk.sum()); d = float(event.sum()); n1 = float(np.sum(risk & (groups == 1))); d1 = float(np.sum(event & (groups == 1)))
        if n <= 1 or d == 0: continue
        exp_i = d * (n1 / n); var_i = d * (n1 / n) * (1.0 - n1 / n) * ((n - d) / (n - 1.0)); obs += d1; exp += exp_i; var += var_i
    z = (obs - exp) / math.sqrt(max(var, 1e-12)); return {"z": float(z), "p_value": float(chi2.sf(z ** 2, 1))}

def grouped_cox_summary(times: np.ndarray, events: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    times = np.asarray(times, float); events = np.asarray(events, int); groups = np.asarray(groups, int)
    if np.unique(groups).size != 2 or np.sum(events[groups == 0]) == 0 or np.sum(events[groups == 1]) == 0:
        return {"hr": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}
    def score(beta: float) -> float:
        out = 0.0
        for t in np.sort(np.unique(times[events == 1])):
            risk = times >= t; event = (times == t) & (events == 1); d = int(np.sum(event))
            if d == 0: continue
            w = np.exp(beta * groups[risk]); out += float(np.sum(groups[event])) - d * float(np.sum(groups[risk] * w) / np.sum(w))
        return out
    left, right = score(-12.0), score(12.0)
    beta = float(brentq(score, -12.0, 12.0, maxiter=500)) if np.isfinite(left) and np.isfinite(right) and left * right < 0 else float(minimize_scalar(lambda b: -sum(float(np.sum(groups[(times == t) & (events == 1)])) * b - int(np.sum((times == t) & (events == 1))) * math.log(float(np.sum(np.exp(b * groups[times >= t])))) for t in np.sort(np.unique(times[events == 1]))), bounds=(-12.0, 12.0), method="bounded").x)
    info = 0.0
    for t in np.sort(np.unique(times[events == 1])):
        risk = times >= t; event = (times == t) & (events == 1); d = int(np.sum(event))
        if d == 0: continue
        x = groups[risk]; w = np.exp(beta * x); mean_x = float(np.sum(x * w) / np.sum(w)); mean_x2 = float(np.sum((x ** 2) * w) / np.sum(w)); info += d * (mean_x2 - mean_x ** 2)
    if not np.isfinite(info) or info <= 0: return {"hr": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}
    se = math.sqrt(1.0 / info); z = beta / se
    return {"hr": float(np.exp(beta)), "ci_low": float(np.exp(beta - 1.96 * se)), "ci_high": float(np.exp(beta + 1.96 * se)), "p_value": float(chi2.sf(z ** 2, 1))}

def cutoff_survival_summary(feature: pd.Series, times: pd.Series, events: pd.Series, cutoff: float, high_inclusive: bool = True) -> dict[str, float]:
    frame = pd.DataFrame({"feature": feature, "time": times, "event": events}).dropna(); x = pd.to_numeric(frame["feature"], errors="coerce")
    high = (x >= cutoff) if high_inclusive else (x > cutoff)
    if int(high.sum()) == 0 or int((~high).sum()) == 0:
        return {"n": float(len(frame)), "cutoff": float(cutoff), "n_low": np.nan, "n_high": np.nan, "logrank_p": np.nan, "hr_high_vs_low": np.nan, "ci_low": np.nan, "ci_high": np.nan, "cox_p": np.nan}
    t = pd.to_numeric(frame["time"], errors="coerce").to_numpy(float); e = pd.to_numeric(frame["event"], errors="coerce").to_numpy(int); g = high.astype(int).to_numpy(int)
    lr = logrank_test(t[g == 0], e[g == 0], t[g == 1], e[g == 1]); cox = grouped_cox_summary(t, e, g)
    return {"n": float(len(frame)), "cutoff": float(cutoff), "n_low": float((g == 0).sum()), "n_high": float((g == 1).sum()), "logrank_p": float(lr["p_value"]), "hr_high_vs_low": float(cox["hr"]), "ci_low": float(cox["ci_low"]), "ci_high": float(cox["ci_high"]), "cox_p": float(cox["p_value"])}

def stage_adjusted_treatment_interaction(table: pd.DataFrame, time_col: str, event_col: str, treatment_col: str, high_score_col: str, stage_col: str) -> dict[str, float]:
    frame = table[[time_col, event_col, treatment_col, high_score_col, stage_col]].dropna().copy()
    tx = pd.to_numeric(frame[treatment_col], errors="coerce").to_numpy(float); high = pd.to_numeric(frame[high_score_col], errors="coerce").to_numpy(float); stage = pd.to_numeric(frame[stage_col], errors="coerce").to_numpy(float)
    fit = fit_cox_model(np.column_stack([tx, high, tx * high, stage]), pd.to_numeric(frame[time_col], errors="coerce").to_numpy(float), pd.to_numeric(frame[event_col], errors="coerce").to_numpy(int), feature_names=["treatment", "high", "interaction", "stage"])
    beta = np.asarray(fit["beta"], float); vcov = np.asarray(fit["vcov"], float)
    def contrast(w: np.ndarray) -> dict[str, float]:
        est = float(w @ beta); var = float(w @ vcov @ w)
        if not np.isfinite(var) or var <= 0: return {"hr": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}
        se = math.sqrt(var); z = est / se
        return {"hr": float(np.exp(est)), "ci_low": float(np.exp(est - 1.96 * se)), "ci_high": float(np.exp(est + 1.96 * se)), "p_value": float(chi2.sf(z ** 2, 1))}
    low, high_res = contrast(np.array([1.0, 0.0, 0.0, 0.0])), contrast(np.array([1.0, 0.0, 1.0, 0.0]))
    return {"n": float(fit["n"]), "events": float(fit["events"]), "low_hr": low["hr"], "low_ci_low": low["ci_low"], "low_ci_high": low["ci_high"], "low_p_value": low["p_value"], "high_hr": high_res["hr"], "high_ci_low": high_res["ci_low"], "high_ci_high": high_res["ci_high"], "high_p_value": high_res["p_value"], "interaction_hr_ratio": float(np.asarray(fit["hr"], float)[2]), "interaction_ci_low": float(np.asarray(fit["ci_low"], float)[2]), "interaction_ci_high": float(np.asarray(fit["ci_high"], float)[2]), "interaction_p_value": float(np.asarray(fit["p_value"], float)[2])}

def km_survival_at(times: np.ndarray, events: np.ndarray, horizon: float) -> float:
    order = np.argsort(times); times = np.asarray(times, float)[order]; events = np.asarray(events, int)[order]; surv = 1.0
    for t in np.sort(np.unique(times[(times <= horizon) & (events == 1)])):
        at_risk = int(np.sum(times >= t)); d = int(np.sum((times == t) & (events == 1))); surv *= 1.0 - d / at_risk
    return float(surv)

def compute_baseline_hazard(times: np.ndarray, events: np.ndarray, linear_predictor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    exp_lp = np.exp(linear_predictor); event_times = np.sort(np.unique(times[events == 1])); out = []; running = 0.0
    for t in event_times: running += float(np.sum((times == t) & (events == 1))) / float(np.sum(exp_lp[times >= t])); out.append(running)
    return event_times.astype(float), np.asarray(out, float)

def baseline_cumhaz_at(horizons: np.ndarray, event_times: np.ndarray, cumulative_hazard: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(event_times, horizons, side="right") - 1; out = np.zeros_like(horizons, float); keep = idx >= 0; out[keep] = cumulative_hazard[idx[keep]]; return out

def predict_event_probability(linear_predictor: np.ndarray, event_times: np.ndarray, cumulative_hazard: np.ndarray, horizon: float) -> np.ndarray:
    h0 = baseline_cumhaz_at(np.asarray([horizon], float), event_times, cumulative_hazard)[0]; return 1.0 - np.exp(-h0 * np.exp(linear_predictor))

def fit_censoring_km(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    censor = 1 - np.asarray(events, int); event_times = np.sort(np.unique(times[censor == 1])); out = []; surv = 1.0
    for t in event_times: surv *= 1.0 - np.sum((times == t) & (censor == 1)) / np.sum(times >= t); out.append(surv)
    return event_times.astype(float), np.asarray(out, float)

def censoring_survival_at(query_times: np.ndarray, censor_times: np.ndarray, censor_surv: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(censor_times, query_times, side="right") - 1; out = np.ones_like(query_times, float); keep = idx >= 0; out[keep] = censor_surv[idx[keep]]; return np.clip(out, 1e-6, None)

def harrell_c_index(times: np.ndarray, events: np.ndarray, scores: np.ndarray) -> float:
    concordant = comparable = 0.0
    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            if times[i] == times[j] and events[i] == events[j]: continue
            if events[i] == 1 and times[i] < times[j]: comparable += 1.0; concordant += float(scores[i] > scores[j]) + 0.5 * float(scores[i] == scores[j])
            elif events[j] == 1 and times[j] < times[i]: comparable += 1.0; concordant += float(scores[j] > scores[i]) + 0.5 * float(scores[i] == scores[j])
    return float(concordant / comparable) if comparable > 0 else np.nan

def cumulative_dynamic_auc(train_times: np.ndarray, train_events: np.ndarray, test_times: np.ndarray, test_events: np.ndarray, test_scores: np.ndarray, eval_times: np.ndarray) -> np.ndarray:
    censor_times, censor_surv = fit_censoring_km(train_times, train_events); auc = []
    for h in eval_times:
        cases = (test_times <= h) & (test_events == 1); controls = test_times > h
        if cases.sum() == 0 or controls.sum() == 0: auc.append(np.nan); continue
        control_scores = test_scores[controls]; case_weights = 1.0 / censoring_survival_at(test_times[cases], censor_times, censor_surv); score_sum = 0.0
        for s, w in zip(test_scores[cases], case_weights): score_sum += w * (np.sum(s > control_scores) + 0.5 * np.sum(s == control_scores))
        auc.append(score_sum / float(np.sum(case_weights) * len(control_scores)))
    return np.asarray(auc, float)

def calibration_table(times: np.ndarray, events: np.ndarray, scores: np.ndarray, baseline_event_times: np.ndarray, baseline_cumhaz: np.ndarray, horizons: np.ndarray, n_groups: int = 5) -> pd.DataFrame:
    rows = []
    for h in horizons:
        pred = np.clip(predict_event_probability(scores, baseline_event_times, baseline_cumhaz, float(h)), 1e-6, 1.0 - 1e-6); groups = np.asarray(pd.qcut(pred, q=n_groups, labels=False, duplicates="drop"), int)
        for gid in np.unique(groups):
            mask = groups == gid; obs = 1.0 - km_survival_at(times[mask], events[mask], float(h))
            rows.append({"horizon": float(h), "group": int(gid) + 1, "n": int(mask.sum()), "mean_predicted_event": float(np.mean(pred[mask])), "observed_event": float(obs)})
    return pd.DataFrame(rows)
