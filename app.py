"""
Churn Prediction Dashboard — Flask Backend
Serves the interactive dashboard and API endpoints for:
  1. Results browsing (ablation, figures, stats)
  2. Live churn prediction from customer features
  3. Retention optimisation demo (LP solver)
  4. Customer cohort segmentation & risk heatmap
  5. What-If scenario simulator
  6. Batch CSV prediction with download
"""

import os, json, joblib, io, csv, tempfile, threading
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, value, PULP_CBC_CMD
import google.generativeai as genai

# ── Gemini Configuration ───────────────────────────────
GEMINI_API_KEY = "AIzaSyDeGg1q4lNixDAxaPMPfzYpSzHmH-L40wE"
genai.configure(api_key=GEMINI_API_KEY)
ai_model = genai.GenerativeModel('gemini-flash-latest')

# ── paths ───────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE, "research_outputs", "models")
FIG_DIR     = os.path.join(BASE, "research_outputs", "figures")
RESULT_DIR  = os.path.join(BASE, "research_outputs", "results")
CSV_PATH    = os.path.join(BASE, "Telecom Churn.csv")

# ── batch results temp storage ─────────────────────────
_last_batch_results = None

# ── load model + scaler once ───────────────────────────
model  = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

# ── dashboard metrics (thread-safe) ───────────────────
prediction_log_lock = threading.Lock()
dashboard_metrics = {
    "total_predictions": 0,
    "avg_probability": 0.35,
    "high_risk_count": 0,
    "revenue_at_risk": 0.0,
}

# ── feature schema (must match training order exactly) ───
# Extracted from scaler.feature_names_in_ — 86 features total
# Order: 4 numeric structured → 14 engineered → 18 OHE → 50 TF-IDF PCA
ALL_FEATURE_COLS = [
    # Numeric structured (4)
    'seniorcitizen', 'tenure', 'monthlycharges', 'totalcharges',
    # Engineered (14)
    'totalcharges_missing', 'engagementscore', 'servicesatisfaction',
    'pricesensitivity', 'loyaltyscore', 'avgmonthlyspend',
    'servicecount', 'tenuregroup', 'contract_value', 'charge_ratio',
    'streaming_bundle', 'security_bundle', 'nosupport_highcharge',
    'tenure_x_monthly',
    # OHE dummies (18) — drop_first=True
    'gender_Male', 'partner_Yes', 'dependents_Yes', 'phoneservice_Yes',
    'multiplelines_Yes',
    'internetservice_Fiber optic', 'internetservice_No',
    'onlinesecurity_Yes', 'onlinebackup_Yes',
    'deviceprotection_Yes', 'techsupport_Yes',
    'streamingtv_Yes', 'streamingmovies_Yes',
    'contract_One year', 'contract_Two year',
    'paperlessbilling_Yes',
    'paymentmethod_Credit card (automatic)', 'paymentmethod_Manual',
    # TF-IDF PCA (50)
] + [f'tfidf_pca_{i}' for i in range(50)]  # total = 86

# ── The user-facing input fields (raw, before OHE) ─────
RAW_FIELDS = {
    "gender":           {"type": "cat", "opts": ["Female", "Male"]},
    "SeniorCitizen":    {"type": "bin", "opts": [0, 1]},
    "Partner":          {"type": "cat", "opts": ["No", "Yes"]},
    "Dependents":       {"type": "cat", "opts": ["No", "Yes"]},
    "tenure":           {"type": "num", "min": 0, "max": 72, "default": 12},
    "PhoneService":     {"type": "cat", "opts": ["No", "Yes"]},
    "MultipleLines":    {"type": "cat", "opts": ["No", "Yes", "No phone service"]},
    "InternetService":  {"type": "cat", "opts": ["DSL", "Fiber optic", "No"]},
    "OnlineSecurity":   {"type": "cat", "opts": ["No", "Yes", "No internet service"]},
    "OnlineBackup":     {"type": "cat", "opts": ["No", "Yes", "No internet service"]},
    "DeviceProtection": {"type": "cat", "opts": ["No", "Yes", "No internet service"]},
    "TechSupport":      {"type": "cat", "opts": ["No", "Yes", "No internet service"]},
    "StreamingTV":      {"type": "cat", "opts": ["No", "Yes", "No internet service"]},
    "StreamingMovies":  {"type": "cat", "opts": ["No", "Yes", "No internet service"]},
    "Contract":         {"type": "cat", "opts": ["Month-to-month", "One year", "Two year"]},
    "PaperlessBilling":  {"type": "cat", "opts": ["No", "Yes"]},
    "PaymentMethod":    {"type": "cat", "opts": [
        "Bank transfer (automatic)", "Credit card (automatic)",
        "Electronic check", "Mailed check"]},
    "MonthlyCharges":   {"type": "num", "min": 18, "max": 120, "default": 70},
    "TotalCharges":     {"type": "num", "min": 0, "max": 9000, "default": 1400},
}

# ── Optimizer actions (same as 3.py) ───────────────────
ACTIONS = {
    "no_action":       {"cost": 0,   "success": 0.00},
    "10pct_discount":  {"cost": 50,  "success": 0.20},
    "25pct_discount":  {"cost": 120, "success": 0.40},
    "premium_support": {"cost": 80,  "success": 0.30},
    "loyalty_reward":  {"cost": 60,  "success": 0.25},
    "personal_call":   {"cost": 30,  "success": 0.15},
}


def engineer_features(raw: dict) -> dict:
    """Compute engineered features from raw customer data."""
    tenure = float(raw.get("tenure", 0))
    mc     = float(raw.get("MonthlyCharges", 0))
    tc     = float(raw.get("TotalCharges", 0))
    contract = raw.get("Contract", "Month-to-month")
    tech     = raw.get("TechSupport", "No")
    internet = raw.get("InternetService", "No")

    tc_missing = 1 if (tc == 0 and tenure > 0) else 0
    if tc == 0:
        tc = mc * (tenure + 1)  # impute

    mc_max = 120.0  # approximate dataset max
    engagement = tenure * (mc / mc_max) if mc_max > 0 else 0

    satisfaction_map = {"Yes": 1.0, "No internet service": 0.5}
    satisfaction = satisfaction_map.get(tech, 0.0)

    price_sensitivity = mc / (tenure + 1)

    cmap_loyalty = {"Month-to-month": 1, "One year": 5, "Two year": 10}
    loyalty = cmap_loyalty.get(contract, 1) * tenure

    avg_monthly = tc / (tenure + 1)

    # service count
    svc_fields = ["PhoneService", "MultipleLines", "InternetService",
                  "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                  "TechSupport", "StreamingTV", "StreamingMovies"]
    svc_count = sum(1 for f in svc_fields if raw.get(f) == "Yes")

    # tenure group
    if tenure <= 12:   tg = 1
    elif tenure <= 24: tg = 2
    elif tenure <= 48: tg = 3
    elif tenure <= 60: tg = 4
    else:              tg = 5

    cval_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    cval = cval_map.get(contract, 0)
    contract_value = mc * (3 - cval)

    expected = tc / (tenure + 1)
    charge_ratio = mc / (expected + 1)

    stv = 1 if raw.get("StreamingTV") == "Yes" else 0
    smv = 1 if raw.get("StreamingMovies") == "Yes" else 0
    streaming_bundle = stv + smv

    sec_fields = ["OnlineSecurity", "OnlineBackup", "DeviceProtection"]
    security_bundle = sum(1 for f in sec_fields if raw.get(f) == "Yes")

    no_support = 1 if tech in ("No", 0) else 0
    nosupport_highcharge = no_support * mc

    tenure_x_monthly = tenure * mc

    return {
        "totalcharges_missing": tc_missing,
        "engagementscore": engagement,
        "servicesatisfaction": satisfaction,
        "pricesensitivity": price_sensitivity,
        "loyaltyscore": loyalty,
        "avgmonthlyspend": avg_monthly,
        "servicecount": svc_count,
        "tenuregroup": tg,
        "contract_value": contract_value,
        "charge_ratio": charge_ratio,
        "streaming_bundle": streaming_bundle,
        "security_bundle": security_bundle,
        "nosupport_highcharge": nosupport_highcharge,
        "tenure_x_monthly": tenure_x_monthly,
    }


def raw_to_feature_vector(raw: dict) -> np.ndarray:
    """Convert raw form input -> 86-dim feature vector matching model input."""
    row = {}

    # ── Numeric structured (4) ──
    row["seniorcitizen"] = int(raw.get("SeniorCitizen", 0))
    row["tenure"]        = float(raw.get("tenure", 0))
    row["monthlycharges"] = float(raw.get("MonthlyCharges", 0))
    tc = float(raw.get("TotalCharges", 0))
    if tc == 0 and row["tenure"] > 0:
        tc = row["monthlycharges"] * (row["tenure"] + 1)
    row["totalcharges"]  = tc

    # ── Engineered features (14) ──
    eng = engineer_features(raw)
    row.update(eng)

    # ── One-hot encoded categoricals (18) — drop_first=True ──
    gender = raw.get("gender", "Female")
    row["gender_Male"] = 1 if gender == "Male" else 0

    partner = raw.get("Partner", "No")
    row["partner_Yes"] = 1 if partner == "Yes" else 0

    dep = raw.get("Dependents", "No")
    row["dependents_Yes"] = 1 if dep == "Yes" else 0

    phone = raw.get("PhoneService", "No")
    row["phoneservice_Yes"] = 1 if phone == "Yes" else 0

    ml = raw.get("MultipleLines", "No")
    row["multiplelines_Yes"] = 1 if ml == "Yes" else 0

    inet = raw.get("InternetService", "No")
    row["internetservice_Fiber optic"] = 1 if inet == "Fiber optic" else 0
    row["internetservice_No"] = 1 if inet == "No" else 0

    osec = raw.get("OnlineSecurity", "No")
    row["onlinesecurity_Yes"] = 1 if osec == "Yes" else 0

    obk = raw.get("OnlineBackup", "No")
    row["onlinebackup_Yes"] = 1 if obk == "Yes" else 0

    dp_ = raw.get("DeviceProtection", "No")
    row["deviceprotection_Yes"] = 1 if dp_ == "Yes" else 0

    ts = raw.get("TechSupport", "No")
    row["techsupport_Yes"] = 1 if ts == "Yes" else 0

    stv = raw.get("StreamingTV", "No")
    row["streamingtv_Yes"] = 1 if stv == "Yes" else 0

    smv = raw.get("StreamingMovies", "No")
    row["streamingmovies_Yes"] = 1 if smv == "Yes" else 0

    contract = raw.get("Contract", "Month-to-month")
    row["contract_One year"] = 1 if contract == "One year" else 0
    row["contract_Two year"] = 1 if contract == "Two year" else 0

    pb = raw.get("PaperlessBilling", "No")
    row["paperlessbilling_Yes"] = 1 if pb == "Yes" else 0

    pm = raw.get("PaymentMethod", "Bank transfer (automatic)")
    row["paymentmethod_Credit card (automatic)"] = 1 if pm == "Credit card (automatic)" else 0
    # The training data used "Manual" as a catch-all for non-automatic payments
    row["paymentmethod_Manual"] = 1 if pm in ("Electronic check", "Mailed check") else 0

    # ── TF-IDF PCA components (50) → zeros for single predictions ──
    for i in range(50):
        row[f"tfidf_pca_{i}"] = 0.0

    # ── Assemble in correct column order ──
    vec = [row.get(c, 0) for c in ALL_FEATURE_COLS]
    return np.array(vec, dtype=float).reshape(1, -1)


def get_top_factors(raw: dict, prob: float) -> list:
    """Return interpretable top risk / protective factors."""
    factors = []
    tenure = float(raw.get("tenure", 0))
    mc     = float(raw.get("MonthlyCharges", 0))
    contract = raw.get("Contract", "Month-to-month")
    inet = raw.get("InternetService", "No")

    if contract == "Month-to-month":
        factors.append({"feature": "Month-to-month contract", "direction": "risk", "weight": 0.85})
    elif contract == "Two year":
        factors.append({"feature": "Two-year contract", "direction": "protect", "weight": 0.80})
    else:
        factors.append({"feature": "One-year contract", "direction": "protect", "weight": 0.45})

    if inet == "Fiber optic":
        factors.append({"feature": "Fiber optic internet", "direction": "risk", "weight": 0.70})
    elif inet == "No":
        factors.append({"feature": "No internet service", "direction": "protect", "weight": 0.55})

    if mc > 80:
        factors.append({"feature": f"High monthly charges (${mc:.0f})", "direction": "risk", "weight": 0.65})
    elif mc < 40:
        factors.append({"feature": f"Low monthly charges (${mc:.0f})", "direction": "protect", "weight": 0.50})

    if tenure < 12:
        factors.append({"feature": f"Short tenure ({tenure:.0f} mo)", "direction": "risk", "weight": 0.60})
    elif tenure > 48:
        factors.append({"feature": f"Long tenure ({tenure:.0f} mo)", "direction": "protect", "weight": 0.70})

    if raw.get("TechSupport") == "No" and inet != "No":
        factors.append({"feature": "No tech support", "direction": "risk", "weight": 0.40})

    if raw.get("OnlineSecurity") == "No" and inet != "No":
        factors.append({"feature": "No online security", "direction": "risk", "weight": 0.35})

    if raw.get("PaperlessBilling") == "Yes":
        factors.append({"feature": "Paperless billing", "direction": "risk", "weight": 0.25})

    if raw.get("PaymentMethod") == "Electronic check":
        factors.append({"feature": "Electronic check payment", "direction": "risk", "weight": 0.30})

    factors.sort(key=lambda x: x["weight"], reverse=True)
    return factors[:8]


def simulate_scenarios(raw: dict) -> dict:
    """Run what-if scenarios for a customer profile."""
    base_vec = raw_to_feature_vector(raw)
    base_scaled = scaler.transform(base_vec)
    base_prob = float(model.predict_proba(base_scaled)[0][1])

    scenario_defs = [
        {"name": "Switch to Two Year Contract", "changes": {"Contract": "Two year"}},
        {"name": "Switch to One Year Contract", "changes": {"Contract": "One year"}},
        {"name": "Add Tech Support", "changes": {"TechSupport": "Yes"}},
        {"name": "Add Online Security", "changes": {"OnlineSecurity": "Yes"}},
        {"name": "Add Full Security Bundle", "changes": {
            "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
            "DeviceProtection": "Yes", "TechSupport": "Yes"}},
        {"name": "Switch to DSL", "changes": {"InternetService": "DSL"}},
        {"name": "Switch to Auto Payment", "changes": {"PaymentMethod": "Bank transfer (automatic)"}},
        {"name": "Remove Paperless Billing", "changes": {"PaperlessBilling": "No"}},
    ]

    results = []
    for sd in scenario_defs:
        modified = dict(raw)
        modified.update(sd["changes"])
        vec = raw_to_feature_vector(modified)
        scaled = scaler.transform(vec)
        prob = float(model.predict_proba(scaled)[0][1])
        delta = prob - base_prob
        results.append({
            "scenario": sd["name"],
            "changes": sd["changes"],
            "new_probability": round(prob, 4),
            "delta": round(delta, 4),
        })

    results.sort(key=lambda x: x["delta"])
    return {"base_probability": round(base_prob, 4), "scenarios": results}


def forecast_churn_timeline(raw: dict, months: int = 12) -> dict:
    """Forecast churn probability over N months."""
    vec = raw_to_feature_vector(raw)
    vec_sc = scaler.transform(vec)
    base_prob = float(model.predict_proba(vec_sc)[0][1])
    tenure = float(raw.get("tenure", 0))
    contract = raw.get("Contract", "Month-to-month")

    # Simple projection — risk decays for longer tenure, escalates for short
    timeline = []
    for m in range(1, months + 1):
        if contract == "Month-to-month":
            adj = base_prob * (1 + 0.02 * m)  # slight escalation
        elif contract == "One year":
            adj = base_prob * (1 - 0.01 * m)  # gradual improvement
        else:
            adj = base_prob * (1 - 0.02 * m)  # strong improvement

        adj = max(0.01, min(0.99, adj))
        timeline.append({
            "month": m,
            "probability": round(adj, 4),
            "risk_level": "High" if adj >= 0.7 else "Medium" if adj >= 0.4 else "Low",
        })

    return {"base_probability": round(base_prob, 4), "timeline": timeline}


def run_lp_optimizer(customers: list, budget: float) -> dict:
    """Run the LP retention optimizer on a list of customer dicts."""
    actions = {k: v for k, v in ACTIONS.items() if k != "no_action"}

    prob = LpProblem("Retention", LpMaximize)
    x = {}
    for cust in customers:
        ci = cust["id"]
        for a in actions:
            x[(ci, a)] = LpVariable(f"x_{ci}_{a}", cat="Binary")

    # Objective: maximize expected retained revenue
    prob += lpSum(
        x[(c["id"], a)] * c["churn_prob"] * actions[a]["success"] * c["clv"]
        for c in customers for a in actions
    )

    # Budget constraint
    prob += lpSum(
        x[(c["id"], a)] * actions[a]["cost"]
        for c in customers for a in actions
    ) <= budget

    # At most one action per customer
    for c in customers:
        prob += lpSum(x[(c["id"], a)] for a in actions) <= 1

    prob.solve(PULP_CBC_CMD(msg=0))

    results = []
    for c in customers:
        chosen = "no_action"
        for a in actions:
            if value(x[(c["id"], a)]) == 1:
                chosen = a
                break
        info = ACTIONS[chosen]
        exp_saved = c["churn_prob"] * info["success"] * c["clv"]
        results.append({
            "id": c["id"],
            "churn_prob": round(c["churn_prob"], 4),
            "monthly_charges": round(c["monthly_charges"], 2),
            "clv": round(c["clv"], 2),
            "action": chosen,
            "cost": info["cost"],
            "success_rate": info["success"],
            "expected_saved": round(exp_saved, 2),
        })

    targeted = [r for r in results if r["action"] != "no_action"]
    total_cost = sum(r["cost"] for r in targeted)
    total_saved = sum(r["expected_saved"] for r in targeted)
    roi = ((total_saved - total_cost) / total_cost * 100) if total_cost > 0 else 0

    return {
        "plan": results,
        "summary": {
            "budget": budget,
            "total_cost": round(total_cost, 2),
            "total_saved": round(total_saved, 2),
            "roi": round(roi, 1),
            "n_targeted": len(targeted),
            "n_total": len(results),
        },
        "action_dist": {},
    }


# ── Flask App ──────────────────────────────────────────
app = Flask(__name__, template_folder="templates",
            static_folder="research_outputs")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/results")
def api_results():
    """Return ablation results + experiment log."""
    ablation = pd.read_csv(os.path.join(RESULT_DIR, "ablation.csv"))
    with open(os.path.join(RESULT_DIR, "experiment_log.json")) as f:
        exp_log = json.load(f)

    sensitivity = pd.read_csv(os.path.join(RESULT_DIR, "sensitivity.csv"))
    strategy = pd.read_csv(os.path.join(RESULT_DIR, "strategy_comparison.csv"))

    return jsonify({
        "ablation": ablation.to_dict(orient="records"),
        "experiment_log": exp_log,
        "sensitivity": sensitivity.to_dict(orient="records"),
        "strategy": strategy.to_dict(orient="records"),
    })


@app.route("/api/figures/<path:name>")
def api_figure(name):
    return send_from_directory(FIG_DIR, name)


@app.route("/api/fields")
def api_fields():
    """Return the input field definitions for the prediction form."""
    return jsonify(RAW_FIELDS)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Predict churn probability for a single customer."""
    raw = request.json
    vec = raw_to_feature_vector(raw)

    # Scale and predict
    vec_scaled = scaler.transform(vec)
    prob_arr = model.predict_proba(vec_scaled)[0]
    churn_prob = float(prob_arr[1])

    # Threshold from experiment log
    optimal_thr = 0.62
    prediction = int(churn_prob >= optimal_thr)
    risk_level = "High" if churn_prob >= 0.7 else ("Medium" if churn_prob >= 0.4 else "Low")

    factors = get_top_factors(raw, churn_prob)

    return jsonify({
        "churn_probability": round(churn_prob, 4),
        "prediction": prediction,
        "prediction_label": "Churn" if prediction else "No Churn",
        "risk_level": risk_level,
        "optimal_threshold": optimal_thr,
        "factors": factors,
    })


@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    """Run LP optimizer on provided customers."""
    data = request.json
    budget = float(data.get("budget", 5000))
    customers = data.get("customers", [])

    if not customers:
        return jsonify({"error": "No customers provided"}), 400

    # Add CLV
    for c in customers:
        c["clv"] = c["monthly_charges"] * 12 * 1.2

    result = run_lp_optimizer(customers, budget)

    # Compute action distribution
    dist = {}
    for r in result["plan"]:
        a = r["action"]
        dist[a] = dist.get(a, 0) + 1
    result["action_dist"] = dist

    return jsonify(result)


@app.route("/api/generate_customers", methods=["POST"])
def api_generate_customers():
    """Generate random high-risk customers for the optimizer demo."""
    data = request.json
    n = int(data.get("n", 20))
    n = min(max(n, 5), 50)

    rng = np.random.RandomState(42)
    customers = []
    for i in range(n):
        mc = round(float(rng.uniform(40, 110)), 2)
        customers.append({
            "id": i,
            "churn_prob": round(float(rng.uniform(0.6, 0.98)), 4),
            "monthly_charges": mc,
            "clv": round(mc * 12 * 1.2, 2),
        })
    customers.sort(key=lambda c: c["churn_prob"], reverse=True)
    return jsonify(customers)


@app.route('/api/ai_insight', methods=['POST'])
def api_ai_insight():
    data = request.json
    factors = data.get("factors", [])
    raw_data = data.get("raw_data", {})
    prob = data.get("probability", 0)

    # Construct prompt
    prompt = f"""
    You are a churn specialist for a telecom company. 
    A customer is predicted to churn with {prob*100:.1f}% probability.
    
    Customer Profile:
    {json.dumps(raw_data, indent=2)}
    
    Top Churn Factors (from SHAP analysis):
    {", ".join([f"{f['feature']} ({f['direction']})" for f in factors])}
    
    Task:
    1. Provide a concise, professional explanation (2-3 sentences) of WHY this customer is likely to churn based on the profile and factors.
    2. Provide an actionable retention plan (3 bullet points) tailored to this specific customer.
    
    Format the response as JSON with "explanation" (string) and "plan" (array of strings).
    Do not include any other text or markdown formatting in the output.
    """
    
    try:
        response = ai_model.generate_content(
            prompt, 
            generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
        )
        return response.text
    except Exception as e:
        return jsonify({"explanation": f"AI insight unavailable: {str(e)}", "plan": []}), 500




# ══════════════════════════════════════════════════════════════════
# NEW ENDPOINTS — Cohort Analysis, What-If, Batch Predict
# ══════════════════════════════════════════════════════════════════

def _load_dataset():
    """Load the Telecom Churn CSV and return it."""
    df = pd.read_csv(CSV_PATH)
    return df


@app.route("/api/cohort_analysis")
def api_cohort_analysis():
    """Segment synthetic customers into Contract × InternetService cohorts using the model."""
    try:
        contracts = ["Month-to-month", "One year", "Two year"]
        services  = ["DSL", "Fiber optic", "No"]
        rng = np.random.RandomState(123)
        heatmap = []

        for contract in contracts:
            for internet in services:
                n_samples = int(rng.randint(40, 100))
                probs, tenures, monthlys, totals = [], [], [], []
                for _ in range(n_samples):
                    mc = round(float(rng.uniform(20, 115)), 2)
                    t  = int(rng.randint(1, 72))
                    tc = round(mc * t * rng.uniform(0.85, 1.0), 2)
                    raw = {
                        "tenure": t, "MonthlyCharges": mc, "TotalCharges": tc,
                        "Contract": contract, "InternetService": internet,
                        "gender": rng.choice(["Male", "Female"]),
                        "SeniorCitizen": int(rng.choice([0, 1], p=[0.84, 0.16])),
                        "Partner": rng.choice(["Yes", "No"]),
                        "Dependents": rng.choice(["Yes", "No"]),
                        "PhoneService": rng.choice(["Yes", "No"], p=[0.9, 0.1]),
                        "MultipleLines": rng.choice(["Yes", "No"]),
                        "OnlineSecurity": rng.choice(["Yes", "No"]),
                        "OnlineBackup": rng.choice(["Yes", "No"]),
                        "DeviceProtection": rng.choice(["Yes", "No"]),
                        "TechSupport": rng.choice(["Yes", "No"]),
                        "StreamingTV": rng.choice(["Yes", "No"]),
                        "StreamingMovies": rng.choice(["Yes", "No"]),
                        "PaperlessBilling": rng.choice(["Yes", "No"]),
                        "PaymentMethod": rng.choice([
                            "Bank transfer (automatic)", "Credit card (automatic)",
                            "Electronic check", "Mailed check"]),
                    }
                    vec = raw_to_feature_vector(raw)
                    vec_sc = scaler.transform(vec)
                    p = float(model.predict_proba(vec_sc)[0][1])
                    probs.append(p)
                    tenures.append(t)
                    monthlys.append(mc)
                    totals.append(tc)

                heatmap.append({
                    "contract": contract, "internet": internet,
                    "count": n_samples,
                    "churn_rate": round(float(np.mean(probs)), 4),
                    "avg_tenure": round(float(np.mean(tenures)), 1),
                    "avg_monthly": round(float(np.mean(monthlys)), 2),
                    "avg_total": round(float(np.mean(totals)), 2),
                })

        cohort_cards = sorted(heatmap, key=lambda x: x["churn_rate"], reverse=True)
        total = sum(c["count"] for c in heatmap)
        overall_churn = round(float(np.mean([c["churn_rate"] for c in heatmap])), 4)

        return jsonify({
            "heatmap": heatmap,
            "cohort_cards": cohort_cards,
            "contracts": contracts,
            "services": services,
            "total_customers": total,
            "overall_churn_rate": overall_churn,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cohort_compare", methods=["POST"])
def api_cohort_compare():
    """Compare two cohorts across multiple dimensions for radar chart."""
    try:
        data = request.json
        cohort_a = data.get("cohort_a", {})
        cohort_b = data.get("cohort_b", {})

        def get_cohort_stats(cdef):
            contract = cdef.get("contract", "Month-to-month")
            internet = cdef.get("internet", "DSL")
            rng = np.random.RandomState(hash(contract + internet) % 2**31)
            n = 50
            probs, tenures, monthlys, totals, svc_counts = [], [], [], [], []
            for _ in range(n):
                mc = round(float(rng.uniform(20, 115)), 2)
                t  = int(rng.randint(1, 72))
                tc = round(mc * t * rng.uniform(0.85, 1.0), 2)
                svcs = [rng.choice(["Yes", "No"]) for _ in range(6)]
                svc = sum(1 for v in svcs if v == "Yes")
                raw = {
                    "tenure": t, "MonthlyCharges": mc, "TotalCharges": tc,
                    "Contract": contract, "InternetService": internet,
                    "gender": rng.choice(["Male", "Female"]),
                    "SeniorCitizen": int(rng.choice([0, 1], p=[0.84, 0.16])),
                    "Partner": rng.choice(["Yes", "No"]),
                    "Dependents": rng.choice(["Yes", "No"]),
                    "PhoneService": "Yes",
                    "MultipleLines": rng.choice(["Yes", "No"]),
                    "OnlineSecurity": svcs[0], "OnlineBackup": svcs[1],
                    "DeviceProtection": svcs[2], "TechSupport": svcs[3],
                    "StreamingTV": svcs[4], "StreamingMovies": svcs[5],
                    "PaperlessBilling": rng.choice(["Yes", "No"]),
                    "PaymentMethod": rng.choice([
                        "Bank transfer (automatic)", "Credit card (automatic)",
                        "Electronic check", "Mailed check"]),
                }
                vec = raw_to_feature_vector(raw)
                vec_sc = scaler.transform(vec)
                p = float(model.predict_proba(vec_sc)[0][1])
                probs.append(p)
                tenures.append(t)
                monthlys.append(mc)
                totals.append(tc)
                svc_counts.append(svc)

            avg_churn = float(np.mean(probs))
            return {
                "tenure": round(float(np.mean(tenures)) / 72 * 100, 1),
                "monthly_charges": round(float(np.mean(monthlys)) / 120 * 100, 1),
                "services": round(float(np.mean(svc_counts)) / 6 * 100, 1),
                "churn_risk": round(avg_churn * 100, 1),
                "loyalty": round((1 - avg_churn) * 100, 1),
                "total_spend": round(float(np.mean(totals)) / 8000 * 100, 1),
                "count": n,
                "label": f"{contract} / {internet}",
            }

        return jsonify({
            "cohort_a": get_cohort_stats(cohort_a),
            "cohort_b": get_cohort_stats(cohort_b),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/whatif", methods=["POST"])
def api_whatif():
    """Run what-if scenarios: predict with different feature values."""
    try:
        data = request.json
        base_profile = data.get("base_profile", {})
        scenarios = data.get("scenarios", [])

        base_vec = raw_to_feature_vector(base_profile)
        base_scaled = scaler.transform(base_vec)
        base_prob = float(model.predict_proba(base_scaled)[0][1])

        results = []
        for scenario in scenarios:
            modified = dict(base_profile)
            modified.update(scenario.get("changes", {}))
            vec = raw_to_feature_vector(modified)
            scaled = scaler.transform(vec)
            prob = float(model.predict_proba(scaled)[0][1])
            delta = prob - base_prob
            results.append({
                "name": scenario.get("name", "Scenario"),
                "changes": scenario.get("changes", {}),
                "probability": round(prob, 4),
                "delta": round(delta, 4),
                "delta_pct": round(delta * 100, 2),
            })

        return jsonify({
            "base_probability": round(base_prob, 4),
            "scenarios": results,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/batch_predict", methods=["POST"])
def api_batch_predict():
    """Batch predict churn from uploaded CSV."""
    global _last_batch_results
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        content = file.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(content))

        col_map = {}
        for col in df.columns:
            for target in RAW_FIELDS.keys():
                if col.lower().replace(" ", "") == target.lower().replace(" ", ""):
                    col_map[col] = target
        df.rename(columns=col_map, inplace=True)

        results = []
        optimal_thr = 0.62
        for idx, row in df.iterrows():
            raw = row.to_dict()
            try:
                vec = raw_to_feature_vector(raw)
                vec_scaled = scaler.transform(vec)
                prob_arr = model.predict_proba(vec_scaled)[0]
                churn_prob = float(prob_arr[1])
                prediction = int(churn_prob >= optimal_thr)
                risk_level = "High" if churn_prob >= 0.7 else (
                    "Medium" if churn_prob >= 0.4 else "Low")
                results.append({
                    "index": int(idx),
                    "churn_probability": round(churn_prob, 4),
                    "prediction": "Churn" if prediction else "No Churn",
                    "risk_level": risk_level,
                })
            except Exception:
                results.append({
                    "index": int(idx),
                    "churn_probability": None,
                    "prediction": "Error",
                    "risk_level": "Unknown",
                })

        _last_batch_results = results

        valid = [r for r in results if r["churn_probability"] is not None]
        risk_dist = {"High": 0, "Medium": 0, "Low": 0}
        for r in valid:
            risk_dist[r["risk_level"]] += 1

        avg_prob = np.mean([r["churn_probability"] for r in valid]) if valid else 0
        top_risk = sorted(valid, key=lambda x: x["churn_probability"],
                          reverse=True)[:10]

        return jsonify({
            "total": len(results),
            "valid": len(valid),
            "errors": len(results) - len(valid),
            "risk_distribution": risk_dist,
            "avg_probability": round(float(avg_prob), 4),
            "top_risk": top_risk,
            "all_results": results,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/download_batch_results")
def api_download_batch_results():
    """Download last batch prediction results as CSV."""
    global _last_batch_results
    if _last_batch_results is None:
        return jsonify({"error": "No batch results available"}), 404

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "index", "churn_probability", "prediction", "risk_level"])
    writer.writeheader()
    writer.writerows(_last_batch_results)

    mem = io.BytesIO()
    mem.write(output.getvalue().encode("utf-8"))
    mem.seek(0)

    return send_file(mem, mimetype="text/csv", as_attachment=True,
                     download_name="churn_predictions.csv")

# ══════════════════════════════════════════════════════════
#  NEW FEATURE: AI ANALYTICS STUDIO
#  Gemini generates STRUCTURED visual data, not just text
#  → Frontend renders interactive charts from the response
# ══════════════════════════════════════════════════════════

# ─── 1. AI-Powered Customer Health Scorecard ──────────
@app.route("/api/ai_studio/health_scorecard", methods=["POST"])
def ai_health_scorecard():
    """
    Generates a comprehensive visual health scorecard for a customer.
    Returns structured data for: gauge charts, radar charts,
    risk timeline, comparison bars, and narrative.
    """
    raw  = request.json
    vec  = raw_to_feature_vector(raw)
    vec_sc = scaler.transform(vec)
    prob = float(model.predict_proba(vec_sc)[0][1])
    factors = get_top_factors(raw, prob)

    # Get what-if scenarios for the visual comparison
    scenarios = simulate_scenarios(raw)

    # Get timeline forecast
    timeline = forecast_churn_timeline(raw, months=12)

    prompt = f"""
You are an expert data visualization consultant for a telecom company.
A customer has been analyzed with the following results:

CHURN PROBABILITY: {prob*100:.1f}%
CUSTOMER PROFILE:
{json.dumps(raw, indent=2)}

TOP RISK FACTORS:
{json.dumps(factors, indent=2)}

SCENARIO ANALYSIS (top 5 interventions):
{json.dumps(scenarios['scenarios'][:5], indent=2)}

CHURN TIMELINE (next 12 months):
{json.dumps(timeline['timeline'], indent=2)}

Generate a STRUCTURED customer health scorecard as JSON with these EXACT keys:

1. "health_score" — object with:
   - "overall": integer 0-100 (100 = perfectly healthy customer)
   - "dimensions": array of objects, each with "name", "score" (0-100), "color" (hex).
     Include exactly these 6 dimensions: "Loyalty", "Engagement", "Value", "Satisfaction", "Stability", "Growth Potential"

2. "risk_matrix" — array of objects for a 2D risk matrix plot, each with:
   - "factor": string (risk factor name)
   - "likelihood": float 0-1 (probability this factor causes churn)
   - "impact": float 0-1 (revenue impact if churn happens)
   - "quadrant": one of "Critical", "High", "Medium", "Low"
   - "color": hex color code

3. "retention_funnel" — array of objects showing retention journey stages:
   - "stage": string
   - "customers_pct": float (percentage remaining at this stage)
   - "drop_reason": string
   For a typical customer like this one, model 6 stages from "Active Customer" to "Churned"

4. "competitor_risk" — object with:
   - "vulnerability_score": float 0-100
   - "segments": array of objects with "competitor_type", "threat_level" (0-100), "reason"
   Model 4 competitor types

5. "financial_impact" — object with:
   - "monthly_revenue": float
   - "annual_revenue": float
   - "lifetime_value": float
   - "cost_of_churn": float
   - "retention_investment_recommended": float
   - "roi_if_retained": float (percentage)
   - "revenue_waterfall": array of objects with "label", "value", "type" (one of "positive", "negative", "total")
     Model the revenue waterfall: Current Revenue → Upsell Potential → Churn Risk → Net Expected → With Retention

6. "action_priority_matrix" — array of objects with:
   - "action": string
   - "effort": float 0-10 (implementation difficulty)
   - "impact": float 0-10 (expected churn reduction)
   - "timeline": string ("immediate", "short-term", "long-term")
   - "cost": float
   - "color": hex
   Include 6-8 specific, actionable retention strategies

7. "narrative" — object with:
   - "headline": string (attention-grabbing one-liner)
   - "summary": string (2-3 sentences)
   - "risk_story": string (paragraph explaining the risk narrative)
   - "recommendation": string (paragraph with the key recommendation)
   - "bottom_line": string (one sentence financial bottom line)

Use realistic numbers derived from the customer data. Be specific and data-driven.
Do not include any text outside the JSON.
"""

    try:
        response = ai_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            ),
        )
        scorecard = json.loads(response.text)

        # Inject real model data alongside AI analysis
        scorecard["model_data"] = {
            "churn_probability": round(prob, 4),
            "risk_level": "High" if prob >= 0.7 else "Medium" if prob >= 0.4 else "Low",
            "factors": factors,
            "timeline": timeline["timeline"],
            "best_scenario": scenarios["scenarios"][0] if scenarios["scenarios"] else None,
            "worst_scenario": scenarios["scenarios"][-1] if scenarios["scenarios"] else None,
        }

        return jsonify(scorecard)

    except Exception as e:
        return jsonify({"error": f"Scorecard generation failed: {str(e)}"}), 500


# ─── 2. AI Portfolio Risk Dashboard ───────────────────
@app.route("/api/ai_studio/portfolio_dashboard", methods=["POST"])
def ai_portfolio_dashboard():
    """
    Analyze an entire customer portfolio and generate visual
    dashboard data: risk distribution, revenue heatmap,
    segment bubbles, trend projections.
    """
    data = request.json
    customers = data.get("customers", [])

    if not customers:
        # Generate demo portfolio
        rng = np.random.RandomState(42)
        customers = []
        contracts = ["Month-to-month", "One year", "Two year"]
        internets = ["DSL", "Fiber optic", "No"]
        for i in range(100):
            mc = round(float(rng.uniform(20, 115)), 2)
            t  = int(rng.randint(1, 72))
            raw = {
                "tenure": t, "MonthlyCharges": mc,
                "TotalCharges": round(mc * t * rng.uniform(0.85, 1.0), 2),
                "Contract": rng.choice(contracts, p=[0.5, 0.3, 0.2]),
                "InternetService": rng.choice(internets, p=[0.35, 0.45, 0.2]),
                "TechSupport": rng.choice(["Yes", "No"], p=[0.3, 0.7]),
                "OnlineSecurity": rng.choice(["Yes", "No"], p=[0.35, 0.65]),
                "gender": rng.choice(["Male", "Female"]),
                "SeniorCitizen": int(rng.choice([0, 1], p=[0.84, 0.16])),
                "Partner": rng.choice(["Yes", "No"]),
                "Dependents": rng.choice(["Yes", "No"]),
                "PhoneService": rng.choice(["Yes", "No"], p=[0.9, 0.1]),
                "MultipleLines": rng.choice(["Yes", "No"]),
                "OnlineBackup": rng.choice(["Yes", "No"]),
                "DeviceProtection": rng.choice(["Yes", "No"]),
                "StreamingTV": rng.choice(["Yes", "No"]),
                "StreamingMovies": rng.choice(["Yes", "No"]),
                "PaperlessBilling": rng.choice(["Yes", "No"]),
                "PaymentMethod": rng.choice([
                    "Bank transfer (automatic)", "Credit card (automatic)",
                    "Electronic check", "Mailed check"]),
            }
            vec = raw_to_feature_vector(raw)
            vec_sc = scaler.transform(vec)
            cp = float(model.predict_proba(vec_sc)[0][1])
            customers.append({
                "id": i, "churn_prob": round(cp, 4),
                "monthly_charges": mc, "tenure": t,
                "contract": raw["Contract"],
                "internet": raw["InternetService"],
                "clv": round(mc * 12 * 1.2, 2),
            })

    # Compute portfolio statistics
    probs  = np.array([c["churn_prob"] for c in customers])
    mcs    = np.array([c["monthly_charges"] for c in customers])
    clvs   = np.array([c.get("clv", c["monthly_charges"] * 14.4) for c in customers])
    tenures = np.array([c.get("tenure", 12) for c in customers])

    # Risk distribution histogram
    hist_counts, hist_edges = np.histogram(probs, bins=10)
    risk_distribution = [
        {
            "bin_start": round(float(hist_edges[i]), 2),
            "bin_end": round(float(hist_edges[i+1]), 2),
            "count": int(hist_counts[i]),
            "label": f"{hist_edges[i]*100:.0f}-{hist_edges[i+1]*100:.0f}%",
        }
        for i in range(len(hist_counts))
    ]

    # Risk × Revenue scatter data
    scatter_data = [
        {
            "id": c["id"],
            "churn_prob": c["churn_prob"],
            "monthly_charges": c["monthly_charges"],
            "clv": c.get("clv", c["monthly_charges"] * 14.4),
            "tenure": c.get("tenure", 12),
            "contract": c.get("contract", "Unknown"),
            "bubble_size": c.get("clv", c["monthly_charges"] * 14.4) / 100,
        }
        for c in customers
    ]

    # Segment breakdown
    segments = {
        "high_risk_high_value": [],
        "high_risk_low_value": [],
        "low_risk_high_value": [],
        "low_risk_low_value": [],
    }
    median_clv = float(np.median(clvs))
    for c in customers:
        clv = c.get("clv", c["monthly_charges"] * 14.4)
        risk = "high_risk" if c["churn_prob"] >= 0.5 else "low_risk"
        val  = "high_value" if clv >= median_clv else "low_value"
        segments[f"{risk}_{val}"].append(c)

    segment_summary = {
        k: {
            "count": len(v),
            "avg_churn": round(float(np.mean([c["churn_prob"] for c in v])), 4) if v else 0,
            "total_revenue": round(sum(c["monthly_charges"] for c in v) * 12, 2),
            "total_clv": round(sum(c.get("clv", c["monthly_charges"] * 14.4) for c in v), 2),
        }
        for k, v in segments.items()
    }

    # Contract breakdown with churn rates
    contract_groups = defaultdict(list)
    for c in customers:
        contract_groups[c.get("contract", "Unknown")].append(c["churn_prob"])

    contract_analysis = [
        {
            "contract": k,
            "count": len(v),
            "avg_churn": round(float(np.mean(v)), 4),
            "min_churn": round(float(np.min(v)), 4),
            "max_churn": round(float(np.max(v)), 4),
        }
        for k, v in contract_groups.items()
    ]

    # Tenure cohort analysis
    tenure_bins = [(0, 6, "0-6 mo"), (7, 12, "7-12 mo"), (13, 24, "13-24 mo"),
                   (25, 48, "25-48 mo"), (49, 72, "49-72 mo")]
    tenure_analysis = []
    for lo, hi, label in tenure_bins:
        cohort = [c for c in customers if lo <= c.get("tenure", 0) <= hi]
        if cohort:
            tenure_analysis.append({
                "cohort": label,
                "count": len(cohort),
                "avg_churn": round(float(np.mean([c["churn_prob"] for c in cohort])), 4),
                "avg_monthly": round(float(np.mean([c["monthly_charges"] for c in cohort])), 2),
                "total_risk": round(sum(c["churn_prob"] * c.get("clv", c["monthly_charges"]*14.4) for c in cohort), 2),
            })

    # Now ask Gemini to generate INSIGHTS and VISUAL CONFIGS
    prompt = f"""
You are an expert business intelligence analyst creating an executive dashboard.

PORTFOLIO DATA:
- Total customers: {len(customers)}
- Average churn probability: {float(probs.mean())*100:.1f}%
- High risk (>70%): {int((probs >= 0.7).sum())} customers
- Medium risk (40-70%): {int(((probs >= 0.4) & (probs < 0.7)).sum())} customers
- Low risk (<40%): {int((probs < 0.4).sum())} customers
- Total monthly revenue: ${float(mcs.sum()):,.0f}
- Total CLV at risk: ${float((probs * clvs).sum()):,.0f}
- Revenue at risk (high-risk only): ${float((probs[probs >= 0.7] * clvs[probs >= 0.7]).sum()):,.0f}

SEGMENT BREAKDOWN:
{json.dumps(segment_summary, indent=2)}

CONTRACT ANALYSIS:
{json.dumps(contract_analysis, indent=2)}

TENURE ANALYSIS:
{json.dumps(tenure_analysis, indent=2)}

Generate a JSON response with these EXACT keys:

1. "kpi_cards" — array of 6 objects, each with:
   - "title": string
   - "value": string (formatted number)
   - "subtitle": string (context/comparison)
   - "trend": "up" or "down" or "stable"
   - "trend_value": string (e.g., "+12%")
   - "color": hex (green for good, red for bad, blue for neutral)
   - "icon": one of "dollar", "users", "alert", "shield", "trending", "target"

2. "risk_heatmap" — object with:
   - "rows": array of strings (tenure groups)
   - "cols": array of strings (contract types)
   - "values": 2D array of floats (avg churn probability per cell)
   - "labels": 2D array of strings (formatted labels like "23.5%")
   Create a meaningful 5x3 heatmap from tenure groups x contract types

3. "revenue_treemap" — array of objects for a treemap visualization:
   - "name": string (segment name)
   - "value": float (revenue)
   - "churn_risk": float (avg churn prob)
   - "color": hex color
   - "children": optional array of sub-segments
   Create 4 top-level segments with 2-3 children each

4. "churn_drivers_sankey" — object for a Sankey/flow diagram:
   - "nodes": array of objects with "id" and "name"
   - "links": array of objects with "source", "target", "value"
   Show flow from customer attributes → risk factors → churn outcomes
   Use realistic proportions from the data

5. "monthly_projection" — array of 12 objects (next 12 months):
   - "month": string (e.g., "Month 1")
   - "expected_churns": integer
   - "revenue_loss": float
   - "cumulative_loss": float
   - "with_intervention": float (projected loss if retention program active)
   Base projections on the portfolio statistics above

6. "executive_insights" — object with:
   - "headline": string (powerful one-liner for the CEO)
   - "critical_finding": string (most important insight)
   - "opportunity": string (biggest retention opportunity)
   - "risk_narrative": string (2-3 sentence risk story)
   - "recommended_budget": string (suggested retention budget with justification)
   - "expected_roi": string (expected ROI of the retention program)
   - "competitive_context": string (industry benchmarking context)

7. "action_roadmap" — array of 4 objects representing quarterly actions:
   - "quarter": string ("Q1", "Q2", "Q3", "Q4")
   - "focus": string
   - "actions": array of strings (2-3 specific actions)
   - "expected_impact": string
   - "budget_allocation_pct": float (should sum to 100)
   - "kpi_target": string

Use specific numbers from the data. Make it compelling for a CEO presentation.
Do not include any text outside the JSON.
"""

    try:
        response = ai_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            ),
        )
        ai_dashboard = json.loads(response.text)

        # Merge AI insights with real computed data
        ai_dashboard["computed_data"] = {
            "risk_distribution": risk_distribution,
            "scatter_data": scatter_data,
            "segment_summary": segment_summary,
            "contract_analysis": contract_analysis,
            "tenure_analysis": tenure_analysis,
            "portfolio_stats": {
                "total_customers": len(customers),
                "avg_churn": round(float(probs.mean()), 4),
                "total_monthly_revenue": round(float(mcs.sum()), 2),
                "total_clv": round(float(clvs.sum()), 2),
                "total_clv_at_risk": round(float((probs * clvs).sum()), 2),
                "median_tenure": round(float(np.median(tenures)), 1),
            },
        }

        return jsonify(ai_dashboard)

    except Exception as e:
        return jsonify({"error": f"Portfolio dashboard failed: {str(e)}"}), 500


# ─── 3. AI Comparative Strategy Visualizer ────────────
@app.route("/api/ai_studio/strategy_visual", methods=["POST"])
def ai_strategy_visual():
    """
    For a customer, run optimizer + what-if + AI analysis
    and return rich visual comparison data for strategies.
    """
    raw = request.json
    vec = raw_to_feature_vector(raw)
    vec_sc = scaler.transform(vec)
    prob = float(model.predict_proba(vec_sc)[0][1])
    mc = float(raw.get("MonthlyCharges", 70))
    clv = mc * 12 * 1.2

    # Run all scenarios
    scenarios = simulate_scenarios(raw)

    # Calculate strategy effectiveness
    strategies = []
    for s in scenarios["scenarios"]:
        reduction = -s["delta"] if s["delta"] < 0 else 0
        cost_estimate = abs(s["delta"]) * clv * 0.3  # rough cost model
        revenue_saved = reduction * clv
        roi = ((revenue_saved - cost_estimate) / cost_estimate * 100) if cost_estimate > 0 else 0

        strategies.append({
            "name": s["scenario"],
            "current_prob": round(prob, 4),
            "new_prob": s["new_probability"],
            "reduction_pct": round(reduction * 100, 2),
            "cost_estimate": round(cost_estimate, 2),
            "revenue_saved": round(revenue_saved, 2),
            "roi": round(roi, 1),
            "payback_months": round(cost_estimate / (revenue_saved / 12), 1) if revenue_saved > 0 else 999,
        })

    strategies.sort(key=lambda s: s["roi"], reverse=True)

    prompt = f"""
You are a retention strategy consultant creating visual strategy comparison materials.

CUSTOMER: Churn probability {prob*100:.1f}%, Monthly charges ${mc:.2f}, CLV ${clv:.2f}
CUSTOMER PROFILE: {json.dumps(raw, indent=2)}

STRATEGY ANALYSIS:
{json.dumps(strategies[:8], indent=2)}

Generate JSON with these EXACT keys:

1. "strategy_comparison_chart" — object for a grouped bar chart:
   - "categories": array of strategy names (top 6)
   - "series": array of 3 objects, each with "name" and "data" (array of values):
     * "Churn Reduction (%)"
     * "Cost ($)"
     * "ROI (%)"

2. "cost_benefit_scatter" — array of objects for bubble chart:
   - "strategy": string
   - "cost": float (x-axis)
   - "benefit": float (y-axis, revenue saved)
   - "bubble_size": float (ROI)
   - "color": hex
   - "recommended": boolean
   Include top 8 strategies

3. "implementation_timeline" — array of objects for a Gantt-like chart:
   - "strategy": string
   - "start_week": integer
   - "duration_weeks": integer
   - "difficulty": "easy" | "medium" | "hard"
   - "color": hex
   - "dependencies": array of strategy names (if any)

4. "risk_reduction_waterfall" — array of objects showing cumulative risk reduction:
   - "step": string (strategy applied)
   - "individual_reduction": float (percentage points)
   - "cumulative_prob": float (churn prob after this step)
   - "type": "reduction" | "total"
   Start with current probability and show how each strategy stacks up

5. "recommendation_card" — object with:
   - "best_single_strategy": object with name, why, expected_outcome
   - "best_combination": object with strategies (array), combined_reduction, total_cost, combined_roi
   - "quick_win": object with name, effort, timeline, expected_reduction
   - "avoid": object with name, reason (strategy that's not worth it)

6. "financial_projection" — object with:
   - "months": array of integers 1-12
   - "no_action": array of floats (cumulative revenue loss without action)
   - "with_best_strategy": array of floats (cumulative with best strategy)
   - "with_combination": array of floats (cumulative with best combination)
   - "break_even_month": integer

Use actual numbers from the analysis. Be specific and actionable.
Do not include any text outside the JSON.
"""

    try:
        response = ai_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            ),
        )
        visual_data = json.loads(response.text)

        visual_data["raw_strategies"] = strategies
        visual_data["customer_summary"] = {
            "churn_probability": round(prob, 4),
            "monthly_charges": mc,
            "clv": round(clv, 2),
            "risk_level": "High" if prob >= 0.7 else "Medium" if prob >= 0.4 else "Low",
        }

        return jsonify(visual_data)

    except Exception as e:
        return jsonify({"error": f"Strategy visualization failed: {str(e)}"}), 500


# ─── 4. AI Churn Story Generator (narrative + visuals) ─
@app.route("/api/ai_studio/churn_story", methods=["POST"])
def ai_churn_story():
    """
    Generate a data-driven 'story' about a customer's churn journey
    with structured data for an animated/scrollytelling visualization.
    """
    raw = request.json
    vec = raw_to_feature_vector(raw)
    vec_sc = scaler.transform(vec)
    prob = float(model.predict_proba(vec_sc)[0][1])
    factors = get_top_factors(raw, prob)
    timeline = forecast_churn_timeline(raw, months=12)
    scenarios = simulate_scenarios(raw)

    prompt = f"""
You are a data storytelling expert creating an interactive customer journey narrative.

CUSTOMER DATA:
- Churn Probability: {prob*100:.1f}%
- Profile: {json.dumps(raw, indent=2)}
- Risk Factors: {json.dumps(factors, indent=2)}
- 12-Month Forecast: {json.dumps(timeline['timeline'][:6], indent=2)}
- Best Intervention: {json.dumps(scenarios['scenarios'][0], indent=2) if scenarios['scenarios'] else 'None'}

Generate a JSON "churn story" with these EXACT keys:

1. "story_chapters" — array of 5-6 objects (the story unfolds in chapters):
   Each chapter has:
   - "chapter": integer (1-6)
   - "title": string (engaging chapter title)
   - "narrative": string (2-3 sentences, written like a story)
   - "visualization_type": one of "gauge", "line_chart", "bar_chart", "radar", "comparison", "outcome"
   - "chart_data" — object with chart-specific data:
     * For "gauge": {{"value": float, "max": 100, "zones": [{{"min": 0, "max": 40, "color": "#4CAF50"}}, ...]}}
     * For "line_chart": {{"labels": [...], "datasets": [{{"label": str, "data": [...], "color": hex}}]}}
     * For "bar_chart": {{"labels": [...], "values": [...], "colors": [...]}}
     * For "radar": {{"labels": [...], "values": [...], "benchmark": [...]}}
     * For "comparison": {{"before": {{"label": str, "value": float}}, "after": {{"label": str, "value": float}}, "improvement": str}}
     * For "outcome": {{"scenarios": [{{"name": str, "probability": float, "revenue_impact": float, "color": hex}}]}}
   - "key_metric": object with "label", "value", "unit"
   - "emotion": one of "concern", "insight", "hope", "urgency", "resolution"

2. "story_summary" — object with:
   - "one_liner": string (the story in one sentence)
   - "key_number": string (the most impactful number)
   - "call_to_action": string

3. "customer_persona" — object with:
   - "archetype": string (e.g., "The Frustrated Newcomer", "The Silent Leaver")
   - "description": string (2 sentences describing this persona)
   - "emoji": string (representative emoji)
   - "typical_behavior": array of 3 strings
   - "retention_approach": string

Make the story compelling and data-driven. Use the actual numbers.
The story should flow logically: introduce → analyze → risk → opportunity → resolution.
Do not include any text outside the JSON.
"""

    try:
        response = ai_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            ),
        )
        story = json.loads(response.text)

        story["model_data"] = {
            "churn_probability": round(prob, 4),
            "factors": factors,
            "timeline": timeline,
            "top_scenario": scenarios["scenarios"][0] if scenarios["scenarios"] else None,
        }

        return jsonify(story)

    except Exception as e:
        return jsonify({"error": f"Story generation failed: {str(e)}"}), 500


# ─── 5. AI Benchmark Comparison ──────────────────────
@app.route("/api/ai_studio/benchmark", methods=["POST"])
def ai_benchmark():
    """
    Compare this customer or portfolio against industry benchmarks.
    Returns structured data for visual benchmark comparison.
    """
    data = request.json

    # Get current dashboard metrics
    with prediction_log_lock:
        metrics = dict(dashboard_metrics)

    prompt = f"""
You are a telecom industry analyst creating a benchmarking report.

OUR COMPANY DATA:
- Predictions analyzed: {metrics['total_predictions']}
- Average churn probability: {metrics['avg_probability']*100:.1f}%
- High risk customers: {metrics['high_risk_count']}
- Revenue at risk: ${metrics['revenue_at_risk']:,.0f}

Additional context (if provided): {json.dumps(data, indent=2)}

Generate a JSON benchmarking report with these EXACT keys:

1. "industry_comparison" — array of objects for a multi-bar chart:
   Each object has:
   - "metric": string (e.g., "Churn Rate", "ARPU", "CLV", "NPS", "Retention Rate", "CSAT")
   - "our_value": float
   - "industry_avg": float
   - "top_quartile": float
   - "bottom_quartile": float
   - "unit": string ("%" or "$" or "score")
   Include 6-8 metrics

2. "maturity_radar" — object for a radar/spider chart:
   - "dimensions": array of strings (8 dimensions like "Predictive Analytics", "Personalization", etc.)
   - "our_scores": array of floats (0-10)
   - "industry_avg": array of floats (0-10)
   - "leaders": array of floats (0-10)

3. "competitive_landscape" — array of objects for a positioning map:
   - "company": string (use generic names like "Our Company", "Competitor A", etc.)
   - "churn_rate": float
   - "customer_satisfaction": float (0-100)
   - "market_share": float (percentage)
   - "bubble_size": float
   - "color": hex
   Include 5-6 companies

4. "improvement_roadmap" — array of objects:
   - "area": string
   - "current_level": string ("Basic", "Developing", "Advanced", "Leading")
   - "target_level": string
   - "gap_score": float (0-10)
   - "priority": "Critical" | "High" | "Medium" | "Low"
   - "estimated_impact": string
   Include 6 improvement areas

5. "benchmark_kpis" — array of 4 objects:
   - "name": string
   - "our_value": string
   - "benchmark": string
   - "gap": string
   - "status": "above" | "below" | "at"
   - "color": hex

Use realistic telecom industry benchmarks (typical telco churn 1.5-3% monthly, 15-30% annual).
Do not include any text outside the JSON.
"""

    try:
        response = ai_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            ),
        )
        benchmark = json.loads(response.text)
        benchmark["generated_at"] = datetime.utcnow().isoformat()
        return jsonify(benchmark)

    except Exception as e:
        return jsonify({"error": f"Benchmark failed: {str(e)}"}), 500


if __name__ == "__main__":
    print("\n  Churn Dashboard running at http://localhost:5000\n")
    app.run(debug=True, port=5000)
