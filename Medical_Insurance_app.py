# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import date

st.set_page_config(page_title="Premium Calculator + Claim Predictor", layout="wide")

# -----------------------------
# Config / filenames
# -----------------------------
CLASSIFIER_PKL = "best_claim_model.pkl"
REGRESSOR_PKL = "best_claim_severity_compressed.pkl"

PREMIUM_TABLES_JSON = "premium_tables.json"
MATERNITY_JSON = "maternity_rates.json"
DAILY_CASH_JSON = "daily_cash.json"

MAX_LOADING_PCT = 1.0  # 100%

# Expected features (exact order your models trained on)
EXPECTED_FEATURES = [
    "person_id", "age", "sex", "region", "urban_rural", "income", "education",
    "marital_status", "employment_status", "household_size", "dependents",
    "bmi", "smoker", "alcohol_freq", "visits_last_year",
    "hospitalizations_last_3yrs", "days_hospitalized_last_3yrs",
    "medication_count", "systolic_bp", "diastolic_bp", "ldl", "hba1c",
    "chronic_count", "hypertension", "diabetes", "asthma", "copd", "cardiovascular_disease",
    "cancer_history", "kidney_disease", "liver_disease", "arthritis",
    "mental_health", "proc_imaging_count", "proc_surgery_count",
    "proc_physio_count", "proc_consult_count", "proc_lab_count",
    "is_high_risk", "had_major_procedure",
    "annual_premium", "monthly_premium", "risk_score"
]

# -----------------------------
# Load resources
# -----------------------------
@st.cache_resource
def load_json_tables():
    with open(PREMIUM_TABLES_JSON, "r") as f:
        premium_tables = json.load(f)
    with open(MATERNITY_JSON, "r") as f:
        maternity_rates = json.load(f)
    with open(DAILY_CASH_JSON, "r") as f:
        daily_cash_raw = json.load(f)
    # normalize DAILY_CASH: accept both { "daily_cash": {...} } or direct {...}
    if isinstance(daily_cash_raw, dict) and "daily_cash" in daily_cash_raw:
        daily_cash = daily_cash_raw["daily_cash"]
    else:
        daily_cash = daily_cash_raw
    return premium_tables, maternity_rates, daily_cash

@st.cache_resource
def load_models():
    clf = joblib.load(CLASSIFIER_PKL)
    reg = joblib.load(REGRESSOR_PKL)
    return clf, reg

PREMIUM_TABLES, MATERNITY_RATES, DAILY_CASH = load_json_tables()
CLASSIFIER, REGRESSOR = load_models()

# -----------------------------
# Helpers: age bands, SI lookup
# -----------------------------
AGE_BANDS = [
    (0, 17), (18, 25), (26, 30), (31, 35), (36, 40),
    (41, 45), (46, 50), (51, 55), (56, 60), (61, 65),
    (66, 70), (71, 75), (76, 200)
]

def get_age_band_label(age):
    for lo, hi in AGE_BANDS:
        if lo <= age <= hi:
            return f"{lo}-{hi if hi < 200 else '75+'}"
    return "75+"

def closest_si(si, zone_table):
    keys = sorted(int(k) for k in zone_table.keys() if str(k).isdigit())
    if si in keys: return si
    return int(min(keys, key=lambda k: abs(k - si)))

def get_premium_for_person(age, zone, si):
    zone = zone.upper()
    if zone not in PREMIUM_TABLES:
        raise ValueError(f"Zone {zone} not found in premium_tables.json")
    zone_table = {int(k):v for k,v in PREMIUM_TABLES[zone].items()}
    si_key = closest_si(si, zone_table)
    age_label = get_age_band_label(age)
    table = zone_table[si_key]
    if age_label in table:
        return float(table[age_label]), si_key
    # fallback try prefix
    for k,v in table.items():
        if str(k).startswith(str(age_label.split("-")[0])):
            return float(v), si_key
    # if missing, raise
    raise KeyError(f"Age band {age_label} not found for SI {si_key} zone {zone}")

def si_category_flag(si):
    if si <= 500000:
        return 0
    elif si <= 1500000:
        return 1
    return 2

# daily cash access helper (compatible with different JSON key names)
def get_daily_cash_value(cash_table, age, si):
    # cash_table expected to have either keys like "<=50"/"51-60"/">60"
    # or keys like "upto_50_years","51_to_60_years","above_60_years"
    # and internal keys either arrays (index 0/1/2) or descriptive keys.
    # Determine age group key
    if age <= 50:
        age_keys_candidates = ["<=50", "upto_50_years", "up_to_50", "0-50"]
    elif age <= 60:
        age_keys_candidates = ["51-60", "51_to_60_years", "51_60"]
    else:
        age_keys_candidates = [">60", "above_60_years", "above_60"]

    # determine SI bracket
    bracket_index = si_category_flag(si)  # 0/1/2

    # find age key
    age_key = None
    for k in age_keys_candidates:
        if k in cash_table:
            age_key = k
            break
    if age_key is None:
        # fallback: try to find closest by substring
        for k in cash_table.keys():
            lk = k.lower()
            if "50" in lk and "51" not in lk and "above" not in lk:
                age_key = k; break
            if "51" in lk or "51_to_60" in lk or "51-60" in lk:
                age_key = k; break
            if "above" in lk or "60" in lk:
                age_key = k; break
    if age_key is None:
        raise KeyError("No matching age group found in daily cash table")

    group = cash_table[age_key]

    # group may be dict with descriptive keys or list/array
    if isinstance(group, dict):
        # try descriptive keys
        keys_map = [
            "base_si_upto_5_lakh",
            "base_si_5_to_15_lakh",
            "base_si_above_15_lakh",
            "si_upto_5l", "si_5_15l", "si_above_15l"
        ]
        for k in keys_map:
            if k in group:
                # mapping bracket_index -> first available mapping
                # build list of available in order:
                ordered = []
                if "base_si_upto_5_lakh" in group:
                    ordered = [group.get("base_si_upto_5_lakh"), group.get("base_si_5_to_15_lakh"), group.get("base_si_above_15_lakh")]
                else:
                    # fallback: all values in dict
                    ordered = list(group.values())
                # safe index
                idx = min(bracket_index, len(ordered)-1)
                return float(ordered[idx])
        # if none of the expected keys exist, try to take values as list
        vals = list(group.values())
        idx = min(bracket_index, len(vals)-1)
        return float(vals[idx])
    else:
        # assume list/array-like
        idx = min(bracket_index, len(group)-1)
        return float(group[idx])

def age_group_for_daily(age):
    if age <= 50:
        return "<=50"
    elif age <= 60:
        return "51-60"
    return ">60"

# Floater discounts
FLOATER_DISCOUNTS_AGE_40_50 = {
    "one_adult_parents": 0.20,
    "one_adult_children": 0.20,
    "one_adult_children_parents": 0.30,
    "two_adults": 0.30,
    "two_adults_children": 0.30,
    "two_adults_parents": 0.30,
    "two_adults_children_parents": 0.30
}
FLOATER_DISCOUNTS_OTHER = {
    "one_adult_parents": 0.15,
    "one_adult_children": 0.15,
    "one_adult_children_parents": 0.30,
    "two_adults": 0.25,
    "two_adults_children": 0.25,
    "two_adults_parents": 0.30,
    "two_adults_children_parents": 0.30
}
FAMILY_DISCOUNT = 0.05

# -----------------------------
# Risk score & loading functions
# -----------------------------
def cond_flag(val):
    return val in (1, True, "yes", "Yes", "true", "True", "1")

def compute_risk_score(member):
    """
    Returns (score: float, breakdown: dict)
    """
    breakdown = {}
    score = 0.0

    # ---- AGE ----
    age = int(member.get("age", 40))
    if age <= 25: pts = 0
    elif age <= 35: pts = 1
    elif age <= 45: pts = 2
    elif age <= 55: pts = 4
    elif age <= 65: pts = 6
    elif age <= 75: pts = 8
    else: pts = 10
    breakdown["Age"] = pts
    score += pts

    # ---- BMI ----
    bmi = float(member.get("bmi", 24))
    if bmi < 18.5: pts = 1
    elif bmi < 25: pts = 0
    elif bmi < 30: pts = 2
    else: pts = 4
    breakdown["BMI"] = pts
    score += pts

    # ---- SMOKER ----
    # Accept both 'Never','Former','Current' and older variants
    smoker = str(member.get("smoker","Never")).lower()
    if smoker in ("current", "yes", "1", "true", "current smoker"):
        pts = 5
    elif smoker in ("former", "ex", "former smoker"):
        pts = 2
    else:
        pts = 0
    breakdown["Smoker"] = pts
    score += pts

    # ---- ALCOHOL ----
    alcohol = str(member.get("alcohol_freq","None")).lower()
    if alcohol in ("none","no"): pts = 0
    elif alcohol in ("occasionally","<=1/week","occasional"): pts = 1
    elif alcohol in ("weekly","2-4/week"): pts = 2
    else: pts = 4
    breakdown["Alcohol"] = pts
    score += pts

    # ---- BP ----
    systolic = float(member.get("systolic_bp",120))
    diastolic = float(member.get("diastolic_bp",80))

    if systolic < 120: sys_pts = 0
    elif systolic < 140: sys_pts = 1
    elif systolic < 160: sys_pts = 3
    else: sys_pts = 5

    if diastolic < 80: dia_pts = 0
    elif diastolic < 90: dia_pts = 1
    elif diastolic < 100: dia_pts = 2
    else: dia_pts = 4

    bp_pts = max(sys_pts, dia_pts)
    breakdown["Blood Pressure"] = bp_pts
    score += bp_pts

    # ---- LDL ----
    ldl = float(member.get("ldl",100))
    if ldl < 100: pts = 0
    elif ldl < 130: pts = 1
    elif ldl < 160: pts = 2
    elif ldl < 190: pts = 3
    else: pts = 5
    breakdown["LDL"] = pts
    score += pts

    # ---- HbA1c ----
    hba1c = float(member.get("hba1c",5.5))
    if hba1c < 5.7: pts = 0
    elif hba1c < 6.5: pts = 2
    else: pts = 5
    breakdown["HbA1c"] = pts
    score += pts

    # ---- Chronic conditions ----
    chronic_map = {
        "Hypertension": ("hypertension", 3),
        "Diabetes": ("diabetes", 4),
        "Asthma": ("asthma", 2),
        "COPD": ("copd", 2),
        "CVD": ("cardiovascular_disease", 6),
        "Cancer History": ("cancer_history", 8),
        "Kidney Disease": ("kidney_disease", 6),
        "Liver Disease": ("liver_disease", 5),
        "Arthritis": ("arthritis", 1),
        "Mental Health": ("mental_health", 2)
    }

    for label, (field, pts_val) in chronic_map.items():
        pts = pts_val if cond_flag(member.get(field,0)) else 0
        breakdown[label] = pts
        score += pts

    # ---- Hospitalizations ----
    hosp = int(member.get("hospitalizations_last_3yrs",0))
    if hosp == 0: pts = 0
    elif hosp == 1: pts = 2
    elif hosp == 2: pts = 4
    else: pts = 6
    breakdown["Hospitalizations (3 yrs)"] = pts
    score += pts

    # ---- Days hospitalised ----
    days = int(member.get("days_hospitalized_last_3yrs",0))
    if days <= 3: pts = 0
    elif days <= 10: pts = 2
    elif days <= 30: pts = 4
    else: pts = 6
    breakdown["Days Hospitalized"] = pts
    score += pts

    # ---- Visits ----
    visits = int(member.get("visits_last_year",0))
    if visits < 2: pts = 0
    elif visits <= 5: pts = 1
    elif visits <= 12: pts = 2
    else: pts = 4
    breakdown["Visits Last Year"] = pts
    score += pts

    # ---- Procedures ----
    surg_pts = 2 * int(member.get("proc_surgery_count",0))
    img_pts = 0.5 * int(member.get("proc_imaging_count",0))
    lab_pts = 0.25 * int(member.get("proc_lab_count",0))

    breakdown["Surgeries"] = surg_pts
    breakdown["Imaging Procedures"] = img_pts
    breakdown["Lab Procedures"] = lab_pts

    score += surg_pts + img_pts + lab_pts

    # ---- Risk Flags ----
    if cond_flag(member.get("is_high_risk",0)):
        breakdown["High Risk Flag"] = 10
        score += 10
    else:
        breakdown["High Risk Flag"] = 0

    if cond_flag(member.get("had_major_procedure",0)):
        breakdown["Major Procedure"] = 4
        score += 4
    else:
        breakdown["Major Procedure"] = 0

    return float(score), breakdown


def risk_score_to_loading(score):
    if score <= 4: return 0.0
    if score <= 9: return 0.10
    if score <= 14: return 0.25
    if score <= 19: return 0.50
    if score <= 24: return 0.75
    return 1.0

def apply_ncr(base_amount, ncr_type, claim_free_years):
    if ncr_type == "NCD":
        pct = min(0.05 * claim_free_years, 0.50)
        return base_amount * (1 - pct), {"type":"NCD","pct":pct}
    if ncr_type == "CB":
        cb_pct = min(0.10 * claim_free_years, 2.0)
        eff_discount = cb_pct / (1 + cb_pct)
        return base_amount * (1 - eff_discount), {"type":"CB","cb_pct":cb_pct,"effective_discount":eff_discount}
    return base_amount, {"type":"none"}

# -----------------------------
# Family category util
# -----------------------------
def determine_family_category(members):
    adults = sum(1 for m in members if m["role"].lower()=="adult")
    children = sum(1 for m in members if m["role"].lower()=="child")
    parents = sum(1 for m in members if m["role"].lower()=="parent")
    if adults == 1 and parents >= 1 and children == 0: return "one_adult_parents"
    if adults == 1 and children >= 1 and parents == 0: return "one_adult_children"
    if adults == 1 and children >= 1 and parents >= 1: return "one_adult_children_parents"
    if adults == 2 and children == 0 and parents == 0: return "two_adults"
    if adults == 2 and children >= 1 and parents == 0: return "two_adults_children"
    if adults == 2 and parents >= 1: return "two_adults_parents"
    if adults == 2 and children >= 1 and parents >= 1: return "two_adults_children_parents"
    if adults >= 2: return "two_adults_children"
    return "one_adult_children"

# -----------------------------
# Premium calculator (individual & floater)
# -----------------------------
def calculate_premium(members, mode="individual", floater_si=None, maternity=False,
                      daily_cash=False, ncr_type="none", last_claim_year=None,
                      direct_channel_discount=0.0):
    details = {"members":[], "base_total":0.0, "discounts":{}, "optional_covers":{}, "final":{}}

    if mode == "individual":
        # per-member lookup
        member_info = []
        for m in members:
            age = int(m["age"])
            zone = m["zone"].upper()   # premium zone (A/B/C)
            si = int(m["si"])
            base_prem, si_used = get_premium_for_person(age, zone, si)
            member_info.append({"member":m, "base":base_prem, "si_used":si_used, "zone":zone})
        base_sum = sum(x["base"] for x in member_info)
        details["members"] = member_info
        details["base_total"] = base_sum

        # family discount
        family_disc = FAMILY_DISCOUNT * base_sum if len(member_info) > 1 else 0.0
        details["discounts"]["family_discount_amount"] = family_disc
        post_family = base_sum - family_disc

        # direct channel
        direct_disc = post_family * direct_channel_discount
        details["discounts"]["direct_channel_discount_amount"] = direct_disc
        base_after_discounts = post_family - direct_disc

        # NCR
        if last_claim_year:
            claim_free_years = max(0, date.today().year - int(last_claim_year) - 1)
        else:
            claim_free_years = 0
        base_after_ncr, ncr_info = apply_ncr(base_after_discounts, ncr_type, claim_free_years)
        details["discounts"]["ncr"] = ncr_info

        # Risk & loading (per member, proportionate)
        loading_list = []
        loading_total = 0.0
        for info in member_info:
            m = info["member"]
            score, breakdown = compute_risk_score(m)
            loading_pct = min(risk_score_to_loading(score), MAX_LOADING_PCT)
            share = info["base"] / base_sum if base_sum > 0 else 0
            post_disc_member = base_after_discounts * share
            loading_amt = post_disc_member * loading_pct
            loading_list.append({
                "age":m["age"],
                "score":score,
                "breakdown":breakdown,
                "loading_pct":loading_pct,
                "loading_amount":loading_amt
            })
            loading_total += loading_amt
        details["discounts"]["loading_details"] = loading_list
        details["discounts"]["total_loading_amount"] = loading_total

        # optional covers
        optional_total = 0.0
        if maternity:
            highest_si = max(int(info["si_used"]) for info in member_info)
            if highest_si < 350000:
                details["optional_covers"]["maternity"] = {"eligible":False, "message":f"Highest SI {highest_si} < 350000 - not eligible", "suggestion": min(sorted([int(k) for k in PREMIUM_TABLES[member_info[0]['zone']].keys()]), key=lambda k: abs(k-350000))}
            else:
                maternity_dict = MATERNITY_RATES["maternity_newborn_per_family"]
                mat_si = closest_si(highest_si, maternity_dict)
                mat_cost = float(maternity_dict[str(mat_si)])
                details["optional_covers"]["maternity"] = {"eligible":True, "si_used":mat_si, "cost":mat_cost}
                optional_total += mat_cost
        if daily_cash:
            dc_total = 0.0
            for info in member_info:
                a = int(info["member"]["age"])
                si = int(info["si_used"])
                dc_val = get_daily_cash_value(DAILY_CASH["individual"], a, si)
                dc_total += int(dc_val)
            details["optional_covers"]["daily_cash"] = {"cost":dc_total}
            optional_total += dc_total

        details["optional_covers"]["total_optional"] = optional_total

        final_before_gst = base_after_ncr + loading_total + optional_total
        details["final"]["annual_premium"] = final_before_gst
        
        details["final"]["final_billed_amount"] = final_before_gst 
        return details

    else:
        # floater
        if floater_si is None:
            raise ValueError("floater_si required for floater mode")
        oldest = max(members, key=lambda x: int(x["age"]))
        zone = oldest["zone"].upper()
        base_prem, si_used = get_premium_for_person(int(oldest["age"]), zone, floater_si)
        details["base_total"] = base_prem
        details["members"] = [{"member": m} for m in members]

        family_cat = determine_family_category(members)
        if 40 <= int(oldest["age"]) <= 50:
            disc_pct = FLOATER_DISCOUNTS_AGE_40_50.get(family_cat, 0)
        else:
            disc_pct = FLOATER_DISCOUNTS_OTHER.get(family_cat, 0)
        floater_disc = base_prem * disc_pct
        details["discounts"]["floater_discount_amount"] = floater_disc

        post_floater = base_prem - floater_disc
        direct_disc = post_floater * direct_channel_discount
        details["discounts"]["direct_channel_discount_amount"] = direct_disc
        base_after_discounts = post_floater - direct_disc

        # NCR
        if last_claim_year:
            claim_free_years = max(0, date.today().year - int(last_claim_year) - 1)
        else:
            claim_free_years = 0
        base_after_ncr, ncr_info = apply_ncr(base_after_discounts, ncr_type, claim_free_years)
        details["discounts"]["ncr"] = ncr_info

        # risk scores for each member -> average
        scores = []
        breakdowns = []
        for m in members:
            s, b = compute_risk_score(m)
            scores.append(s)
            breakdowns.append(b)
        avg_score = float(np.mean(scores))
        loading_pct = min(risk_score_to_loading(avg_score), MAX_LOADING_PCT)
        loading_amt = base_after_ncr * loading_pct
        details["discounts"]["avg_score"] = avg_score
        details["discounts"]["loading_pct"] = loading_pct
        details["discounts"]["total_loading_amount"] = loading_amt
        details["discounts"]["per_member_scores"] = scores
        details["discounts"]["per_member_breakdowns"] = breakdowns

        # optional covers
        optional_total = 0.0
        if maternity:
            if floater_si < 350000:
                details["optional_covers"]["maternity"] = {"eligible":False, "message":"Floater SI < 350000 - not eligible", "suggestion": min(sorted([int(k) for k in PREMIUM_TABLES[zone].keys()]), key=lambda k: abs(k-350000))}
            else:
                maternity_dict = MATERNITY_RATES["maternity_newborn_per_family"]
                mat_si = closest_si(floater_si, maternity_dict)

                mat_cost = float(maternity_dict[str(mat_si)])
                details["optional_covers"]["maternity"] = {"eligible":True, "si_used":mat_si, "cost":mat_cost}
                optional_total += mat_cost
        if daily_cash:
            dc_val = get_daily_cash_value(DAILY_CASH["floater"], int(oldest["age"]), floater_si)
            details["optional_covers"]["daily_cash"] = {"cost":int(dc_val)}
            optional_total += int(dc_val)

        details["optional_covers"]["total_optional"] = optional_total

        final_before_gst = base_after_ncr + loading_amt + optional_total
        details["final"]["annual_premium"] = final_before_gst
        details["final"]["final_billed_amount"] = final_before_gst
        return details

# Normalization constant for risk score -> 0..1
MAX_RISK_SCORE = 127.0  # adjust if you want a different cap

def normalize_risk_score(raw_score, cap=MAX_RISK_SCORE):
    """Map raw_score to [0,1] by dividing by cap and clipping at 1."""
    try:
        norm = float(raw_score) / float(cap)
    except Exception:
        return 0.0
    if norm < 0:
        return 0.0
    return min(norm, 1.0)



# -----------------------------
# UI flow
# -----------------------------
st.title("Health Insurance — Premium Calculator & Claim Predictor")

st.markdown("Select policy type and enter full details. For floater, enter details for each member.")

policy_type = st.radio("Policy type", ("Individual", "Floater"))

# policy-level inputs (some are used for ML columns)
# with st.expander("Policy-level inputs (plan / cost fields)"):
#     plan_type = st.selectbox("Plan Type", ["HMO","PPO","POS","Standard"], index=3)
#     network_tier = st.selectbox("Network Tier", ["Bronze","Silver","Gold","Platinum"], index=1)
#     deductible = st.number_input("Deductible", 0, 100000, 1000)
#     copay = st.number_input("Copay", 0, 10000, 0)
#     policy_term_years = st.number_input("Policy Term Years", 1, 30, 1)
#     policy_changes_last_2yrs = st.number_input("Policy changes in last 2 years", 0, 10, 0)
#     provider_quality = st.number_input("Provider quality (1-5)", 1.0, 5.0, 3.5)

# number of members
# num_members = st.number_input("Number of members on policy (1-10)", min_value=1, max_value=10, value=1)

# number of members
if policy_type == "Individual":
    st.write("Number of members: **1 (fixed for individual policy)**")
    num_members = 1
else:
    num_members = st.number_input(
        "Number of members on policy (2-10)",
        min_value=2,
        max_value=10,
        value=2
    )

members = []
st.markdown("Enter member-level details (all fields required):")
for i in range(int(num_members)):
    st.markdown(f"--- Member #{i+1}")
    cols = st.columns(3)
    with cols[0]:
        person_id = st.number_input(f"person_id (member {i+1})", min_value=0, value=0, key=f"pid_{i}")
        age = st.number_input(f"Age (member {i+1})", 0, 120, 30, key=f"age_{i}")
        # MODEL region (for ML) - textual labels expected by ML
        region_model = st.selectbox(f"Region (for model) (member {i+1})", ["North","South","East","West","Central"], key=f"rmod_{i}")
        # PREMIUM zone (A/B/C)
        zone = st.selectbox(f"Premium Zone (A/B/C) (member {i+1})", ["A","B","C"], key=f"zone_{i}")
        urban_rural = st.selectbox(f"Urban/Rural (member {i+1})", ["Urban","Rural","Suburban"], key=f"ur_{i}")
        income = st.number_input(f"Income (member {i+1})", 0.0, 1e9, 0.0, key=f"in_{i}")
        education = st.selectbox(f"Education (member {i+1})", ["No HS","HS","Some College","Bachelor","Master","Doctorate"], key=f"edu_{i}")
        marital_status = st.selectbox(f"Marital Status (member {i+1})", ["Single","Married","Divorced","Widowed"], key=f"mar_{i}")
    with cols[1]:
        employment_status = st.selectbox(f"Employment Status (member {i+1})", ["Employed","Self-employed","Unemployed","Retired"], key=f"emp_{i}")
        household_size = st.number_input(f"Household size (member {i+1})", 1, 20, 1, key=f"hs_{i}")
        dependents = st.number_input(f"Dependents (member {i+1})", 0, 20, 0, key=f"dep_{i}")
        bmi = st.number_input(f"BMI (member {i+1})", 10.0, 60.0, 24.0, key=f"bmi_{i}")
        # MODEL smoker labels
        smoker = st.selectbox(f"Smoker (member {i+1})", ["Never","Former","Current"], key=f"smk_{i}")
        alcohol_freq = st.selectbox(f"Alcohol freq (member {i+1})", ["None","Occasionally","Weekly","Daily"], key=f"alc_{i}")
        visits_last_year = st.number_input(f"Visits last year (member {i+1})", 0, 200, 1, key=f"vis_{i}")
        hospitalizations_last_3yrs = st.number_input(f"Hospitalizations last 3 yrs (member {i+1})", 0, 50, 0, key=f"hosp_{i}")
        days_hospitalized_last_3yrs = st.number_input(f"Days hospitalized last 3 yrs (member {i+1})", 0, 1000, 0, key=f"days_{i}")
    with cols[2]:
        medication_count = st.number_input(f"Medication count (member {i+1})", 0, 100, 0, key=f"med_{i}")
        systolic_bp = st.number_input(f"Systolic BP (member {i+1})", 50, 300, 120, key=f"sys_{i}")
        diastolic_bp = st.number_input(f"Diastolic BP (member {i+1})", 30, 200, 80, key=f"dia_{i}")
        ldl = st.number_input(f"LDL (member {i+1})", 0.0, 500.0, 100.0, key=f"ldl_{i}")
        hba1c = st.number_input(f"HbA1c (member {i+1})", 3.0, 20.0, 5.5, key=f"hba_{i}")
        chronic_count = st.number_input(f"Chronic count (member {i+1})", 0, 50, 0, key=f"chc_{i}")
        hypertension = st.selectbox(f"Hypertension (member {i+1})", [0,1], key=f"ht_{i}")
        diabetes = st.selectbox(f"Diabetes (member {i+1})", [0,1], key=f"db_{i}")
        asthma = st.selectbox(f"Asthma (member {i+1})", [0,1], key=f"as_{i}")
        copd = st.selectbox(f"COPD (member {i+1})", [0,1], key=f"cp_{i}")
        cardiovascular_disease = st.selectbox(f"Cardiovascular disease (member {i+1})", [0,1], key=f"cv_{i}")
        cancer_history = st.selectbox(f"Cancer history (member {i+1})", [0,1], key=f"cn_{i}")
        kidney_disease = st.selectbox(f"Kidney disease (member {i+1})", [0,1], key=f"kd_{i}")
        liver_disease = st.selectbox(f"Liver disease (member {i+1})", [0,1], key=f"lv_{i}")
        arthritis = st.selectbox(f"Arthritis (member {i+1})", [0,1], key=f"ar_{i}")
        mental_health = st.selectbox(f"Mental health (member {i+1})", [0,1], key=f"mh_{i}")
        proc_imaging_count = st.number_input(f"Imaging proc count (member {i+1})", 0, 500, 0, key=f"pi_{i}")
        proc_surgery_count = st.number_input(f"Surgery proc count (member {i+1})", 0, 200, 0, key=f"ps_{i}")
        proc_physio_count = st.number_input(f"Physio proc count (member {i+1})", 0, 500, 0, key=f"pp_{i}")
        proc_consult_count = st.number_input(f"Consult proc count (member {i+1})", 0, 500, 0, key=f"pc_{i}")
        proc_lab_count = st.number_input(f"Lab proc count (member {i+1})", 0, 500, 0, key=f"pl_{i}")
        is_high_risk = st.selectbox(f"High risk flag (member {i+1})", [0,1], key=f"hr_{i}")
        had_major_procedure = st.selectbox(f"Had major procedure (member {i+1})", [0,1], key=f"mp_{i}")

    # premium SI and role
    si_options = sorted([int(k) for k in PREMIUM_TABLES[zone].keys()])
    si = st.selectbox(f"Sum Insured (member {i+1})", options=si_options, key=f"si_{i}")
    role = st.selectbox(f"Role (member {i+1})", ["adult","child","parent"], key=f"role_{i}")

    member_dict = {
        "person_id": person_id, "age": age, "sex": st.selectbox(f"Sex (member {i+1})", ["Male","Female","Other"], key=f"sex_{i}"),
        # model-region vs premium-zone:
        "region_model": region_model, "zone": zone,
        "urban_rural": urban_rural, "income": income, "education": education,
        "marital_status": marital_status, "employment_status": employment_status,
        "household_size": household_size, "dependents": dependents,
        "bmi": bmi, "smoker": smoker, "alcohol_freq": alcohol_freq,
        "visits_last_year": visits_last_year, "hospitalizations_last_3yrs": hospitalizations_last_3yrs,
        "days_hospitalized_last_3yrs": days_hospitalized_last_3yrs,
        "medication_count": medication_count, "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp, "ldl": ldl, "hba1c": hba1c,
        "chronic_count": chronic_count,
        "hypertension": hypertension, "diabetes": diabetes, "asthma": asthma,
        "copd": copd, "cardiovascular_disease": cardiovascular_disease,
        "cancer_history": cancer_history, "kidney_disease": kidney_disease,
        "liver_disease": liver_disease, "arthritis": arthritis,
        "mental_health": mental_health, "proc_imaging_count": proc_imaging_count,
        "proc_surgery_count": proc_surgery_count, "proc_physio_count": proc_physio_count,
        "proc_consult_count": proc_consult_count, "proc_lab_count": proc_lab_count,
        "is_high_risk": is_high_risk, "had_major_procedure": had_major_procedure,
        "si": si, "role": role
    }

    members.append(member_dict)

# Policy-level optional inputs
st.write("### Policy options")
col_a, col_b = st.columns(2)
with col_a:
    maternity = st.checkbox("Add maternity cover (family-level)")
    daily_cash = st.checkbox("Add daily cash allowance")
with col_b:
    ncr_type = st.selectbox("No Claim Reward Type", ("none","NCD","CB"))
    last_claim_year_input = st.text_input("Year of last claim (YYYY) (leave blank if none)")
    direct_channel_discount = st.number_input("Direct channel discount (fraction 0-1)", 0.0, 1.0, 0.0)
    

floater_si = None
if policy_type == "Floater":
    # floater SI chosen once
    # use zone of oldest member by default
    floater_zone = max(members, key=lambda x: int(x["age"]))["zone"]
    floater_si = st.selectbox("Choose Floater Sum Insured", sorted([int(k) for k in PREMIUM_TABLES[floater_zone].keys()]))

# Button to calculate and predict
if st.button("Calculate premium & Predict claim"):
    # calculate risk scores and show breakdown per member
    per_member_scores = []
    per_member_breakdowns = []
    for idx, member in enumerate(members):
        score, breakdown = compute_risk_score(member)
        per_member_scores.append(score)
        per_member_breakdowns.append(breakdown)

        st.markdown(f"## Member #{idx+1} — Risk Score Breakdown")
        df_break = pd.DataFrame({
            "Factor": list(breakdown.keys()),
            "Points": list(breakdown.values())
        })
        st.table(df_break)
        norm = normalize_risk_score(score)
        st.markdown(f"### **Total Risk Score (raw): {score:.2f} — Normalized: {norm:.3f} (0-1)**")
        # optional: show risk bucket
        if norm < 0.25:
            st.info("Risk band: Low")
        elif norm < 0.6:
            st.warning("Risk band: Medium")
        else:
            st.error("Risk band: High")


    avg_score = float(np.mean(per_member_scores))
    if policy_type == "Floater":
        st.write(f"Average risk score (floater): **{avg_score:.2f}**")

    # calc premium
    last_claim_year = int(last_claim_year_input) if last_claim_year_input.isdigit() else None
    details = calculate_premium(members=members, mode="individual" if policy_type=="Individual" else "floater",
                                floater_si=floater_si, maternity=maternity, daily_cash=daily_cash,
                                ncr_type=ncr_type, last_claim_year=last_claim_year,
                                direct_channel_discount=direct_channel_discount)
    st.write("### Premium breakdown")
    st.json(details)

    annual_premium = details["final"]["annual_premium"]
    
    final_billed = details["final"]["final_billed_amount"]
    
    st.write(f"Annual premium : **₹{annual_premium:,.2f}**")
    
    st.write(f"Final billed amount: **₹{final_billed:,.2f}**")

    # ----------------------------------------------------------
    #  NEW CHECK: Annual premium must not exceed annual income
    # ----------------------------------------------------------

    # Compute total annual income across all members (or oldest, your choice)
    total_annual_income = sum(float(m.get("income", 0)) for m in members)
    max_affordable_premium = 0.40 * total_annual_income  # 40% affordability rule

    if annual_premium > max_affordable_premium:
        st.error(
            f"❌ Your annual premium ₹{annual_premium:,.2f} exceeds "
            f"40% of your household annual income (40% = ₹{max_affordable_premium:,.2f}).\n\n"
            "To proceed, please go back and:\n"
            "- Choose a lower Sum Insured, OR\n"
            "- Remove optional covers (maternity / daily cash), OR\n"
            "- Reduce add-on features.\n\n"
            "Once the premium is within the 40% affordability limit, you can run the prediction again."
        )
        st.stop()   # 🚫 Prevents claim prediction from running
    


    # Prepare ML input row (complete with all EXPECTED_FEATURES)
    def aggregate_for_model(members, details):
        row = {}
        # pick oldest member for categorical defaults
        oldest = max(members, key=lambda x: int(x["age"]))
        # numeric fields to average
        numeric_fields = ["age","income","household_size","dependents","bmi",
                          "visits_last_year","hospitalizations_last_3yrs","days_hospitalized_last_3yrs",
                          "medication_count","systolic_bp","diastolic_bp","ldl","hba1c",
                          "chronic_count","proc_imaging_count","proc_surgery_count","proc_physio_count",
                          "proc_consult_count","proc_lab_count"]
        for f in numeric_fields:
            vals = []
            for m in members:
                v = m.get(f)
                try:
                    vals.append(float(v))
                except:
                    vals.append(0.0)
            row[f] = float(np.mean(vals)) if len(vals)>0 else 0.0

        # boolean/chronic flags -> max
        bool_fields = ["hypertension","diabetes","asthma","copd","cardiovascular_disease",
                       "cancer_history","kidney_disease","liver_disease","arthritis","mental_health",
                       "is_high_risk","had_major_procedure"]
        for f in bool_fields:
            row[f] = int(any(int(m.get(f,0)) for m in members))

        # categorical / identifiers
        row["person_id"] = int(oldest.get("person_id", 0))
        row["age"] = float(np.mean([int(m.get("age",0)) for m in members]))
        row["sex"] = oldest.get("sex", "Male")
        # for model, use 'region_model' textual label
        row["region"] = oldest.get("region_model", "North")
        row["urban_rural"] = oldest.get("urban_rural", "Urban")
        row["education"] = oldest.get("education", "Bachelor")
        row["marital_status"] = oldest.get("marital_status", "Single")
        row["employment_status"] = oldest.get("employment_status","Employed")
        row["smoker"] = oldest.get("smoker","Never")
        row["alcohol_freq"] = oldest.get("alcohol_freq","None")
        

        # derived fields
        row["annual_premium"] = float(annual_premium)
        row["monthly_premium"] = float(annual_premium/12.0)
        # risk_score: mean of member scores
        raw_mean = float(np.mean([compute_risk_score(m)[0] for m in members]))
        row["risk_score"] = normalize_risk_score(raw_mean)
        # optionally also keep raw in the ordered dict as extra column if you want to inspect
        row["risk_score_raw"] = raw_mean 


        # ensure all expected fields present
        ordered = {k: row.get(k, 0) for k in EXPECTED_FEATURES}
        return pd.DataFrame([ordered])

    X_input = aggregate_for_model(members, details)
    st.write("### Features prepared for model (preview)")
    st.dataframe(X_input.astype(str).T)

    # Call models
    try:
        if hasattr(CLASSIFIER, "predict_proba"):
            prob = float(CLASSIFIER.predict_proba(X_input)[:,1][0])
            st.write(f"Predicted claim probability: **{prob:.4f}**")
        else:
            cls = CLASSIFIER.predict(X_input)[0]
            st.write(f"Predicted claim class: {cls}")
    except Exception as e:
        st.error(f"Classifier error: {e}")

    try:
        sev = float(REGRESSOR.predict(X_input)[0])
        st.write(f"Predicted claim severity (expected): ₹{sev:,.2f}")
    except Exception as e:
        st.error(f"Regressor error: {e}")

    st.success("Done — annual_premium (ex-GST) is passed to the models as 'annual_premium' feature.")

# End of app.py
