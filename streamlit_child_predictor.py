#!/usr/bin/env python3
"""
streamlit_child_predictor.py

Child risk predictor Streamlit app — robust dtype handling with binary flags enforced.

Run:
    streamlit run streamlit_child_predictor.py

Behavior:
- LGA selector is outside the form (updates immediately).
- Settlement selectbox inside the form dynamically shows settlements for chosen LGA (via st.session_state).
- Vaccine flags & Reason flags rendered side-by-side at the end of the form.
"""
from pathlib import Path
import base64
import logging
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
import joblib
import streamlit as st

# ----------------------
# Config & constants
# ----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE = Path.cwd()
ARTIFACTS = BASE / "artifacts"
CHILD_MODEL_FILE = ARTIFACTS / "child_dropoff_model.joblib"
CHILD_FEATURES_FILE = ARTIFACTS / "child_features.joblib"

# Reason-for-ZD flags used in your training
REASON_FLAGS = [
    'busy_caregiver', 'family_problems', 'family_problems_busy_caregiver',
    'family_problems_missed_appointment', 'family_problems_transportation',
    'fear_of_AE', 'financial_resources', 'financial_resources_missed_appointment_busy_caregiver',
    'hp_poor_attitude', 'low_trust', 'missed_appointment_hp_poor_attitude',
    'missed_appointment_uncooperative_husband', 'no_need_felt', 'no_need_felt_busy_caregiver',
    'no_need_felt_financial_resources', 'no_need_felt_low_trust', 'no_permissions',
    'reported_AE', 'sick_child', 'transportation', 'uncooperative_husband',
    'uncooperative_husband_low_trust'
]

st.set_page_config(page_title="Child Risk Predictor", layout="centered")

# ----------------------
# Helpers
# ----------------------
def download_link_df(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)

def safe_numeric(val, default=0.0):
    try:
        if val is None or (isinstance(val, str) and val.strip() == ""):
            return default
        return float(val)
    except Exception:
        return default

def inspect_pipeline_feature_groups(pipeline) -> Tuple[List[str], List[str], List[str]]:
    numeric_cols, categorical_cols, passthrough_cols = [], [], []
    try:
        pre = pipeline.named_steps.get('pre', None)
        if pre is None and hasattr(pipeline, 'steps'):
            first = pipeline.steps[0][1]
            pre = first if hasattr(first, 'transformers_') else None
        if pre is None:
            return [], [], []
        for name, transformer, cols in pre.transformers_:
            if transformer is None or transformer == 'passthrough':
                if isinstance(cols, (list, tuple)):
                    passthrough_cols.extend(list(cols))
            else:
                t = transformer
                if hasattr(t, 'named_steps'):
                    names = " ".join(t.named_steps.keys()).lower()
                    if 'ohe' in names or 'onehot' in names:
                        if isinstance(cols, (list, tuple)):
                            categorical_cols.extend(list(cols))
                    else:
                        if isinstance(cols, (list, tuple)):
                            numeric_cols.extend(list(cols))
                else:
                    if isinstance(cols, (list, tuple)):
                        numeric_cols.extend(list(cols))
        return sorted(set(numeric_cols)), sorted(set(categorical_cols)), sorted(set(passthrough_cols))
    except Exception as e:
        logger.warning(f"Pipeline inspection failed: {e}")
        return [], [], []

@st.cache_resource
def load_artifacts():
    res = {}
    if CHILD_MODEL_FILE.exists() and CHILD_FEATURES_FILE.exists():
        res['model'] = joblib.load(CHILD_MODEL_FILE)
        res['features'] = joblib.load(CHILD_FEATURES_FILE)
    else:
        missing = []
        if not CHILD_MODEL_FILE.exists(): missing.append(str(CHILD_MODEL_FILE))
        if not CHILD_FEATURES_FILE.exists(): missing.append(str(CHILD_FEATURES_FILE))
        st.warning(f"Missing artifact files: {', '.join(missing)}. Place them in ./artifacts/")
    return res

@st.cache_data
def load_local_uniques(max_unique=200) -> Dict[str, List[str]]:
    path = BASE / "zerodose.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, dtype=str, low_memory=False)
        uniques = {}
        for c in df.columns:
            vals = df[c].dropna().unique().tolist()
            if 1 <= len(vals) <= max_unique:
                uniques[c] = sorted([str(v) for v in vals])
        return uniques
    except Exception:
        return {}

@st.cache_data
def load_lga_settlements() -> Dict[str, List[str]]:
    path = BASE / "zerodose.csv"
    mapping: Dict[str, List[str]] = {}
    if not path.exists():
        return mapping
    try:
        df = pd.read_csv(path, dtype=str, low_memory=False)
        if 'LGA' in df.columns and 'Settlement' in df.columns:
            df['LGA_norm'] = df['LGA'].astype(str).str.strip()
            df['Settlement_norm'] = df['Settlement'].astype(str).str.strip()
            grp = df.groupby('LGA_norm')['Settlement_norm'].unique().to_dict()
            for k, v in grp.items():
                cleaned = sorted([s for s in map(str, v) if s and s.lower() != 'nan'])
                mapping[k] = cleaned
    except Exception as e:
        logger.warning(f"Failed to build LGA->Settlement mapping: {e}")
    return mapping

# ----------------------
# Load model + introspect
# ----------------------
local_uniques = load_local_uniques()
lga_settlement_map = load_lga_settlements()
art = load_artifacts()
model = art.get('model')
child_features = art.get('features')

if model is None or child_features is None:
    st.title("Individual Child Risk Predictor")
    st.error("Child model or feature list not found in ./artifacts/. Place `child_dropoff_model.joblib` and `child_features.joblib` there.")
    st.stop()

pipe_numeric, pipe_categorical, pipe_passthrough = inspect_pipeline_feature_groups(model)

# Build sets for numeric/categorical
numeric_features = set(pipe_numeric)
for c in pipe_passthrough:
    if c.startswith('lga_vacc_') or c in REASON_FLAGS or c.startswith('rate_') or c in ['Distance to HF', 'estimated_age_months']:
        numeric_features.add(c)
for f in child_features:
    if f in ['Distance to HF', 'estimated_age_months'] or f.startswith('lga_vacc_') or f.startswith('rate_') or f in REASON_FLAGS:
        numeric_features.add(f)
numeric_features = sorted(numeric_features)

st.title("Individual Child Risk Predictor")

# NOTE: single prediction only
batch_choices: Dict[str, List[str]] = {}

def prepare_single_from_inputs(single_inputs: Dict[str, Any]) -> pd.DataFrame:
    row = {}
    for f in child_features:
        if f in single_inputs:
            row[f] = single_inputs[f]
        else:
            row[f] = 0.0 if f in numeric_features else "missing"
    X = pd.DataFrame([row])
    for c in numeric_features:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0.0).astype(float)
    for c in X.columns:
        if c not in numeric_features:
            X[c] = X[c].fillna('missing').astype(str)
    return X

# ----------------------
# LGA selectbox OUTSIDE the form (so it updates immediately)
# ----------------------
# Determine LGA choices (from zerodose mapping or local uniques)
lga_choices = []
if lga_settlement_map:
    lga_choices = sorted(list(lga_settlement_map.keys()))
elif 'LGA' in local_uniques:
    lga_choices = local_uniques.get('LGA', [])
# ensure "missing" present
if "missing" not in lga_choices:
    lga_choices = ["missing"] + lga_choices

st.markdown("### Select LGA (choosing an LGA will filter Settlement choices inside the form)")


# ----------------------
# UI - Single prediction form (Settlement reads selected_lga live from session_state)
# ----------------------
st.subheader("Single child prediction")
with st.form("single_form"):
    single_inputs: Dict[str, Any] = {}

    vaccine_features = [f for f in child_features if f.startswith('lga_vacc_')]
    reason_features = [f for f in child_features if f in REASON_FLAGS]
    vaccine_features = [f for f in vaccine_features if f is not None]
    reason_features = [f for f in reason_features if f is not None]

    remaining_features = [f for f in child_features if f not in set(vaccine_features + reason_features)]
    if remaining_features:
        st.markdown("**Other features (including Settlement if present)**")

    # fallback settlements (if zero mapping)
    fallback_settlements = local_uniques.get('Settlement', [])

    for f in remaining_features:
        if f == 'LGA':
            # put the selected_lga value into inputs (we keep the actual selectbox outside the form)
            selected_lga = st.selectbox("LGA (select)", options=lga_choices, index=0, key="selected_lga")
            single_inputs['LGA'] = st.session_state.get("selected_lga", "missing")
            st.write(f"Selected LGA (locked for this prediction): **{single_inputs['LGA']}**")
        elif f == 'Settlement':
            sel_lga = st.session_state.get("selected_lga", None)
            settlement_options: List[str] = []
            if sel_lga and sel_lga != "missing":
                settlement_options = lga_settlement_map.get(sel_lga, [])
            if not settlement_options:
                settlement_options = fallback_settlements.copy()
            if "missing" not in settlement_options:
                settlement_options = ["missing"] + settlement_options
            # default index logic
            default_idx = 0
            existing = st.session_state.get("Settlement", None)
            if existing and existing in settlement_options:
                default_idx = settlement_options.index(existing)
            try:
                single_inputs['Settlement'] = st.selectbox("Settlement (select)", options=settlement_options, index=default_idx, key="form_settlement")
            except Exception:
                single_inputs['Settlement'] = st.text_input("Settlement (text)", value="missing", key="form_settlement_text")
        else:
            if f in numeric_features:
                default_val = 0.0
                try:
                    single_inputs[f] = st.number_input(f"{f} (numeric)", min_value=0.0, value=float(default_val), step=1.0, format="%.2f")
                except Exception:
                    val = st.text_input(f"{f} (numeric)", value=str(default_val))
                    single_inputs[f] = safe_numeric(val, default=default_val)
            else:
                choices: Optional[List[str]] = None
                if f in batch_choices:
                    choices = ["missing"] + batch_choices[f]
                elif f in local_uniques:
                    choices = ["missing"] + local_uniques[f]
                else:
                    if f.lower() == 'gender':
                        choices = ["missing", "Male", "Female", "Other"]
                    elif f.lower() in ['woman or child', 'woman_or_child', 'womanorchild']:
                        choices = ["missing", "Woman", "Child"]
                    else:
                        choices = ["missing"]
                try:
                    single_inputs[f] = st.selectbox(f"{f} (categorical)", options=choices, index=0)
                except Exception:
                    single_inputs[f] = "missing"

    # Vaccine & Reason flags at the end, side-by-side
    if vaccine_features or reason_features:
        col_vac, col_reason = st.columns([1, 1])
        with col_vac:
            st.markdown("**Vaccine flags**")
            for f in vaccine_features:
                try:
                    val = st.checkbox(f"{f}", value=False)
                except Exception:
                    val = False
                single_inputs[f] = 1 if val else 0
        with col_reason:
            st.markdown("**Reason flags**")
            for f in reason_features:
                try:
                    val = st.checkbox(f"{f}", value=False)
                except Exception:
                    val = False
                single_inputs[f] = 1 if val else 0

    submitted = st.form_submit_button("Predict")

# ----------------------
# On submit: prepare, predict, show results
# ----------------------
if submitted:
    try:
        X_single = prepare_single_from_inputs(single_inputs)

        # Predict
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X_single)[:,1][0])
        else:
            prob = None
        pred = int(model.predict(X_single)[0])
        st.markdown("### Prediction")
        if prob is not None:
            st.metric("Dropoff probability", f"{prob:.3f}")
        st.write("Predicted class:", "DROPOFF (likely)" if pred == 1 else "LOW DROPOFF RISK")
        if prob is not None:
            if prob >= 0.94:
                st.warning("High risk — recommend immediate follow-up/outreach.")
            elif prob >= 0.6:
                st.info("Moderate risk — consider targeted outreach or reminder.")
            else:
                st.success("Low risk — routine monitoring.")
        out = X_single.copy()
        out['pred_prob'] = prob
        out['pred_class'] = pred
        download_link_df(out, "single_child_prediction.csv", "Download prediction (CSV)")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        logger.exception(e)


