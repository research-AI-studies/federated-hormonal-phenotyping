"""Map a raw registry export to the analysis feature set.

The raw export uses coded categorical fields documented in an external
codebook. This module recodes those fields into the ten hormonal/lifestyle
clustering inputs, the EORTC patient-reported outcome scales, and the
adjustment covariates used by the downstream models. No patient records are
embedded here; only the schema-level recoding logic, which mirrors the public
codebook crosswalk.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Ten primary clustering inputs (birth_number excluded for high missingness).
CLUSTERING_INPUTS = [
    "age", "bmi", "menarche", "pregnancies", "menopausal",
    "contraceptive", "alcohol", "smoking", "bust", "cup",
]

# EORTC scales used for phenotype-to-symptom mapping.
EORTC_PRIMARY = ["fa", "sl", "ef", "brbi", "brsef", "dy"]
EORTC_CONDITIONAL = ["brsee", "brhl"]
EORTC_ALL = EORTC_PRIMARY + EORTC_CONDITIONAL

COVARIATES = ["age", "diagnosis_group", "marital_status", "pre_op"]


def _recode_smoking(v):
    # raw: 0=No(never), 1=Yes(current), 2=Ex-smoker -> ordinal never<former<current
    return {0: 0, 2: 1, 1: 2}.get(v, np.nan)


def _recode_contraceptive(v):
    # 0 none; {1,2,4,6} hormonal-systemic; {3,5} hormonal-local;
    # 7 non-hormonal mechanical; 888 other
    if v == 0:
        return "none"
    if v in (1, 2, 4, 6):
        return "hormonal_systemic"
    if v in (3, 5):
        return "hormonal_local"
    if v == 7:
        return "mechanical"
    if v == 888:
        return "other"
    return np.nan


def _diagnosis_group(v):
    # 1 breast cancer, 2 DCIS -> malignant; 3 fibroadenoma, 4 other -> benign
    return {1: "breast_cancer", 2: "dcis", 3: "fibroadenoma", 4: "other_benign"}.get(v, np.nan)


@dataclass
class MappedCohort:
    frame: pd.DataFrame
    inputs: list = field(default_factory=lambda: list(CLUSTERING_INPUTS))
    eortc: list = field(default_factory=lambda: list(EORTC_ALL))


def map_raw(df_raw: pd.DataFrame, menarche_bounds=(8, 18)) -> MappedCohort:
    """Return a tidy analysis frame from the raw registry columns."""
    d = pd.DataFrame(index=df_raw.index)

    d["age"] = pd.to_numeric(df_raw["age"], errors="coerce")
    d["bmi"] = pd.to_numeric(df_raw["bmi"], errors="coerce")

    men = pd.to_numeric(df_raw["menstruation_firsttime_age"], errors="coerce")
    lo, hi = menarche_bounds
    d["menarche"] = men.where((men >= lo) & (men <= hi))

    d["pregnancies"] = pd.to_numeric(df_raw["pregnancy_number"], errors="coerce")
    d["menopausal"] = pd.to_numeric(df_raw["menopause_yn"], errors="coerce")
    d["contraceptive"] = df_raw["contraceptive_kind"].map(_recode_contraceptive)
    d["alcohol"] = pd.to_numeric(df_raw["alcohol"], errors="coerce")
    d["smoking"] = df_raw["smokingstatus"].map(_recode_smoking)
    d["bust"] = pd.to_numeric(df_raw["bust"], errors="coerce")
    d["cup"] = pd.to_numeric(df_raw["cupsize"], errors="coerce")

    # EORTC scales (already 0-100 decimals)
    for s in EORTC_ALL:
        d[s] = pd.to_numeric(df_raw[s], errors="coerce")

    # covariates / strata
    d["diagnosis_code"] = pd.to_numeric(df_raw["diagnosis"], errors="coerce")
    d["diagnosis_group"] = df_raw["diagnosis"].map(_diagnosis_group)
    d["marital_status"] = pd.to_numeric(df_raw["marital_status"], errors="coerce")
    d["pre_op"] = pd.to_numeric(df_raw["pre_op"], errors="coerce")

    return MappedCohort(frame=d)


def analytic_mask(d: pd.DataFrame) -> pd.Series:
    """Analytic cohort: valid diagnosis and observed age and BMI."""
    return d["diagnosis_group"].notna() & d["age"].notna() & d["bmi"].notna()
