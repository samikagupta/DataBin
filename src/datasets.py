"""
Dataset registry. Each entry is one (file, bin column, target) combination.

Every dataset is loaded the same way:
  - drop the listed columns (identifiers, leaking alternative targets)
  - apply target_fix if given, then drop rows missing the bin column or target
  - drop numeric feature columns missing in more than MAX_MISSING of rows
  - encode the target and every non-numeric or boolean feature as category
    codes, in sorted order, as the notebooks do with .astype('category').cat.codes;
    a missing category value becomes its own code (-1)
  - fill remaining missing numeric values with the column median

For binary targets the positive class is code 1; `positive` says what that is.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

# Every file is in data/, gzipped (only the 15 UCI files are there so far)
DATA = BUSINESS = UCI = Path(__file__).resolve().parent / "data"

MAX_MISSING = 0.5


@dataclass(frozen=True)
class Dataset:
    file: Path
    bin_col: str
    target: str
    positive: str = ""    # meaning of class code 1 (binary targets)
    drop: tuple = ()
    default: bool = True   # run when neither --datasets nor --group is given
    group: str = "paper"   # paper | business | uci; select with --group
    target_fix: Optional[Callable] = None  # applied to the raw target column


def _strip_dot(s):
    """UCI Adult's test rows end their label with '.' ('>50K.')."""
    return s.str.rstrip(".")


REGISTRY = {
    # data/ -- used in the paper so far
    "adult": Dataset(DATA / "adult_process.csv", "age", "income", positive=">50K"),
    "bank": Dataset(DATA / "bank.csv", "age", "y", positive="subscribed"),
    "law": Dataset(DATA / "law_dataset.csv", "lsat", "pass_bar", positive="passed the bar"),
    "rice": Dataset(DATA / "rice.csv", "Major_Axis_Length", "Class", positive="Osmancik"),
    "german": Dataset(DATA / "german_processed.csv", "amount", "credit_risk", positive="good risk"),
    "beans": Dataset(DATA / "beans.csv", "Perimeter", "Class"),
    "dropout": Dataset(DATA / "dropout.csv", "gradetwo", "Target"),
    "covtype": Dataset(DATA / "covtype.csv", "Elevation", "Cover_Type", default=False),  # off for now

    # business data/
    "credit_limit": Dataset(BUSINESS / "01_credit_and_lending/default_of_credit_card_clients.csv",
                            "LIMIT_BAL", "default_payment_next_month", positive="defaulted",
                            group="business"),
    "credit_age": Dataset(BUSINESS / "01_credit_and_lending/default_of_credit_card_clients.csv",
                          "AGE", "default_payment_next_month", positive="defaulted",
                          group="business"),

    # business data/ candidates, off until approved: pass --datasets or set default=True
    "shoppers": Dataset(BUSINESS / "02_ecommerce_web_behavior/online_shoppers_intention.csv",
                        "ProductRelated_Duration", "Revenue", positive="purchased",
                        default=False, group="business"),
    "polish3": Dataset(BUSINESS / "04_firm_financials/polish_bankruptcy_year3.csv",
                       "Attr29", "bankrupt", positive="bankrupt",  # Attr29 = log total assets
                       drop=("Attr37", "Attr21"), default=False, group="business"),
    "shill": Dataset(BUSINESS / "02_ecommerce_web_behavior/shill_bidding.csv",
                     "Bidder_Tendency", "Class", positive="shill bidder",
                     drop=("Record_ID", "Auction_ID", "Bidder_ID"), default=False, group="business"),
    "churn": Dataset(BUSINESS / "05_marketing_and_churn/iranian_churn.csv",
                     "Customer Value", "Churn", positive="churned", default=False, group="business"),

    # uci_viability/datasets -- targets from its manifest.csv, age binned wherever the
    # dataset has it. Smallest first, so `--group uci` finishes the quick ones early.
    "uci863_maternal": Dataset(UCI / "863_maternal_health_risk.csv", "Age", "RiskLevel",
                               default=False, group="uci"),
    "uci503_hcv": Dataset(UCI / "503_hepatitis_c_virus_hcv_for_egyptian_patients.csv",
                          "Age ", "Baselinehistological staging", default=False, group="uci"),
    "uci30_contraceptive": Dataset(UCI / "30_contraceptive_method_choice.csv", "wife_age",
                                   "contraceptive_method", default=False, group="uci"),
    "uci579_myocardial": Dataset(UCI / "579_myocardial_infarction_complications.csv", "AGE", "LET_IS",
                                 drop=("ID", "FIBR_PREDS", "PREDS_TAH", "JELUD_TAH", "FIBR_JELUD",
                                       "A_V_BLOK", "OTEK_LANC", "RAZRIV", "DRESSLER", "ZSN",
                                       "REC_IM", "P_IM_STEN"),  # the other 11 outcome columns
                                 default=False, group="uci"),
    "uci544_obesity": Dataset(UCI / "544_estimation_of_obesity_levels_based_on_eating_habit.csv",
                              "Age", "NObeyesdad", default=False, group="uci"),
    "uci890_aids": Dataset(UCI / "890_aids_clinical_trials_group_study_175.csv", "age", "cid",
                           positive="failure event", drop=("pidnum",), default=False, group="uci"),
    "uci887_nhanes": Dataset(UCI / "887_national_health_and_nutrition_health_survey_2013_2.csv",
                             "BMXBMI", "age_group", positive="Senior",
                             drop=("SEQN", "RIDAGEYR"),  # age_group is RIDAGEYR >= 65
                             default=False, group="uci"),
    "uci272_skillcraft": Dataset(UCI / "272_skillcraft1_master_table_dataset.csv", "Age",
                                 "LeagueIndex",  # manifest has no target; UCI's class variable
                                 drop=("GameID",), default=False, group="uci"),
    "uci697_dropout": Dataset(UCI / "697_predict_students_dropout_and_academic_success.csv",
                              "Age at enrollment", "Target", default=False, group="uci"),
    "uci852_gendergap": Dataset(UCI / "852_gender_gap_in_spanish_wp.csv", "NEds", "gender",
                                drop=("C_api", "C_man"),  # alternative gender labels
                                default=False, group="uci"),
    "uci350_credit": Dataset(UCI / "350_default_of_credit_card_clients.csv", "X5", "Y",  # X5 = age
                             positive="defaulted", drop=("ID",), default=False, group="uci"),
    "uci222_bank": Dataset(UCI / "222_bank_marketing.csv", "age", "y", positive="subscribed",
                           default=False, group="uci"),
    "uci2_adult": Dataset(UCI / "2_adult.csv", "age", "income", positive=">50K",
                          target_fix=_strip_dot, default=False, group="uci"),
    "uci827_sepsis": Dataset(UCI / "827_sepsis_survival_minimal_clinical_records.csv", "age_years",
                             "hospital_outcome_1alive_0dead", positive="survived",
                             default=False, group="uci"),
    "uci117_kdd": Dataset(UCI / "117_census_income_kdd.csv", "AAGE", "income", positive="50000+",
                          drop=("split", "MARSUPWRT"),  # split marker; instance weight
                          default=False, group="uci"),
}

# Files in uci_viability/datasets that are not registered, and why.
NOT_RUN = {
    "20_census_income.csv": "byte-identical to 2_adult.csv (same md5); uci2_adult covers it",
    "189_parkinsons_telemonitoring.csv": "regression targets only (motor_UPDRS, total_UPDRS)",
    "211_communities_and_crime_unnormalized.csv": "regression targets only (crime counts and rates)",
    "127_ipums_census_database.csv": "no target column in the manifest",
    "102_thyroid_disease.csv": "removed from consideration (2026-09-15): 5 of its 32 diagnosis "
                               "classes have one row, so stratified sampling and xgb fail; "
                               "earlier results in results/excluded/",
}


def default_names():
    return [name for name, spec in REGISTRY.items() if spec.default]


def group_names(group):
    return [name for name, spec in REGISTRY.items() if spec.group == group]


@lru_cache(maxsize=None)
def load(name):
    """
    Returns (data, spec, n_classes). Cached, so every config for a dataset
    shares one frame: callers must not modify it.
    """
    spec = REGISTRY[name]
    df = pd.read_csv(f"{spec.file}.gz", low_memory=False).drop(columns=list(spec.drop))
    if spec.target_fix is not None:
        df[spec.target] = spec.target_fix(df[spec.target])
    df = df.dropna(subset=[spec.bin_col, spec.target]).reset_index(drop=True)
    sparse = [c for c in df.columns
              if c not in (spec.bin_col, spec.target)
              and pd.api.types.is_numeric_dtype(df[c]) and df[c].isna().mean() > MAX_MISSING]
    df = df.drop(columns=sparse)
    for c in df.columns:
        if c == spec.target or df[c].dtype == bool or not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("category").cat.codes
    df = df.fillna(df.median(numeric_only=True))
    return df, spec, int(df[spec.target].nunique())


# Integer codes of nominal attributes: no order to cut along, so multi-column
# binning skips them (all_columns.CATEGORICAL)
CATEGORICAL = {
    "uci697_dropout": {"Nacionality", "Mother's qualification", "Father's qualification",
                       "Mother's occupation", "Father's occupation"},
    "uci2_adult": {"native-country"},
    "uci117_kdd": {"ADTINK", "ADTOCC", "AMJIND", "GRINST", "HHDFMX",
                   "PEFNTVTY", "PEMNTVTY", "PENATVTY"},
}
