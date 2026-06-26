
from __future__ import annotations

DATASET_LABELS = {
    "icgt_cacao": "ICGT cacao",
    "agrosavia_matina": "Agrosavia cacao (Matina reference)",
    "agrosavia_criollo": "Agrosavia cacao (Criollo reference)",
    "coffee_int": "Coffea canephora Intermediate",
    "coffee_prec": "Coffea canephora Precocious",
}

VALIDATION_LABELS = {
    "random_five_fold": "Random five-fold",
    "cluster_out": "Cluster-out",
}

MODEL_LABELS = {
    "mean": "Mean baseline",
    "ridge": "Ridge regression",
    "elastic_net": "Elastic net",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
}

PANEL_STRATEGY_LABELS = {
    "random": "Random SNPs",
    "high_maf": "High-MAF SNPs",
    "anova_f": "ANOVA-F selected SNPs (nested)",
    "full": "Full marker set",
}

TRAIT_LABELS = {
    "FL_SEP_LEN": "Flower sepal length",
    "FL_LIG_WID": "Flower ligule width",
    "FL_OV_NUM": "Flower ovule number",
    "FL_STY_LEN": "Flower style length",
    "FR_POL": "Pod length",
    "FR_POW": "Pod width",
    "PODL_W": "Pod length:width ratio",
    "FR_WABW": "Wet bean mass per pod",
    "FR_BEN": "Seed number",
    "FR_BW": "Seed/cotyledon mass",
    "FR_BL": "Seed length",
    "FR_BEW": "Seed width",
    "BENL_W": "Seed length:width ratio",
    "PodIndex": "Pod index",
    "Total pods": "Total pods",
    "Healthy pods": "Healthy pods",
    "AUDPC pods infected by FPRD": "Frosty pod rot AUDPC",
    "AUDPC flower cushion broom infected by WBD": "Witches' broom flower-cushion AUDPC",
    "AUDPC deformed branches infected by WBD": "Witches' broom branch AUDPC",
    "production": "Coffee production",
    "rust": "Leaf rust incidence",
    "green": "Green bean yield",
}

def label_value(value: object, mapping: dict[str, str]) -> object:
    if value is None:
        return value
    text = str(value)
    return mapping.get(text, text)
