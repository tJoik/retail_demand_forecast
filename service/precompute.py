"""
Pre-compute inference feature matrices for all three horizons.
Run once (or after data update):  python service/precompute.py
Output: service/precomputed/X_{week,month,3month}.csv + period_*.txt
"""
from __future__ import annotations

import os
import sys
from datetime import timedelta
from pathlib import Path

os.chdir(Path(__file__).parent.parent)
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

BASE = Path(__file__).parent.parent
DATA = BASE / "features" / "features_one_week.csv"
NOM  = BASE / "features" / "nomenclature_with_supplier.csv"
OUT  = Path(__file__).parent / "precomputed"


# ─────────────────────────── helpers ─────────────────────────────

def _fix(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


# ─────────────────────────── raw data ────────────────────────────

def load_raw() -> pd.DataFrame:
    print("  загрузка features_one_week.csv...", flush=True)
    df = pd.read_csv(DATA, sep=";")
    df = df[df["ОстатокНачалоНедели"] % 1 == 0]
    df = df[df["Количество"] >= 0]
    df = df[df["Количество"] <= 350]
    df["НеделяНачало"] = pd.to_datetime(df["НеделяНачало"])

    print("  TF-IDF по номенклатуре...", flush=True)
    uniq  = df.drop_duplicates("Номенклатура").copy()
    tfidf = TfidfVectorizer()
    tfidf.fit(uniq["Номенклатура"])
    vals  = []
    for s in uniq["Номенклатура"]:
        m = pd.DataFrame(tfidf.transform([s]).T.todense())
        m = m[m.values > 0]
        vals.append(float(m.mean().iloc[0]))
    uniq["tfidf_mean"] = pd.Series(vals, index=uniq.index)
    df = df.merge(uniq[["Номенклатура", "tfidf_mean"]], on="Номенклатура", how="left")
    del uniq

    nom = pd.read_csv(NOM)
    nom.loc[(nom["supplier"] == "НК") | nom["supplier"].isna(), "supplier"] = "Нет значения"
    df = df.merge(nom, how="left", left_on="Номенклатура", right_on="product").drop(columns=["product"])
    df["supplier"] = df["supplier"].fillna("Нет значения")
    del nom

    s = pd.to_datetime(df["ДатаПоследнегоПоступления"], errors="coerce")
    df["МесяцПоследнегоПоступления"] = s.dt.month
    df["ГодПоследнегоПоступления"]   = s.dt.year
    df.drop(columns=["ДатаПоследнегоПоступления"], inplace=True)
    return df


def climate_by_month(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(df["НеделяНачало"].dt.month).agg(
        temp_mean_week   =("temp_mean_week",   "mean"),
        precip_sum_week  =("precip_sum_week",  "mean"),
        temp_max_week    =("temp_max_week",    "mean"),
        temp_min_week    =("temp_min_week",    "mean"),
        days_off_in_week =("days_off_in_week", "mean"),
    )


def _climate_row(climate: pd.DataFrame, month: int) -> pd.Series:
    return climate.loc[month] if month in climate.index else climate.mean()


# ─────────────────────────── WEEKLY ──────────────────────────────

def _ts_w(sells, dt, minus, periods):
    cols = pd.date_range(pd.to_datetime(dt) - timedelta(weeks=minus), periods=periods, freq="W-MON")
    return sells.reindex(columns=cols, fill_value=0)


def _feats_week(sells, t):
    X = {}
    for i in [2, 4, 8, 12]:
        tmp = _ts_w(sells, t, i, i)
        X[f"diff_{i}_mean"]  = tmp.diff(axis=1).mean(axis=1).values
        X[f"mean_{i}_decay"] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
        X[f"mean_{i}"]       = tmp.mean(axis=1).values
        X[f"median_{i}"]     = tmp.median(axis=1).values
        X[f"min_{i}"]        = tmp.min(axis=1).values
        X[f"max_{i}"]        = tmp.max(axis=1).values
        X[f"std_{i}"]        = tmp.std(axis=1).values
    for i in [2, 4, 8, 12]:
        tmp = _ts_w(sells, t - timedelta(weeks=1), i, i)
        X[f"diff_{i}_mean_2"]  = tmp.diff(axis=1).mean(axis=1).values
        X[f"mean_{i}_decay_2"] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
        X[f"mean_{i}_2"]       = tmp.mean(axis=1).values
        X[f"median_{i}_2"]     = tmp.median(axis=1).values
        X[f"min_{i}_2"]        = tmp.min(axis=1).values
        X[f"max_{i}_2"]        = tmp.max(axis=1).values
        X[f"std_{i}_2"]        = tmp.std(axis=1).values
    return pd.DataFrame(X)


def build_weekly(df: pd.DataFrame, climate: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    df2 = df.copy()
    df2["КодТовара"] = _fix(df2["КодТовара"])
    df2["НеделяНачало"] = pd.to_datetime(df2["НеделяНачало"]).dt.tz_localize(None).dt.normalize()

    sells = (
        df2[["НеделяНачало", "Количество", "КодТовара"]].copy()
        .set_index(["КодТовара", "НеделяНачало"])[["Количество"]]
        .unstack(-1).fillna(0).sort_index().sort_index(axis=1)
    )
    sells.columns = pd.to_datetime(sells.columns.get_level_values(1)).tz_localize(None)
    sells.index   = _fix(sells.index)

    t = df2["НеделяНачало"].max() + timedelta(weeks=1)
    period_label = t.strftime("неделя с %d.%m.%Y")

    feats = _feats_week(sells, t)
    feats.index      = sells.index
    feats.index.name = "КодТовара"
    feats = feats.reset_index()
    feats["КодТовара"] = _fix(feats["КодТовара"])

    last = df2.sort_values("НеделяНачало").groupby("КодТовара").last().reset_index()
    last["НеделяНачало"] = t
    w = _climate_row(climate, t.month)
    last["temp_mean_week"]   = w["temp_mean_week"]
    last["precip_sum_week"]  = w["precip_sum_week"]
    last["temp_max_week"]    = w["temp_max_week"]
    last["temp_min_week"]    = w["temp_min_week"]
    last["days_off_in_week"] = w["days_off_in_week"]

    df_inf = last.merge(feats, on="КодТовара", how="inner")
    df_inf["НеделяНачало"] = t

    df_inf = df_inf[df_inf["Папка1"] != "разобрать"]
    df_inf = df_inf[df_inf["ЕдиницаИзмерения"] == "шт"].drop("ЕдиницаИзмерения", axis=1)
    df_inf = df_inf[df_inf["ТоварнаяКатегория"] == "Штучный товар"].drop("ТоварнаяКатегория", axis=1)
    bad = df_inf.loc[
        (df_inf[["ОстатокНачалоНедели", "Количество"]].astype(float) % 1 != 0).any(axis=1),
        "КодТовара",
    ].unique()
    df_inf = df_inf[~df_inf["КодТовара"].isin(bad)]
    df_inf = df_inf[df_inf["Номенклатура"] != "1"]

    X = df_inf.drop(columns=["Количество"], errors="ignore").copy()
    X["ТекущийМесяц"] = t.month
    X["ТекущийГод"]   = t.year
    X = X.drop(columns=["НеделяНачало"], errors="ignore")
    return X, period_label


# ─────────────────────────── MONTHLY ─────────────────────────────

def _ts_m(sells, dt, minus, periods):
    cols = pd.date_range(pd.to_datetime(dt) - pd.DateOffset(months=minus), periods=periods, freq="MS")
    return sells.reindex(columns=cols, fill_value=0)


def _feats_month(sells, t):
    X = {}
    for i in [2, 4, 6, 12]:
        tmp = _ts_m(sells, t, i, i)
        X[f"diff_{i}_mean"]  = tmp.diff(axis=1).mean(axis=1).values
        X[f"mean_{i}_decay"] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
        X[f"mean_{i}"]       = tmp.mean(axis=1).values
        X[f"median_{i}"]     = tmp.median(axis=1).values
        X[f"min_{i}"]        = tmp.min(axis=1).values
        X[f"max_{i}"]        = tmp.max(axis=1).values
        X[f"std_{i}"]        = tmp.std(axis=1).values
    for i in [2, 4, 6, 12]:
        tmp = _ts_m(sells, pd.to_datetime(t) - pd.DateOffset(months=1), i, i)
        X[f"diff_{i}_mean_2"]  = tmp.diff(axis=1).mean(axis=1).values
        X[f"mean_{i}_decay_2"] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
        X[f"mean_{i}_2"]       = tmp.mean(axis=1).values
        X[f"median_{i}_2"]     = tmp.median(axis=1).values
        X[f"min_{i}_2"]        = tmp.min(axis=1).values
        X[f"max_{i}_2"]        = tmp.max(axis=1).values
        X[f"std_{i}_2"]        = tmp.std(axis=1).values
    return pd.DataFrame(X)


def build_monthly(df: pd.DataFrame, climate: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    df = df.copy()
    df["МесяцНачало"] = df["НеделяНачало"].dt.to_period("M").dt.start_time
    dm = df.groupby(["КодТовара", "МесяцНачало"]).agg({
        "Номенклатура": "last", "Папка1": "last", "Папка2": "last",
        "Поставщик": "last", "ЕдиницаИзмерения": "last", "ТоварнаяКатегория": "last",
        "Количество": "sum", "Розничная30%": "last", "ЗакупочнаяЦена": "last",
        "ОстатокНачалоНедели": "first",
        "temp_mean_week": "mean", "precip_sum_week": "sum",
        "temp_max_week": "max", "temp_min_week": "min",
        "НДС": "last", "days_off_in_week": "sum",
        "tfidf_mean": "last", "supplier": "last",
        "МесяцПоследнегоПоступления": "last", "ГодПоследнегоПоступления": "last",
    }).reset_index().rename(columns={
        "ОстатокНачалоНедели": "ОстатокНачалоМесяца",
        "temp_mean_week": "temp_mean_month", "precip_sum_week": "precip_sum_month",
        "temp_max_week": "temp_max_month", "temp_min_week": "temp_min_month",
        "days_off_in_week": "days_off_month",
    })
    dm["КодТовара"]   = _fix(dm["КодТовара"])
    dm["МесяцНачало"] = pd.to_datetime(dm["МесяцНачало"]).dt.tz_localize(None).dt.normalize()

    sells = (
        dm.set_index(["КодТовара", "МесяцНачало"])[["Количество"]]
        .unstack(-1).fillna(0).sort_index().sort_index(axis=1)
    )
    sells.columns = pd.to_datetime(sells.columns.get_level_values(1)).tz_localize(None).normalize()
    sells.index   = _fix(sells.index)

    t = dm["МесяцНачало"].max() + pd.DateOffset(months=1)
    t = pd.Timestamp(t.year, t.month, 1)
    period_label = t.strftime("%B %Y")

    feats = _feats_month(sells, t)
    feats.index      = sells.index
    feats.index.name = "КодТовара"
    feats = feats.reset_index()
    feats["КодТовара"] = _fix(feats["КодТовара"])

    last = dm.sort_values("МесяцНачало").groupby("КодТовара").last().reset_index()
    last["МесяцНачало"] = t
    w = _climate_row(climate, t.month)
    last["temp_mean_month"]  = w["temp_mean_week"]
    last["precip_sum_month"] = w["precip_sum_week"]
    last["temp_max_month"]   = w["temp_max_week"]
    last["temp_min_month"]   = w["temp_min_week"]
    last["days_off_month"]   = w["days_off_in_week"]

    df_inf = last.merge(feats, on="КодТовара", how="inner")
    df_inf["МесяцНачало"] = t

    df_inf = df_inf[df_inf["Папка1"] != "разобрать"]
    df_inf = df_inf[df_inf["ЕдиницаИзмерения"] == "шт"].drop("ЕдиницаИзмерения", axis=1)
    df_inf = df_inf[df_inf["ТоварнаяКатегория"] == "Штучный товар"].drop("ТоварнаяКатегория", axis=1)
    bad = df_inf.loc[
        (df_inf[["ОстатокНачалоМесяца", "Количество"]].astype(float) % 1 != 0).any(axis=1),
        "КодТовара",
    ].unique()
    df_inf = df_inf[~df_inf["КодТовара"].isin(bad)]
    df_inf = df_inf[df_inf["Номенклатура"] != "1"]

    X = df_inf.drop(columns=["Количество"], errors="ignore").copy()
    X["ТекущийМесяц"] = t.month
    X["ТекущийГод"]   = t.year
    X = X.drop(columns=["МесяцНачало"], errors="ignore")
    return X, period_label


# ─────────────────────────── QUARTERLY ───────────────────────────

def _ts_q(sells, dt, minus, periods):
    cols = pd.date_range(pd.to_datetime(dt) - pd.DateOffset(months=3 * minus), periods=periods, freq="QS")
    return sells.reindex(columns=cols, fill_value=0)


def _feats_quarter(sells, t):
    X = {}
    for i in [1, 2, 4, 8]:
        tmp = _ts_q(sells, t, i, i)
        X[f"diff_{i}q_mean"]  = tmp.diff(axis=1).mean(axis=1).values
        X[f"mean_{i}q_decay"] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
        X[f"mean_{i}q"]       = tmp.mean(axis=1).values
        X[f"median_{i}q"]     = tmp.median(axis=1).values
        X[f"min_{i}q"]        = tmp.min(axis=1).values
        X[f"max_{i}q"]        = tmp.max(axis=1).values
        X[f"std_{i}q"]        = tmp.std(axis=1).values
    for i in [1, 2, 4, 8]:
        tmp = _ts_q(sells, pd.to_datetime(t) - pd.DateOffset(months=3), i, i)
        X[f"diff_{i}q_mean_2"]  = tmp.diff(axis=1).mean(axis=1).values
        X[f"mean_{i}q_decay_2"] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
        X[f"mean_{i}q_2"]       = tmp.mean(axis=1).values
        X[f"median_{i}q_2"]     = tmp.median(axis=1).values
        X[f"min_{i}q_2"]        = tmp.min(axis=1).values
        X[f"max_{i}q_2"]        = tmp.max(axis=1).values
        X[f"std_{i}q_2"]        = tmp.std(axis=1).values
    s1 = _ts_q(sells, t, 4, 1).values.flatten()
    s2 = _ts_q(sells, t, 8, 1).values.flatten()
    X["sales_1y_ago"] = s1
    X["sales_2y_ago"] = s2
    X["yoy_ratio"]    = np.where(s2 > 0, s1 / (s2 + 1e-6), 0.0)
    return pd.DataFrame(X)


def build_quarterly(df: pd.DataFrame, climate: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    df = df.copy()
    df["КварталНачало"] = df["НеделяНачало"].dt.to_period("Q").dt.start_time
    dq = df.groupby(["КодТовара", "КварталНачало"]).agg({
        "Номенклатура": "last", "Папка1": "last", "Папка2": "last",
        "Поставщик": "last", "ЕдиницаИзмерения": "last", "ТоварнаяКатегория": "last",
        "Количество": "sum", "Розничная30%": "last", "ЗакупочнаяЦена": "last",
        "ОстатокНачалоНедели": "first",
        "temp_mean_week": "mean", "precip_sum_week": "sum",
        "temp_max_week": "max", "temp_min_week": "min",
        "НДС": "last", "days_off_in_week": "sum",
        "tfidf_mean": "last", "supplier": "last",
        "МесяцПоследнегоПоступления": "last", "ГодПоследнегоПоступления": "last",
    }).reset_index().rename(columns={
        "ОстатокНачалоНедели": "ОстатокНачалоКвартала",
        "temp_mean_week": "temp_mean_q", "precip_sum_week": "precip_sum_q",
        "temp_max_week": "temp_max_q", "temp_min_week": "temp_min_q",
        "days_off_in_week": "days_off_q",
    })
    dq["КодТовара"]     = _fix(dq["КодТовара"])
    dq["КварталНачало"] = pd.to_datetime(dq["КварталНачало"]).dt.tz_localize(None).dt.normalize()

    sells = (
        dq.set_index(["КодТовара", "КварталНачало"])[["Количество"]]
        .unstack(-1).fillna(0).sort_index().sort_index(axis=1)
    )
    sells.columns = pd.to_datetime(sells.columns.get_level_values(1)).tz_localize(None).normalize()
    sells.index   = _fix(sells.index)

    t = dq["КварталНачало"].max() + pd.DateOffset(months=3)
    t = pd.Timestamp(t.year, t.month, 1)
    q_num = ((t.month - 1) // 3) + 1
    period_label = f"Q{q_num} {t.year}"

    feats = _feats_quarter(sells, t)
    feats.index      = sells.index
    feats.index.name = "КодТовара"
    feats = feats.reset_index()
    feats["КодТовара"] = _fix(feats["КодТовара"])

    last = dq.sort_values("КварталНачало").groupby("КодТовара").last().reset_index()
    last["КварталНачало"] = t
    w = _climate_row(climate, t.month)
    last["temp_mean_q"]  = w["temp_mean_week"]
    last["precip_sum_q"] = w["precip_sum_week"] * 13
    last["temp_max_q"]   = w["temp_max_week"]
    last["temp_min_q"]   = w["temp_min_week"]
    last["days_off_q"]   = w["days_off_in_week"] * 13

    df_inf = last.merge(feats, on="КодТовара", how="inner")
    df_inf["КварталНачало"] = t

    df_inf = df_inf[df_inf["Папка1"] != "разобрать"]
    df_inf = df_inf[df_inf["ЕдиницаИзмерения"] == "шт"].drop("ЕдиницаИзмерения", axis=1)
    df_inf = df_inf[df_inf["ТоварнаяКатегория"] == "Штучный товар"].drop("ТоварнаяКатегория", axis=1)
    bad = df_inf.loc[
        (df_inf[["ОстатокНачалоКвартала", "Количество"]].astype(float) % 1 != 0).any(axis=1),
        "КодТовара",
    ].unique()
    df_inf = df_inf[~df_inf["КодТовара"].isin(bad)]
    df_inf = df_inf[df_inf["Номенклатура"] != "1"]

    X = df_inf.drop(columns=["Количество"], errors="ignore").copy()
    X["ТекущийМесяц"]   = t.month
    X["ТекущийГод"]     = t.year
    X["ТекущийКвартал"] = q_num
    X = X.drop(columns=["КварталНачало"], errors="ignore")
    return X, period_label


# ─────────────────────────── main ────────────────────────────────

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)

    print("Загрузка и подготовка данных...")
    df      = load_raw()
    climate = climate_by_month(df)

    print("Строим недельные фичи...")
    X_w, p_w = build_weekly(df, climate)
    X_w.to_csv(OUT / "X_week.csv", index=False, sep=";")
    (OUT / "period_week.txt").write_text(p_w, encoding="utf-8")
    print(f"  → {len(X_w)} товаров, период: {p_w}")

    print("Строим месячные фичи...")
    X_m, p_m = build_monthly(df, climate)
    X_m.to_csv(OUT / "X_month.csv", index=False, sep=";")
    (OUT / "period_month.txt").write_text(p_m, encoding="utf-8")
    print(f"  → {len(X_m)} товаров, период: {p_m}")

    print("Строим квартальные фичи...")
    X_q, p_q = build_quarterly(df, climate)
    X_q.to_csv(OUT / "X_3month.csv", index=False, sep=";")
    (OUT / "period_3month.txt").write_text(p_q, encoding="utf-8")
    print(f"  → {len(X_q)} товаров, период: {p_q}")

    print("\nГотово! Файлы сохранены в service/precomputed/")
