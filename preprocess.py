# noshow_iq/preprocess.py
# Step 1 of the pipeline: clean and engineer features from raw CSV data.

import pandas as pd
import numpy as np

RENAME_MAP = {
    'No-show':        'no_show',
    'Hipertension':   'hypertension',
    'Handcap':        'handicap',
    'SMS_received':   'sms_received',
    'ScheduledDay':   'scheduled_day',
    'AppointmentDay': 'appointment_day',
}


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw Kaggle CSV and rename columns to snake_case."""
    df = pd.read_csv(path)
    return df.rename(columns=RENAME_MAP)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, fix ages, and encode the target column."""
    df = df.copy()
    df['scheduled_day']   = pd.to_datetime(df['scheduled_day'],   utc=True)
    df['appointment_day'] = pd.to_datetime(df['appointment_day'], utc=True)
    df = df[(df['Age'] >= 0) & (df['Age'] <= 110)]          # remove bad ages
    df['no_show'] = (df['no_show'] == 'Yes').astype(int)    # 1 = no-show
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features used by the model."""
    df = df.copy()
    # Days between booking and appointment (stronger signal than raw date)
    df['days_in_advance'] = (
        df['appointment_day'].dt.normalize() - df['scheduled_day'].dt.normalize()
    ).dt.days
    df.loc[df['days_in_advance'] < 0, 'days_in_advance'] = 0  # fix rare negatives

    # Day of week (0 = Monday, 6 = Sunday)
    df['appointment_dow'] = df['appointment_day'].dt.dayofweek

    # Age group bucket – handles non-linear age effects in tree models
    df['age_group'] = pd.cut(
        df['Age'],
        bins=[-1, 12, 18, 35, 60, 120],
        labels=['child', 'teen', 'young_adult', 'adult', 'senior'],
    )
    return df


def prepare(df: pd.DataFrame):
    """Full pipeline – returns (X, y) ready for model.train()."""
    df = clean(df)
    df = engineer(df)
    df = pd.get_dummies(df, columns=['Gender', 'age_group'], drop_first=True)

    drop_cols = {'no_show', 'PatientId', 'AppointmentID',
                 'scheduled_day', 'appointment_day', 'Neighbourhood'}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    return df[feature_cols], df['no_show']
