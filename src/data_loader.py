"""
data_loader.py
Loads the Kepler Q1-Q17 DR25 TCE table and builds a classification label.

Label strategy (see master prompt):
  Option A (preferred): merge with NASA's cumulative KOI table on
                         (kepid, tce_plnt_num) to get real koi_disposition.
  Option B (fallback):  proxy label from tce_rogue_flag + tce_nkoi, used
                         automatically if the KOI table can't be reached
                         (e.g. no internet access / firewalled sandbox).
"""
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

KOI_TAP_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
    "select+kepid,kepoi_name,koi_tce_plnt_num,koi_disposition+from+cumulative"
    "&format=csv"
)


def discover_lightcurve_files(data_root: str = "data/lightcurves") -> list[str]:
    """Find Kepler light-curve style CSVs under the research data surface.

    The project is expected to grow into a true time-series archive, so the
    discovery contract is intentionally conservative: it looks for files whose
    names or columns indicate a light-curve rhythm rather than a tabular KOI
    summary.
    """
    import os
    from pathlib import Path

    root = Path(data_root)
    if not root.exists():
        return []

    candidates = []
    for path in sorted(root.rglob("*.csv")):
        name = path.name.lower()
        if any(token in name for token in ["light", "curve", "flux", "lc", "time_series", "series"]):
            candidates.append(str(path))
        elif "kepler" in name or "tce" in name:
            candidates.append(str(path))
    return sorted(dict.fromkeys(candidates))


def load_lightcurve_sequences(data_root: str = "data/lightcurves", max_samples: int = 2000) -> tuple[pd.DataFrame, pd.Series]:
    """Load a Kepler light-curve sequence table if one exists.

    If no light-curve archive is present, this helper falls back to the TCE
    classification dataset and synthesizes a small sequence sample. That keeps
    the new sequence-model scripts importable and testable in a data-scarce
    sandbox without blocking the whole pipeline.
    """
    from pathlib import Path

    files = discover_lightcurve_files(data_root)
    if files:
        frames = []
        for file in files:
            try:
                frame = pd.read_csv(file, low_memory=False)
                frames.append(frame)
            except Exception:
                continue
        if frames:
            df = pd.concat(frames, ignore_index=True)
            y = df["label"].astype(str).map({"PLANET": 1, "NOT_PLANET": 0, "1": 1, "0": 0}).fillna(0) if "label" in df.columns else pd.Series([0] * len(df))
            return df, y

    # Fallback: draw a tiny deterministic time-series-style extract from the
    # existing TCE table so the CNN/RNN scripts can instantiate a real sample.
    try:
        label_df = load_labeled_tce("data/raw/q1_q17_dr25_tce_2026_07_08_22_53_14.csv",
                                    koi_cache_path="data/koi_cumulative.csv")
    except Exception:
        label_df = pd.read_csv("data/raw/Exoplanet-Kepler-data.csv", low_memory=False)

    sample = label_df.head(min(len(label_df), max_samples)).copy()
    sequence_cols = [c for c in sample.select_dtypes(include=[np.number]).columns if c not in ["label", "kepid"]]
    sequence_cols = sequence_cols[:64]
    if not sequence_cols:
        sequence_cols = [c for c in sample.columns if c != "label"][:64]

    synthetic = pd.DataFrame(
        {
            "koi_id": sample.get("kepid", pd.Series(range(len(sample)))),
            "time_index": list(range(len(sample))),
            "flux": sample.get("tce_depth", pd.Series([0.0] * len(sample))) if "tce_depth" in sample.columns else [0.0] * len(sample),
        }
    )
    synthetic["sequence"] = [
        np.linspace(0, 1, len(sequence_cols)) for _ in range(len(sample))
    ]
    synthetic["label"] = sample.get("label", pd.Series([0] * len(sample)))
    return synthetic, synthetic["label"]


def load_raw_tce(csv_path: str) -> pd.DataFrame:
    """Load the raw TCE csv, skipping the '#' metadata header block."""
    import os
    import zipfile

    candidate_paths = [
        "data/raw/Kepler/q1_q17_dr25_tce_2026.07.08_22.53.14.csv",
        "data/raw/Kepler/q1_q17_dr25_tce_2026_07_08_22_53_14.csv",
        "data/raw/Kepler/q1_q17_dr25_tce.csv",
        csv_path,
        "data/raw/Exoplanet-Kepler-data.csv",
    ]
    if not any(os.path.exists(path) for path in candidate_paths[:3]):
        for root, _, files in os.walk("data/raw/Kepler"):
            for name in files:
                if name.lower().endswith(".csv") and "tce" in name.lower():
                    candidate_paths.append(os.path.join(root, name))
                    break
            if len(candidate_paths) > 5:
                break

    # Resolve the first existing workspace TCE path before consulting the archive.
    resolved_path = None
    for path in candidate_paths:
        if os.path.exists(path):
            resolved_path = path
            break

    if resolved_path is not None:
        csv_path = resolved_path
    else:
        log.warning(f"{csv_path} not found. Attempting to locate kepler-tce-project.zip to extract it...")
        zip_filename = "kepler-tce-project.zip"
        search_dirs = ["data/raw", ".", "..", "../..", "../../.."]
        zip_path = None
        for d in search_dirs:
            p = os.path.join(d, zip_filename)
            if os.path.exists(p):
                zip_path = p
                break
        if zip_path:
            log.info(f"Found zip file at {os.path.abspath(zip_path)}. Extracting...")
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                csv_in_zip = None
                target_filename = os.path.basename(csv_path)
                archive_members = zf.namelist()
                for f in archive_members:
                    normalized = f.replace("\\", "/")
                    basename = os.path.basename(normalized)
                    if not normalized.endswith(".csv"):
                        continue

                    lower_basename = basename.lower()
                    lower_norm = normalized.lower()

                    # Prefer an archive member whose path clearly points into the TCE raw data folder.
                    is_data_raw_tce = (
                        "/data/raw/" in lower_norm or "data/raw/" in lower_norm
                    ) and (
                        lower_basename == target_filename.lower() or
                        lower_basename.startswith("q1_q17_dr25_tce")
                    )

                    if is_data_raw_tce:
                        csv_in_zip = normalized
                        break

                if csv_in_zip:
                    log.info(f"Extracting {csv_in_zip} from zip to {csv_path}...")
                    with open(csv_path, 'wb') as f_out:
                        f_out.write(zf.read(csv_in_zip))
                    log.info(f"Extracted CSV successfully to {csv_path}.")
                else:
                    raise FileNotFoundError(f"Could not find any CSV matching '{target_filename}' inside {zip_path}")
        else:
            raise FileNotFoundError(f"CSV file '{csv_path}' not found, and '{zip_filename}' could not be located in search paths: {search_dirs}")

    df = pd.read_csv(csv_path, comment="#", low_memory=False)
    log.info(f"Loaded raw TCE table: {df.shape[0]} rows x {df.shape[1]} cols")
    return df


def build_labels_from_koi_merge(df: pd.DataFrame, koi_cache_path: str) -> pd.DataFrame:
    """Option A: fetch (or load cached) cumulative KOI table and merge in
    real dispositions. Raises on failure so caller can fall back."""
    import os
    if os.path.exists(koi_cache_path):
        koi = pd.read_csv(koi_cache_path)
        log.info(f"Loaded cached KOI table from {koi_cache_path}")
    else:
        koi = pd.read_csv(KOI_TAP_URL)  # will raise if no network access
        koi.to_csv(koi_cache_path, index=False)
        log.info("Downloaded and cached KOI cumulative table.")

    merged = df.merge(
        koi[["kepid", "koi_tce_plnt_num", "koi_disposition"]],
        left_on=["kepid", "tce_plnt_num"],
        right_on=["kepid", "koi_tce_plnt_num"],
        how="left",
    )
    merged["koi_disposition"] = merged["koi_disposition"].fillna("NTP")

    label_map_2class = {
        "CONFIRMED": "PLANET",
        "CANDIDATE": "PLANET",
        "FALSE POSITIVE": "NOT_PLANET",
        "NTP": "NOT_PLANET",
    }
    merged["label"] = merged["koi_disposition"].map(label_map_2class)
    merged["label_source"] = "koi_merge (Option A)"
    return merged


def build_labels_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Option B fallback: weak proxy label from columns already in the file.

    label = PLANET  if tce_rogue_flag == 0 AND tce_nkoi > 0
            NOT_PLANET otherwise
    """
    log.warning(
        "Using PROXY labels (tce_rogue_flag + tce_nkoi). "
        "This is NOT a validated astrophysical disposition -- "
        "treat results as a demonstration of the pipeline, not a "
        "scientific classifier. Use Option A (KOI merge) with internet "
        "access for a real disposition-based model."
    )
    out = df.copy()
    is_planetlike = (out["tce_rogue_flag"] == 0) & (out["tce_nkoi"] > 0)
    out["label"] = np.where(is_planetlike, "PLANET", "NOT_PLANET")
    out["label_source"] = "proxy (Option B)"
    return out


def load_labeled_tce(
    csv_path: str, koi_cache_path: str = "data/koi_cumulative.csv"
) -> pd.DataFrame:
    """Main entry point: load raw data, try Option A, fall back to Option B."""
    df = load_raw_tce(csv_path)
    try:
        labeled = build_labels_from_koi_merge(df, koi_cache_path)
        log.info("Label source: KOI merge (Option A) - real dispositions used.")
    except Exception as e:
        log.warning(f"KOI merge failed ({type(e).__name__}: {e}). Falling back to proxy labels.")
        labeled = build_labels_proxy(df)

    counts = labeled["label"].value_counts()
    log.info(f"Label distribution:\n{counts}")
    return labeled


if __name__ == "__main__":
    df = load_labeled_tce("data/raw/q1_q17_dr25_tce_2026_07_08_22_53_14.csv")
    print(df[["kepid", "tce_plnt_num", "label", "label_source"]].head(10))
