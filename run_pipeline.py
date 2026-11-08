import os
import sys
import subprocess
import shutil
import zipfile
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def main():
    # 1. Unzip the CSV if it is not present in data/raw
    csv_path = "data/raw/q1_q17_dr25_tce_2026_07_08_22_53_14.csv"
    if not os.path.exists(csv_path):
        log.info(f"{csv_path} not found. Searching for zip file to extract...")
        zip_filename = "kepler-tce-project.zip"
        search_dirs = ["data/raw", ".", "..", "../.."]
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
                    log.error(f"Could not find any CSV matching '{target_filename}' inside {zip_path}")
                    sys.exit(1)
        else:
            log.error(f"CSV file '{csv_path}' not found, and '{zip_filename}' could not be located in search paths: {search_dirs}")
            sys.exit(1)

    # 2. Run train_ml.py (Classical ML models)
    log.info("=" * 60)
    log.info("Step 1: Training Classical ML Models (LogReg, RF, XGBoost, SVM)")
    log.info("=" * 60)
    subprocess.run([sys.executable, "src/train_ml.py"], check=True)

    # 3. Run train_nn.py --model mlp (MLP Deep Learning model)
    log.info("=" * 60)
    log.info("Step 2: Training MLP Neural Network")
    log.info("=" * 60)
    subprocess.run([sys.executable, "src/train_nn.py", "--model", "mlp", "--epochs", "30"], check=True)

    # 4. Run train_nn.py --model transformer (Tabular Transformer model)
    log.info("=" * 60)
    log.info("Step 3: Training Tabular Transformer")
    log.info("=" * 60)
    # Using optimized parameters for fast execution
    subprocess.run([
        sys.executable, "src/train_nn.py", 
        "--model", "transformer", 
        "--epochs", "10", 
        "--batch_size", "2048", 
        "--d_model", "16", 
        "--n_heads", "2", 
        "--n_layers", "1", 
        "--patience", "4"
    ], check=True)

    # 5. Run evaluate.py
    log.info("=" * 60)
    log.info("Step 4: Evaluating All Models on Test Set")
    log.info("=" * 60)
    subprocess.run([sys.executable, "src/evaluate.py"], check=True)

    # 6. Create tain-data-patern folder and copy outputs
    output_dir = "tain-data-patern"
    log.info("=" * 60)
    log.info(f"Step 5: Creating '{output_dir}' and copying learned patterns/models")
    log.info("=" * 60)
    
    # Create clean directory structure inside tain-data-patern
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "models", "ml"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "models", "dl"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "reports", "figures"), exist_ok=True)

    # Copy ML models
    ml_models = ["logistic_regression.joblib", "random_forest.joblib", "gradient_boosting.joblib", "svm.joblib"]
    for m in ml_models:
        src_path = os.path.join("models", "ml", m)
        dst_path = os.path.join(output_dir, "models", "ml", m)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            log.info(f"Copied classical model: {src_path} -> {dst_path}")

    # Copy DL models
    dl_files = [
        "mlp_best.pt", "mlp_config.joblib", "mlp_final.pt",
        "transformer_best.pt", "transformer_config.joblib", "transformer_final.pt",
        "preprocessor.joblib", "feature_columns.joblib"
    ]
    for d in dl_files:
        src_path = os.path.join("models", "dl", d)
        dst_path = os.path.join(output_dir, "models", "dl", d)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            log.info(f"Copied neural model: {src_path} -> {dst_path}")

    # Copy reports
    report_files = ["model_comparison.csv", "summary_report.txt"]
    for r in report_files:
        src_path = os.path.join("reports", r)
        dst_path = os.path.join(output_dir, "reports", r)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            log.info(f"Copied report: {src_path} -> {dst_path}")

    # Copy figures (learned patterns/plots)
    figures = [
        "mlp_embedding_evolution.gif", "mlp_test_diagnostics.png", "mlp_training_curves.png",
        "transformer_embedding_evolution.gif", "transformer_test_diagnostics.png", "transformer_training_curves.png"
    ]
    brain_dir = r"C:\Users\Lenovo\.gemini\antigravity-ide\brain\c83bbe28-868f-43db-ac4d-5346ed0f68d7"
    for fig in figures:
        src_path = os.path.join("reports", "figures", fig)
        dst_path = os.path.join(output_dir, "reports", "figures", fig)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            log.info(f"Copied visualization: {src_path} -> {dst_path}")
            
            # Also copy to the brain folder for artifact visualization if it exists
            if os.path.exists(brain_dir):
                brain_dst_path = os.path.join(brain_dir, fig)
                shutil.copy2(src_path, brain_dst_path)
                log.info(f"Copied visualization to brain: {src_path} -> {brain_dst_path}")

    log.info("=" * 60)
    log.info(f"Pipeline executed successfully. All outputs saved in '{output_dir}/'")
    log.info("=" * 60)

if __name__ == "__main__":
    main()
