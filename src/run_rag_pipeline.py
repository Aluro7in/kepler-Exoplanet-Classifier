import os
import subprocess
import sys


def main():
    os.makedirs("reports", exist_ok=True)
    os.makedirs("visuals", exist_ok=True)
    subprocess.run([sys.executable, "src/train_ml.py"], check=True)
    subprocess.run([sys.executable, "src/train_nn.py", "--model", "mlp", "--epochs", "12"], check=True)
    subprocess.run([sys.executable, "src/train_nn.py", "--model", "transformer", "--epochs", "8", "--batch_size", "1024", "--d_model", "16", "--n_heads", "2", "--n_layers", "1", "--patience", "3"], check=True)
    subprocess.run([sys.executable, "src/evaluate.py"], check=True)
    subprocess.run([sys.executable, "src/rag_pipeline.py", "--query", "Summarize the most promising exoplanet candidate signals from all available Kepler datasets."], check=True)


if __name__ == "__main__":
    main()
