"""
langchain_report.py
Uses LangChain to turn the raw model_comparison.csv metrics into a plain-
English summary report. If an LLM API key is available in the environment
(OPENAI_API_KEY or ANTHROPIC_API_KEY) it calls the real model through
LangChain's chat interface; otherwise it falls back to a deterministic
template-based report built with the same LangChain PromptTemplate, so the
script always runs end-to-end offline.
"""
import os
import pandas as pd

try:
    from langchain_core.prompts import PromptTemplate
except Exception:
    PromptTemplate = None

if PromptTemplate is not None:
    REPORT_PROMPT = PromptTemplate.from_template(
        """You are an ML analyst. Given this model comparison table for a Kepler
exoplanet TCE (Threshold Crossing Event) classifier, write a concise
(150-200 word) report for a technical audience covering:
- which model performed best and by how much
- any notable gap between classical ML and the neural models
- one caveat about the label strategy (koi_disposition merge vs proxy label)
- one concrete next step to improve the pipeline

Metrics table (csv):
{metrics_csv}

Label source in use: {label_source}
"""
    )
else:
    REPORT_PROMPT = None


def _try_llm_report(prompt_text: str) -> str | None:
    """Attempt a real LLM call via LangChain. Returns None if no provider
    is configured or the call fails, so the caller can fall back."""
    try:
        if os.environ.get("ANTHROPIC_API_KEY"):
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.3)
        elif os.environ.get("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        else:
            return None
        response = llm.invoke(prompt_text)
        return response.content
    except Exception as e:
        print(f"[langchain_report] LLM call unavailable ({e}); using template fallback.")
        return None


def _template_fallback_report(df: pd.DataFrame, label_source: str) -> str:
    best = df.sort_values("f1", ascending=False).iloc[0]
    worst = df.sort_values("f1", ascending=False).iloc[-1]
    neural = df[df["model"].isin(["mlp", "transformer"])]
    classical = df[~df["model"].isin(["mlp", "transformer"])]

    lines = [
        f"MODEL COMPARISON SUMMARY (template report — no LLM API key found)\n",
        f"Best model: {best['model']} (F1={best['f1']:.3f}, ROC-AUC={best['roc_auc']:.3f}, "
        f"accuracy={best['accuracy']:.3f}).",
        f"Weakest model: {worst['model']} (F1={worst['f1']:.3f}).",
    ]
    if not neural.empty and not classical.empty:
        lines.append(
            f"Classical ML mean F1={classical['f1'].mean():.3f} vs. "
            f"neural models mean F1={neural['f1'].mean():.3f} — "
            f"{'neural models edge out classical ML' if neural['f1'].mean() > classical['f1'].mean() else 'classical ML matches or beats the neural models on this feature set'}."
        )
    lines.append(
        f"Label source: {label_source}. "
        + (
            "Labels came from a real koi_disposition merge, so metrics reflect a "
            "genuine astrophysical classification task."
            if "koi_merge" in label_source
            else "Labels are a weak proxy (tce_rogue_flag + tce_nkoi), so treat these "
                 "numbers as a pipeline sanity check, not a validated science result — "
                 "re-run with internet access to merge real KOI dispositions."
        )
    )
    lines.append(
        "Next step: run a permutation-importance or SHAP analysis on the best model "
        "to check whether it's relying on physically meaningful features (SNR, depth, "
        "centroid offsets) rather than incidental artifacts of the proxy label."
    )
    return "\n".join(lines)


def generate_report(comparison_csv="reports/model_comparison.csv", label_source="unknown"):
    if os.path.exists("reports/label_source.txt"):
        try:
            with open("reports/label_source.txt", "r", encoding="utf-8") as f:
                label_source = f.read().strip() or label_source
        except Exception:
            pass

    df = pd.read_csv(comparison_csv)

    if REPORT_PROMPT is not None:
        prompt_text = REPORT_PROMPT.format(metrics_csv=df.to_csv(index=False), label_source=label_source)
        report = _try_llm_report(prompt_text)
    else:
        report = None

    if report is None:
        report = _template_fallback_report(df, label_source)

    with open("reports/summary_report.txt", "w") as f:
        f.write(report)
    print(report)
    return report


if __name__ == "__main__":
    generate_report()
