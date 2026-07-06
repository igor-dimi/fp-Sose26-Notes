from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


CODE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = CODE_DIR / "results" / "raw"
PLOTS_DIR = CODE_DIR / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

csv_file = RAW_DIR / "convergence_fp16_fp64_fp128.csv"
df = pd.read_csv(csv_file)

# Avoid plotting exact zeros on log scale.
# Matplotlib cannot show 0 on a logarithmic axis.
plot_df = df.copy()
for col in ["forward_error_inf", "backward_error_inf", "rel_correction"]:
    plot_df.loc[plot_df[col] <= 0.0, col] = float("nan")


def plot_history(column, ylabel, filename):
    plt.figure()

    for kappa, group in plot_df.groupby("kappa"):
        # kappa = 1 has exact zero errors in this experiment, so it may disappear
        # on log plots. That is fine.
        plt.plot(
            group["iteration"],
            group[column],
            marker="o",
            label=f"kappa={kappa:g}",
        )

    plt.yscale("log")
    plt.xlabel("Refinement iteration")
    plt.ylabel(ylabel)
    plt.title(ylabel + " vs. refinement iteration")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    out = PLOTS_DIR / filename
    plt.savefig(out, dpi=200)
    print(f"Wrote {out}")


plot_history(
    "forward_error_inf",
    "Relative forward error, infinity norm",
    "convergence_forward_error_fp16_fp64_fp128.png",
)

plot_history(
    "backward_error_inf",
    "Normwise backward error, infinity norm",
    "convergence_backward_error_fp16_fp64_fp128.png",
)

plot_history(
    "rel_correction",
    "Relative correction norm",
    "convergence_rel_correction_fp16_fp64_fp128.png",
)


# One final row per kappa.
summary = df.sort_values(["kappa", "iteration"]).groupby("kappa").tail(1).copy()

summary_plot = summary.copy()
for col in ["forward_error_inf", "backward_error_inf", "final_rel_correction"]:
    summary_plot.loc[summary_plot[col] <= 0.0, col] = float("nan")


plt.figure()
plt.plot(
    summary_plot["kappa"],
    summary_plot["forward_error_inf"],
    marker="o",
    label="final forward error",
)
plt.plot(
    summary_plot["kappa"],
    summary_plot["backward_error_inf"],
    marker="o",
    label="final backward error",
)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Condition number kappa")
plt.ylabel("Final error")
plt.title("Final errors vs. condition number")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()

out = PLOTS_DIR / "summary_final_errors_fp16_fp64_fp128.png"
plt.savefig(out, dpi=200)
print(f"Wrote {out}")


plt.figure()
plt.plot(
    summary["kappa"],
    summary["total_iterations"],
    marker="o",
)

plt.xscale("log")
plt.xlabel("Condition number kappa")
plt.ylabel("Total refinement iterations")
plt.title("Iterations vs. condition number")
plt.grid(True, which="both")
plt.tight_layout()

out = PLOTS_DIR / "summary_iterations_fp16_fp64_fp128.png"
plt.savefig(out, dpi=200)
print(f"Wrote {out}")


print("\nSummary:")
print(summary[[
    "kappa",
    "converged",
    "total_iterations",
    "forward_error_inf",
    "backward_error_inf",
    "final_rel_correction",
]])