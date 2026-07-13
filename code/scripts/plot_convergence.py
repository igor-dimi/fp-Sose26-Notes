from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CODE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = CODE_DIR / "results" / "raw"
PLOTS_DIR = CODE_DIR / "results" / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_arguments():
    parser = ArgumentParser(
        description="Plot convergence results from a mixed-IR CSV file."
    )

    parser.add_argument(
        "csv_file",
        nargs="?",
        default="convergence_fp16_fp64_fp128.csv",
        help=(
            "CSV filename in results/raw, or a direct path to a CSV file. "
            "Defaults to convergence_fp16_fp64_fp128.csv."
        ),
    )

    parser.add_argument(
        "--label",
        default="",
        help="Optional dataset description used in plot titles.",
    )

    parser.add_argument(
        "--tag",
        default=None,
        help=(
            "Optional tag used in output filenames. "
            "By default, it is derived from the CSV filename."
        ),
    )

    return parser.parse_args()


def resolve_csv_path(csv_argument):
    path = Path(csv_argument)

    # First allow a direct or relative path supplied by the user.
    if path.exists():
        return path.resolve()

    # Otherwise, look in code/results/raw.
    path = RAW_DIR / csv_argument

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find CSV file: {csv_argument}\n"
            f"Also checked: {path}"
        )

    return path


def output_tag(csv_path, explicit_tag):
    if explicit_tag:
        return explicit_tag

    tag = csv_path.stem

    # Avoid filenames such as:
    #
    # convergence_forward_error_convergence_random_spd_...
    if tag.startswith("convergence_"):
        tag = tag[len("convergence_"):]

    return tag


def title_with_label(title, label):
    if label:
        return f"{label}: {title}"

    return title


def save_figure(fig, filename):
    output_path = PLOTS_DIR / filename

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Wrote {output_path}")


def positive_for_log_scale(dataframe, columns):
    result = dataframe.copy()

    for column in columns:
        result.loc[result[column] <= 0.0, column] = float("nan")

    return result


def plot_history(
    dataframe,
    column,
    ylabel,
    title,
    filename,
    label,
):
    fig, ax = plt.subplots()

    for kappa, group in dataframe.groupby("kappa"):
        group = group.sort_values("iteration")

        ax.plot(
            group["iteration"],
            group[column],
            marker="o",
            label=fr"$\kappa={kappa:g}$",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Refinement iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title_with_label(title, label))
    ax.grid(True, which="both")
    ax.legend()

    save_figure(fig, filename)


def main():
    args = parse_arguments()

    csv_path = resolve_csv_path(args.csv_file)
    tag = output_tag(csv_path, args.tag)

    print(f"Reading {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = {
        "kappa",
        "iteration",
        "forward_error_inf",
        "backward_error_inf",
        "rel_correction",
        "converged",
        "total_iterations",
        "final_rel_correction",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"CSV file is missing required columns: {missing}"
        )

    df = df.sort_values(["kappa", "iteration"])

    # Matplotlib cannot display zero on a logarithmic axis.
    history_plot = positive_for_log_scale(
        df,
        [
            "forward_error_inf",
            "backward_error_inf",
            "rel_correction",
        ],
    )

    plot_history(
        history_plot,
        column="forward_error_inf",
        ylabel="Relative forward error, infinity norm",
        title=(
            "Relative forward error, infinity norm "
            "vs. refinement iteration"
        ),
        filename=f"convergence_forward_error_{tag}.png",
        label=args.label,
    )

    plot_history(
        history_plot,
        column="backward_error_inf",
        ylabel="Normwise backward error, infinity norm",
        title=(
            "Normwise backward error, infinity norm "
            "vs. refinement iteration"
        ),
        filename=f"convergence_backward_error_{tag}.png",
        label=args.label,
    )

    plot_history(
        history_plot,
        column="rel_correction",
        ylabel="Relative correction norm",
        title="Relative correction norm vs. refinement iteration",
        filename=f"convergence_rel_correction_{tag}.png",
        label=args.label,
    )

    # Final row for every condition number.
    final_summary = (
        df.groupby("kappa", as_index=False)
        .tail(1)
        .sort_values("kappa")
        .copy()
    )

    final_summary_plot = positive_for_log_scale(
        final_summary,
        [
            "forward_error_inf",
            "backward_error_inf",
        ],
    )

    fig, ax = plt.subplots()

    ax.plot(
        final_summary_plot["kappa"],
        final_summary_plot["forward_error_inf"],
        marker="o",
        label="final forward error",
    )

    ax.plot(
        final_summary_plot["kappa"],
        final_summary_plot["backward_error_inf"],
        marker="o",
        label="final backward error",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Condition number $\kappa$")
    ax.set_ylabel("Final error")
    ax.set_title(
        title_with_label(
            "Final errors vs. condition number",
            args.label,
        )
    )
    ax.grid(True, which="both")
    ax.legend()

    save_figure(
        fig,
        f"summary_final_errors_{tag}.png",
    )

    # Best errors attained at any iteration.
    best_summary = (
        df.groupby("kappa", as_index=False)
        .agg(
            best_forward_error=("forward_error_inf", "min"),
            best_backward_error=("backward_error_inf", "min"),
        )
        .sort_values("kappa")
    )

    best_summary_plot = positive_for_log_scale(
        best_summary,
        [
            "best_forward_error",
            "best_backward_error",
        ],
    )

    fig, ax = plt.subplots()

    ax.plot(
        best_summary_plot["kappa"],
        best_summary_plot["best_forward_error"],
        marker="o",
        label="best forward error",
    )

    ax.plot(
        best_summary_plot["kappa"],
        best_summary_plot["best_backward_error"],
        marker="o",
        label="best backward error",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Condition number $\kappa$")
    ax.set_ylabel("Best attained error")
    ax.set_title(
        title_with_label(
            "Best attained errors vs. condition number",
            args.label,
        )
    )
    ax.grid(True, which="both")
    ax.legend()

    save_figure(
        fig,
        f"summary_best_errors_{tag}.png",
    )

    # Number of refinement iterations.
    fig, ax = plt.subplots()

    ax.plot(
        final_summary["kappa"],
        final_summary["total_iterations"],
        marker="o",
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"Condition number $\kappa$")
    ax.set_ylabel("Total refinement iterations")
    ax.set_title(
        title_with_label(
            "Iterations vs. condition number",
            args.label,
        )
    )
    ax.grid(True, which="both")

    save_figure(
        fig,
        f"summary_iterations_{tag}.png",
    )

    # Add the iterations at which the best errors occurred.
    best_forward_rows = df.loc[
        df.groupby("kappa")["forward_error_inf"].idxmin(),
        ["kappa", "iteration"],
    ].rename(
        columns={"iteration": "best_forward_iteration"}
    )

    best_backward_rows = df.loc[
        df.groupby("kappa")["backward_error_inf"].idxmin(),
        ["kappa", "iteration"],
    ].rename(
        columns={"iteration": "best_backward_iteration"}
    )

    printed_summary = (
        final_summary[
            [
                "kappa",
                "converged",
                "total_iterations",
                "forward_error_inf",
                "backward_error_inf",
                "final_rel_correction",
            ]
        ]
        .merge(best_summary, on="kappa")
        .merge(best_forward_rows, on="kappa")
        .merge(best_backward_rows, on="kappa")
    )

    print("\nSummary:")
    print(printed_summary.to_string(index=False))


if __name__ == "__main__":
    main()