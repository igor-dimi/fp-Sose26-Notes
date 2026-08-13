# Convenience targets for building, testing, running experiments,
# and generating plots.
#
# CMake remains the build system. This Makefile only orchestrates
# common project workflows.

.DEFAULT_GOAL := help

CMAKE  ?= cmake
PYTHON ?= python3
QUARTO ?= quarto
PRESET ?= release

REPORT_PROFILE ?= report
MANUAL_PROFILE ?= manual

REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
CODE_DIR  := $(REPO_ROOT)/code
SCRIPT_DIR := $(CODE_DIR)/scripts

RAW_RESULTS_DIR   := $(REPO_ROOT)/results/raw
PLOT_RESULTS_DIR  := $(REPO_ROOT)/results/plots

ifeq ($(PRESET),release)
BUILD_DIR := $(REPO_ROOT)/build/fp-release
else ifeq ($(PRESET),debug)
BUILD_DIR := $(REPO_ROOT)/build/fp-debug
else
$(error PRESET must be either 'release' or 'debug')
endif

BIN_DIR := $(BUILD_DIR)/bin

STAMP_DIR        := $(BUILD_DIR)/make-stamps
EXPERIMENTS_STAMP := $(STAMP_DIR)/experiments.stamp
PLOTS_STAMP       := $(STAMP_DIR)/plots.stamp
REPORT_STAMP      := $(STAMP_DIR)/report.stamp
MANUAL_STAMP      := $(STAMP_DIR)/manual.stamp

REPORT_INDEX := $(REPO_ROOT)/report/index.qmd
MANUAL_INDEX := $(REPO_ROOT)/manual/index.qmd

# Inputs used to decide whether generated results are still current.
EXPERIMENT_INPUTS := \
	$(wildcard $(CODE_DIR)/CMakeLists.txt) \
	$(wildcard $(CODE_DIR)/experiments/*.cc) \
	$(wildcard $(CODE_DIR)/include/*.hpp)

PLOT_INPUTS := \
	$(wildcard $(SCRIPT_DIR)/*.py) \
	$(wildcard $(SCRIPT_DIR)/mpir_plotting/*.py)

RAW_RESULTS := $(shell find "$(RAW_RESULTS_DIR)" -type f -name '*.csv' 2>/dev/null)
GENERATED_PLOTS := $(shell find "$(PLOT_RESULTS_DIR)" -type f \
	\( -name '*.png' -o -name '*.pdf' -o -name '*.svg' \) 2>/dev/null)

REPORT_INPUTS := \
	$(REPORT_INDEX) \
	$(wildcard $(REPO_ROOT)/_quarto.yml) \
	$(wildcard $(REPO_ROOT)/_quarto-$(REPORT_PROFILE).yml) \
	$(wildcard $(REPO_ROOT)/references.bib) \
	$(GENERATED_PLOTS)

MANUAL_INPUTS := \
	$(MANUAL_INDEX) \
	$(wildcard $(REPO_ROOT)/_quarto.yml) \
	$(wildcard $(REPO_ROOT)/_quarto-$(MANUAL_PROFILE).yml)


# --------------------------------------------------------------------
# Experiment executables
# --------------------------------------------------------------------

EXP_CONDITION_SWEEPS := \
	$(BIN_DIR)/mp_exp_condition_sweeps

EXP_CONVERGENCE_HISTORIES := \
	$(BIN_DIR)/mp_exp_convergence_histories

EXP_DIRECT_SOLVE_COMPARISON := \
	$(BIN_DIR)/mp_exp_direct_solve_comparison

EXP_RESIDUAL_PRECISION := \
	$(BIN_DIR)/mp_exp_residual_precision

EXP_RESIDUAL_SCALING := \
	$(BIN_DIR)/mp_exp_residual_scaling_fp16_fp64_fp128


# --------------------------------------------------------------------
# Phony targets
# --------------------------------------------------------------------

.PHONY: help configure build tests \
	experiments \
	experiment-condition-sweeps \
	experiment-convergence-histories \
	experiment-direct-solve-comparison \
	experiment-residual-precision \
	experiment-residual-scaling \
	plots \
	plot-condition-sweeps \
	plot-convergence-histories \
	plot-direct-solve-comparison \
	plot-residual-precision \
	plot-residual-scaling-diagnostics \
	plot-residual-scaling-errors \
	reproduce \
	report manual


# --------------------------------------------------------------------
# Help
# --------------------------------------------------------------------

help:
	@echo "Mixed-precision iterative refinement project"
	@echo
	@echo "Build:"
	@echo "  make configure                     Configure CMake ($(PRESET))"
	@echo "  make build                         Build the project"
	@echo "  make tests                         Build and run CTest"
	@echo
	@echo "Experiments:"
	@echo "  make experiments                   Run all report experiments"
	@echo "  make experiment-condition-sweeps"
	@echo "  make experiment-convergence-histories"
	@echo "  make experiment-direct-solve-comparison"
	@echo "  make experiment-residual-precision"
	@echo "  make experiment-residual-scaling"
	@echo
	@echo "Plots:"
	@echo "  make plots                         Generate all report plots"
	@echo
	@echo "Documents:"
	@echo "  make report                        Reproduce results and render the report"
	@echo "  make manual                        Render the manual"
	@echo
	@echo "Complete workflow:"
	@echo "  make reproduce                     Build, run experiments, plot results"
	@echo
	@echo "Use PRESET=debug for a debug build, e.g."
	@echo "  make PRESET=debug build"


# --------------------------------------------------------------------
# Build and tests
# --------------------------------------------------------------------

configure:
	@echo "==> Configuring $(PRESET) build"
	@cd "$(CODE_DIR)" && $(CMAKE) --preset "$(PRESET)"

build: configure
	@echo "==> Building $(PRESET) configuration"
	@cd "$(CODE_DIR)" && $(CMAKE) --build --preset "$(PRESET)"

tests: build
	@echo "==> Running tests"
	@cd "$(CODE_DIR)" && ctest --preset "$(PRESET)"


# --------------------------------------------------------------------
# Individual experiments
# --------------------------------------------------------------------

experiment-condition-sweeps: build
	@echo "==> Running condition-number sweeps"
	@"$(EXP_CONDITION_SWEEPS)"

experiment-convergence-histories: build
	@echo "==> Running convergence histories"
	@"$(EXP_CONVERGENCE_HISTORIES)"

experiment-direct-solve-comparison: build
	@echo "==> Running direct-solve comparison"
	@"$(EXP_DIRECT_SOLVE_COMPARISON)"

experiment-residual-precision: build
	@echo "==> Running residual-precision comparison"
	@"$(EXP_RESIDUAL_PRECISION)"

experiment-residual-scaling: build
	@echo "==> Running residual-scaling experiment"
	@"$(EXP_RESIDUAL_SCALING)"


# Run all report experiments.
experiments: \
	experiment-condition-sweeps \
	experiment-convergence-histories \
	experiment-direct-solve-comparison \
	experiment-residual-precision \
	experiment-residual-scaling
	@mkdir -p "$(STAMP_DIR)"
	@touch "$(EXPERIMENTS_STAMP)"


# --------------------------------------------------------------------
# Individual plot groups
# --------------------------------------------------------------------

plot-condition-sweeps:
	@echo "==> Plotting condition-number sweeps"
	@$(PYTHON) "$(SCRIPT_DIR)/plot_condition_sweeps.py" \
		--raw-root "$(RAW_RESULTS_DIR)" \
		--plots-root "$(PLOT_RESULTS_DIR)"

plot-convergence-histories:
	@echo "==> Plotting convergence histories"
	@$(PYTHON) "$(SCRIPT_DIR)/plot_convergence_histories.py" \
		--raw-root "$(RAW_RESULTS_DIR)" \
		--plots-root "$(PLOT_RESULTS_DIR)"

plot-direct-solve-comparison:
	@echo "==> Plotting direct-solve comparison"
	@$(PYTHON) "$(SCRIPT_DIR)/plot_direct_solve_comparison.py" \
		--raw-root "$(RAW_RESULTS_DIR)" \
		--plots-root "$(PLOT_RESULTS_DIR)"

plot-residual-precision:
	@echo "==> Plotting residual-precision comparison"
	@$(PYTHON) "$(SCRIPT_DIR)/plot_residual_precision.py" \
		--raw-root "$(RAW_RESULTS_DIR)" \
		--plots-root "$(PLOT_RESULTS_DIR)"

plot-residual-scaling-diagnostics:
	@echo "==> Plotting residual-scaling diagnostics"
	@$(PYTHON) "$(SCRIPT_DIR)/plot_residual_scaling_diagnostics.py" \
		--raw-root "$(RAW_RESULTS_DIR)" \
		--plots-root "$(PLOT_RESULTS_DIR)"

plot-residual-scaling-errors:
	@echo "==> Plotting residual-scaling error histories"
	@$(PYTHON) "$(SCRIPT_DIR)/plot_residual_scaling_errors.py" \
		--raw-root "$(RAW_RESULTS_DIR)" \
		--plots-root "$(PLOT_RESULTS_DIR)"


# Generate every report plot from the existing raw CSV files.
plots: \
	plot-condition-sweeps \
	plot-convergence-histories \
	plot-direct-solve-comparison \
	plot-residual-precision \
	plot-residual-scaling-diagnostics \
	plot-residual-scaling-errors
	@mkdir -p "$(STAMP_DIR)"
	@touch "$(PLOTS_STAMP)"


# --------------------------------------------------------------------
# Complete reproducibility workflow
# --------------------------------------------------------------------

$(STAMP_DIR):
	@mkdir -p "$@"

# Re-run the experiments only when their C++ inputs have changed.
$(EXPERIMENTS_STAMP): $(EXPERIMENT_INPUTS) | $(STAMP_DIR)
	@$(MAKE) -f "$(firstword $(MAKEFILE_LIST))" experiments

# Recreate the plots when experiments, raw CSV files, or plotting code change.
$(PLOTS_STAMP): $(EXPERIMENTS_STAMP) $(RAW_RESULTS) $(PLOT_INPUTS) | $(STAMP_DIR)
	@$(MAKE) -f "$(firstword $(MAKEFILE_LIST))" plots

reproduce: $(PLOTS_STAMP)


# --------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------

# Render the report when its source or generated plots change.
$(REPORT_STAMP): $(REPORT_INPUTS) $(PLOTS_STAMP) | $(STAMP_DIR)
	@echo "==> Rendering report"
	@cd "$(REPO_ROOT)" && $(QUARTO) render --profile "$(REPORT_PROFILE)"
	@touch "$@"

report: $(REPORT_STAMP)

# The manual depends only on its own source and Quarto configuration.
$(MANUAL_STAMP): $(MANUAL_INPUTS) | $(STAMP_DIR)
	@echo "==> Rendering manual"
	@cd "$(REPO_ROOT)" && $(QUARTO) render --profile "$(MANUAL_PROFILE)"
	@touch "$@"

manual: $(MANUAL_STAMP)