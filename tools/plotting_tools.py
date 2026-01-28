"""
Plotting and Visualization Tools for HEP Analysis

Tools for creating publication-quality physics plots.
"""

import json
import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path
from smolagents import Tool


class HistogramPlotTool(Tool):
    """
    Create histogram plots with various styles.
    """
    name = "histogram_plot"
    description = """
    Creates histogram plots with experiment-specific styles:
    - CMS, ATLAS, LHCb styling
    - Data points with error bars
    - Stacked histograms
    - Ratio panels
    - Statistical and systematic uncertainty bands
    """
    inputs = {
        "histogram_data": {
            "type": "string",
            "description": "JSON with histogram data (bin_edges, values, errors)"
        },
        "style": {
            "type": "string",
            "description": "Style: 'CMS', 'ATLAS', 'LHCb', 'default'"
        },
        "options": {
            "type": "string",
            "description": "JSON with plot options (xlabel, ylabel, title, etc.)"
        },
        "output_path": {
            "type": "string",
            "description": "Path to save the plot (without extension)"
        }
    }
    output_type = "string"

    def forward(self, histogram_data: str, style: str = "CMS",
                options: str = "{}", output_path: str = "./plot") -> str:
        try:
            data = json.loads(histogram_data)
            opts = json.loads(options)
        except json.JSONDecodeError as e:
            return f"JSON parse error: {str(e)}"

        try:
            import matplotlib.pyplot as plt
            import matplotlib as mpl

            # Try to import mplhep
            try:
                import mplhep as hep
                has_mplhep = True
            except ImportError:
                has_mplhep = False
                print("mplhep not available, using default style")

            # Set style
            if has_mplhep:
                style_map = {
                    "CMS": hep.style.CMS,
                    "ATLAS": hep.style.ATLAS,
                    "LHCb": hep.style.LHCb,
                    "default": "default"
                }
                if style in style_map and style != "default":
                    hep.style.use(style_map[style])
            else:
                plt.style.use('default')
                mpl.rcParams.update({
                    'font.size': 12,
                    'axes.labelsize': 14,
                    'axes.titlesize': 14,
                    'legend.fontsize': 10,
                    'xtick.labelsize': 12,
                    'ytick.labelsize': 12,
                })

            # Extract data
            edges = np.array(data.get("bin_edges", data.get("edges", [])))
            values = np.array(data.get("values", data.get("counts", [])))
            errors = np.array(data.get("errors", np.sqrt(np.maximum(values, 1))))

            if len(edges) == 0:
                # Generate sample histogram
                sample_data = np.random.normal(91.2, 2.5, 10000)
                values, edges = np.histogram(sample_data, bins=60, range=(60, 120))
                errors = np.sqrt(np.maximum(values, 1))

            centers = (edges[:-1] + edges[1:]) / 2

            # Create figure
            include_ratio = opts.get("include_ratio", False)
            if include_ratio:
                fig, (ax, rax) = plt.subplots(
                    2, 1, figsize=(10, 10),
                    gridspec_kw={"height_ratios": [3, 1]},
                    sharex=True
                )
                fig.subplots_adjust(hspace=0.05)
            else:
                fig, ax = plt.subplots(figsize=(10, 8))

            # Plot histogram
            plot_type = opts.get("plot_type", "errorbar")

            if plot_type == "errorbar":
                ax.errorbar(centers, values, yerr=errors, fmt='ko', markersize=4,
                           label=opts.get("label", "Data"))
            elif plot_type == "step":
                ax.stairs(values, edges, label=opts.get("label", "Data"))
            elif plot_type == "fill":
                ax.stairs(values, edges, fill=True, alpha=0.7,
                         label=opts.get("label", "MC"))

            # Add uncertainty band if provided
            if "syst_errors" in data:
                syst_up = values + np.array(data["syst_errors"])
                syst_down = values - np.array(data["syst_errors"])
                ax.fill_between(centers, syst_down, syst_up, alpha=0.3,
                               color='gray', label='Syst. Unc.')

            # Labels
            ax.set_xlabel(opts.get("xlabel", "Mass [GeV]"))
            ax.set_ylabel(opts.get("ylabel", "Events"))
            if opts.get("title"):
                ax.set_title(opts.get("title"))

            ax.legend(loc=opts.get("legend_loc", "best"))

            # Set limits
            if opts.get("xlim"):
                ax.set_xlim(opts["xlim"])
            if opts.get("ylim"):
                ax.set_ylim(opts["ylim"])
            if opts.get("log_y", False):
                ax.set_yscale('log')

            # Add experiment label
            if has_mplhep and style in ["CMS", "ATLAS", "LHCb"]:
                lumi = opts.get("luminosity", 139)
                if style == "CMS":
                    hep.cms.label(ax=ax, data=True, lumi=lumi)
                elif style == "ATLAS":
                    hep.atlas.label(ax=ax, data=True, lumi=lumi)

            # Ratio panel
            if include_ratio and "expected" in data:
                expected = np.array(data["expected"])
                ratio = values / np.where(expected > 0, expected, 1)
                ratio_err = errors / np.where(expected > 0, expected, 1)

                rax.errorbar(centers, ratio, yerr=ratio_err, fmt='ko', markersize=4)
                rax.axhline(1, color='gray', linestyle='--')
                rax.set_ylabel("Data / MC")
                rax.set_ylim(0.5, 1.5)
                rax.set_xlabel(opts.get("xlabel", "Mass [GeV]"))

            # Save plot
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            formats = opts.get("formats", ["pdf", "png"])
            saved_files = []
            for fmt in formats:
                filepath = f"{output_path}.{fmt}"
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                saved_files.append(filepath)
                print(f"Saved: {filepath}")

            plt.close(fig)

            result = {
                "status": "success",
                "output_files": saved_files,
                "style": style,
                "n_bins": len(values),
                "integral": float(np.sum(values))
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            return f"Plotting error: {str(e)}"


class FitResultPlotTool(Tool):
    """
    Plot fit results with residuals.
    """
    name = "fit_result_plot"
    description = """
    Creates fit result plots with:
    - Data points with errors
    - Fit curve overlay
    - Individual components (signal, background)
    - Pull/residual distribution
    - Parameter values annotation
    """
    inputs = {
        "fit_data": {
            "type": "string",
            "description": "JSON with fit data (histogram, fit_result, model_values)"
        },
        "style": {
            "type": "string",
            "description": "Plot style: 'CMS', 'ATLAS', 'default'"
        },
        "output_path": {
            "type": "string",
            "description": "Output file path"
        }
    }
    output_type = "string"

    def forward(self, fit_data: str, style: str = "CMS",
                output_path: str = "./fit_plot") -> str:
        try:
            data = json.loads(fit_data)
        except json.JSONDecodeError as e:
            return f"JSON parse error: {str(e)}"

        try:
            import matplotlib.pyplot as plt

            try:
                import mplhep as hep
                has_mplhep = True
                if style in ["CMS", "ATLAS"]:
                    hep.style.use(style)
            except ImportError:
                has_mplhep = False

            # Extract data
            edges = np.array(data.get("bin_edges", []))
            values = np.array(data.get("values", []))
            errors = np.array(data.get("errors", np.sqrt(np.maximum(values, 1))))
            fit_values = np.array(data.get("fit_values", []))

            if len(edges) == 0:
                # Generate sample
                x = np.linspace(60, 120, 61)
                edges = x
                centers = (x[:-1] + x[1:]) / 2
                true_sig = 1000 * np.exp(-0.5 * ((centers - 91.2) / 2.5)**2)
                true_bkg = 200 * np.exp(-0.02 * (centers - 60))
                values = np.random.poisson(true_sig + true_bkg).astype(float)
                errors = np.sqrt(np.maximum(values, 1))
                fit_values = true_sig + true_bkg

            centers = (edges[:-1] + edges[1:]) / 2

            if len(fit_values) == 0:
                fit_values = values  # Fallback

            # Create figure with ratio panel
            fig, (ax, rax) = plt.subplots(
                2, 1, figsize=(10, 10),
                gridspec_kw={"height_ratios": [3, 1]},
                sharex=True
            )
            fig.subplots_adjust(hspace=0.05)

            # Main panel - data points
            ax.errorbar(centers, values, yerr=errors, fmt='ko', markersize=4, label='Data')

            # Fit curve
            ax.plot(centers, fit_values, 'r-', linewidth=2, label='Fit')

            # Signal and background components if available
            if "signal_component" in data:
                ax.plot(centers, data["signal_component"], 'b--', linewidth=1.5, label='Signal')
            if "background_component" in data:
                ax.plot(centers, data["background_component"], 'g--', linewidth=1.5, label='Background')

            ax.set_ylabel("Events / bin")
            ax.legend(loc='best')

            # Add experiment label
            if has_mplhep:
                if style == "CMS":
                    hep.cms.label(ax=ax, data=True, lumi=data.get("luminosity", 139))
                elif style == "ATLAS":
                    hep.atlas.label(ax=ax, data=True, lumi=data.get("luminosity", 139))

            # Add fit parameters annotation
            if "parameters" in data:
                param_text = "Fit Parameters:\n"
                for name, val in data["parameters"].items():
                    if isinstance(val, dict):
                        param_text += f"  {name}: {val['value']:.3f} ± {val['error']:.3f}\n"
                    else:
                        param_text += f"  {name}: {val:.3f}\n"
                ax.text(0.02, 0.98, param_text, transform=ax.transAxes,
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Pull panel
            pulls = (values - fit_values) / errors
            rax.errorbar(centers, pulls, yerr=1, fmt='ko', markersize=4)
            rax.axhline(0, color='gray', linestyle='--')
            rax.axhline(2, color='red', linestyle=':', alpha=0.5)
            rax.axhline(-2, color='red', linestyle=':', alpha=0.5)
            rax.set_ylabel("Pull")
            rax.set_xlabel(data.get("xlabel", "Mass [GeV]"))
            rax.set_ylim(-4, 4)

            # Save
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            for fmt in ["pdf", "png"]:
                fig.savefig(f"{output_path}.{fmt}", dpi=300, bbox_inches='tight')
                print(f"Saved: {output_path}.{fmt}")

            plt.close(fig)

            return json.dumps({
                "status": "success",
                "output_files": [f"{output_path}.pdf", f"{output_path}.png"],
                "chi2_ndf": float(np.sum(pulls**2) / len(pulls))
            }, indent=2)

        except Exception as e:
            return f"Fit plotting error: {str(e)}"


class ComparisonPlotTool(Tool):
    """
    Create comparison plots (data vs MC, different samples).
    """
    name = "comparison_plot"
    description = """
    Creates comparison plots:
    - Data vs MC with stacked backgrounds
    - Multiple distributions overlay
    - Shape comparisons (normalized)
    - Systematic variation comparisons
    """
    inputs = {
        "datasets": {
            "type": "string",
            "description": "JSON array of datasets to compare"
        },
        "comparison_type": {
            "type": "string",
            "description": "Type: 'stack', 'overlay', 'shape', 'ratio'"
        },
        "options": {
            "type": "string",
            "description": "JSON with plot options"
        },
        "output_path": {
            "type": "string",
            "description": "Output file path"
        }
    }
    output_type = "string"

    def forward(self, datasets: str, comparison_type: str = "stack",
                options: str = "{}", output_path: str = "./comparison") -> str:
        try:
            data_list = json.loads(datasets)
            opts = json.loads(options)
        except json.JSONDecodeError as e:
            return f"JSON parse error: {str(e)}"

        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors

            # Try mplhep
            try:
                import mplhep as hep
                hep.style.use(opts.get("style", "CMS"))
                has_mplhep = True
            except ImportError:
                has_mplhep = False

            # Default colors
            colors = list(mcolors.TABLEAU_COLORS.values())

            # Create figure
            include_ratio = opts.get("include_ratio", True) and comparison_type in ["stack", "overlay"]

            if include_ratio:
                fig, (ax, rax) = plt.subplots(
                    2, 1, figsize=(10, 10),
                    gridspec_kw={"height_ratios": [3, 1]},
                    sharex=True
                )
            else:
                fig, ax = plt.subplots(figsize=(10, 8))

            # Process datasets
            if len(data_list) == 0:
                # Generate sample data
                centers = np.linspace(62.5, 117.5, 55)
                data_list = [
                    {"label": "Data", "values": np.random.poisson(500, 55).tolist(),
                     "errors": np.sqrt(np.random.poisson(500, 55)).tolist(), "type": "data"},
                    {"label": "Z+jets", "values": (300 * np.exp(-0.5*((centers-91)/3)**2)).tolist(), "type": "mc"},
                    {"label": "tt", "values": (50 * np.ones(55)).tolist(), "type": "mc"},
                    {"label": "WW", "values": (30 * np.ones(55)).tolist(), "type": "mc"},
                ]

            edges = np.array(opts.get("bin_edges", np.linspace(60, 120, 56)))
            centers = (edges[:-1] + edges[1:]) / 2

            if comparison_type == "stack":
                # Separate data and MC
                data_sample = None
                mc_samples = []

                for i, ds in enumerate(data_list):
                    if ds.get("type") == "data":
                        data_sample = ds
                    else:
                        mc_samples.append(ds)

                # Stack MC
                mc_values = [np.array(ds.get("values", np.zeros(len(centers)))) for ds in mc_samples]
                mc_labels = [ds.get("label", f"MC_{i}") for i, ds in enumerate(mc_samples)]
                mc_colors = [colors[i % len(colors)] for i in range(len(mc_samples))]

                if has_mplhep:
                    hep.histplot(mc_values, bins=edges, ax=ax, label=mc_labels,
                                color=mc_colors, stack=True, histtype="fill")
                else:
                    bottom = np.zeros(len(centers))
                    for vals, label, color in zip(mc_values, mc_labels, mc_colors):
                        ax.bar(centers, vals, width=np.diff(edges), bottom=bottom,
                              label=label, color=color, alpha=0.7)
                        bottom += vals

                # Plot data
                if data_sample:
                    data_vals = np.array(data_sample.get("values", []))
                    data_errs = np.array(data_sample.get("errors", np.sqrt(data_vals)))
                    ax.errorbar(centers, data_vals, yerr=data_errs, fmt='ko',
                               markersize=4, label=data_sample.get("label", "Data"))

                    # Ratio
                    if include_ratio:
                        mc_total = np.sum(mc_values, axis=0)
                        ratio = data_vals / np.where(mc_total > 0, mc_total, 1)
                        ratio_err = data_errs / np.where(mc_total > 0, mc_total, 1)
                        rax.errorbar(centers, ratio, yerr=ratio_err, fmt='ko', markersize=4)
                        rax.axhline(1, color='gray', linestyle='--')
                        rax.set_ylabel("Data / MC")
                        rax.set_ylim(0.5, 1.5)

            elif comparison_type == "overlay":
                for i, ds in enumerate(data_list):
                    vals = np.array(ds.get("values", []))
                    errs = np.array(ds.get("errors", np.sqrt(np.maximum(vals, 1))))
                    label = ds.get("label", f"Dataset_{i}")
                    color = colors[i % len(colors)]

                    if ds.get("type") == "data":
                        ax.errorbar(centers[:len(vals)], vals, yerr=errs, fmt='o',
                                   color=color, markersize=4, label=label)
                    else:
                        ax.stairs(vals, edges[:len(vals)+1], color=color, label=label, linewidth=2)

            elif comparison_type == "shape":
                # Normalize all distributions
                for i, ds in enumerate(data_list):
                    vals = np.array(ds.get("values", []))
                    if np.sum(vals) > 0:
                        vals = vals / np.sum(vals)
                    label = ds.get("label", f"Dataset_{i}")
                    color = colors[i % len(colors)]
                    ax.stairs(vals, edges[:len(vals)+1], color=color, label=label, linewidth=2)

                ax.set_ylabel("Normalized")

            # Labels and styling
            ax.set_xlabel(opts.get("xlabel", "Mass [GeV]"))
            if comparison_type != "shape":
                ax.set_ylabel(opts.get("ylabel", "Events"))
            ax.legend(loc=opts.get("legend_loc", "best"))

            if opts.get("log_y", False):
                ax.set_yscale('log')

            if has_mplhep:
                hep.cms.label(ax=ax, data=True, lumi=opts.get("luminosity", 139))

            if include_ratio:
                rax.set_xlabel(opts.get("xlabel", "Mass [GeV]"))

            # Save
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            for fmt in ["pdf", "png"]:
                fig.savefig(f"{output_path}.{fmt}", dpi=300, bbox_inches='tight')
                print(f"Saved: {output_path}.{fmt}")

            plt.close(fig)

            return json.dumps({
                "status": "success",
                "output_files": [f"{output_path}.pdf", f"{output_path}.png"],
                "comparison_type": comparison_type,
                "n_datasets": len(data_list)
            }, indent=2)

        except Exception as e:
            return f"Comparison plotting error: {str(e)}"


class PublicationPlotTool(Tool):
    """
    Create publication-quality plots with all required elements.
    """
    name = "publication_plot"
    description = """
    Creates publication-ready plots with:
    - Proper fonts and sizing
    - Experiment labels and luminosity
    - Legend with proper formatting
    - Systematic uncertainty bands
    - Multiple panels (if needed)
    - Vector graphics output (PDF)
    """
    inputs = {
        "plot_config": {
            "type": "string",
            "description": "JSON with complete plot configuration"
        },
        "output_path": {
            "type": "string",
            "description": "Output path for the plot"
        }
    }
    output_type = "string"

    def forward(self, plot_config: str, output_path: str = "./publication_plot") -> str:
        try:
            config = json.loads(plot_config)
        except json.JSONDecodeError as e:
            return f"JSON parse error: {str(e)}"

        try:
            import matplotlib.pyplot as plt
            import matplotlib as mpl

            # Publication-quality settings
            mpl.rcParams.update({
                'font.family': 'serif',
                'font.size': 14,
                'axes.labelsize': 16,
                'axes.titlesize': 16,
                'legend.fontsize': 12,
                'xtick.labelsize': 14,
                'ytick.labelsize': 14,
                'xtick.direction': 'in',
                'ytick.direction': 'in',
                'xtick.top': True,
                'ytick.right': True,
                'axes.linewidth': 1.2,
                'lines.linewidth': 1.5,
                'figure.dpi': 150,
            })

            # Try to use mplhep
            try:
                import mplhep as hep
                style = config.get("style", "CMS")
                if style in ["CMS", "ATLAS", "LHCb"]:
                    hep.style.use(style)
                has_mplhep = True
            except ImportError:
                has_mplhep = False

            # Figure setup
            figsize = config.get("figsize", [10, 8])
            n_panels = config.get("n_panels", 1)

            if n_panels == 2:
                fig, axes = plt.subplots(2, 1, figsize=figsize,
                                        gridspec_kw={"height_ratios": config.get("height_ratios", [3, 1])},
                                        sharex=True)
                fig.subplots_adjust(hspace=0.05)
            else:
                fig, axes = plt.subplots(figsize=figsize)
                axes = [axes]

            ax = axes[0] if isinstance(axes, (list, np.ndarray)) else axes

            # Plot data
            data = config.get("data", {})

            # Main histogram/data
            if "histogram" in data:
                hist = data["histogram"]
                edges = np.array(hist.get("edges", np.linspace(60, 120, 61)))
                values = np.array(hist.get("values", np.random.poisson(100, 60)))
                errors = np.array(hist.get("errors", np.sqrt(values)))
                centers = (edges[:-1] + edges[1:]) / 2

                ax.errorbar(centers, values, yerr=errors, fmt='ko', markersize=5,
                           label=hist.get("label", "Data"), capsize=0)

            # Fit or model curve
            if "model" in data:
                model = data["model"]
                x = np.array(model.get("x", centers if 'centers' in dir() else np.linspace(60, 120, 200)))
                y = np.array(model.get("y", 100 * np.exp(-0.5*((x-91)/3)**2)))
                ax.plot(x, y, 'r-', linewidth=2, label=model.get("label", "Fit"))

            # Uncertainty band
            if "uncertainty_band" in data:
                band = data["uncertainty_band"]
                x = np.array(band.get("x", centers if 'centers' in dir() else np.linspace(60, 120, 60)))
                y_up = np.array(band.get("up", np.zeros(len(x))))
                y_down = np.array(band.get("down", np.zeros(len(x))))
                ax.fill_between(x, y_down, y_up, alpha=0.3, color='gray',
                               label=band.get("label", "Uncertainty"))

            # Styling
            ax.set_xlabel(config.get("xlabel", ""))
            ax.set_ylabel(config.get("ylabel", "Events"))

            if config.get("log_y", False):
                ax.set_yscale('log')

            ax.legend(loc=config.get("legend_loc", "best"), frameon=config.get("legend_frame", True))

            # Experiment label
            if has_mplhep:
                lumi = config.get("luminosity", 139)
                label_options = config.get("label_options", {})

                if style == "CMS":
                    hep.cms.label(
                        ax=ax,
                        data=label_options.get("data", True),
                        lumi=lumi,
                        label=label_options.get("label", "")
                    )
                elif style == "ATLAS":
                    hep.atlas.label(
                        ax=ax,
                        data=label_options.get("data", True),
                        lumi=lumi,
                        label=label_options.get("label", "Internal")
                    )

            # Add text annotations
            for annotation in config.get("annotations", []):
                ax.text(
                    annotation.get("x", 0.05),
                    annotation.get("y", 0.95),
                    annotation.get("text", ""),
                    transform=ax.transAxes,
                    fontsize=annotation.get("fontsize", 12),
                    verticalalignment=annotation.get("va", "top"),
                    horizontalalignment=annotation.get("ha", "left")
                )

            # Ratio panel
            if n_panels == 2 and len(axes) > 1:
                rax = axes[1]
                ratio_data = config.get("ratio", {})

                x = np.array(ratio_data.get("x", centers if 'centers' in dir() else np.linspace(60, 120, 60)))
                y = np.array(ratio_data.get("y", np.ones(len(x))))
                yerr = np.array(ratio_data.get("yerr", 0.1 * np.ones(len(x))))

                rax.errorbar(x, y, yerr=yerr, fmt='ko', markersize=4)
                rax.axhline(1, color='gray', linestyle='--', linewidth=1)
                rax.set_ylabel(ratio_data.get("ylabel", "Ratio"))
                rax.set_xlabel(config.get("xlabel", ""))
                rax.set_ylim(ratio_data.get("ylim", [0.8, 1.2]))

            # Save with multiple formats
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            formats = config.get("formats", ["pdf", "png"])
            saved = []
            for fmt in formats:
                filepath = f"{output_path}.{fmt}"
                fig.savefig(filepath, dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                saved.append(filepath)
                print(f"Saved: {filepath}")

            plt.close(fig)

            return json.dumps({
                "status": "success",
                "output_files": saved,
                "style": config.get("style", "default"),
                "publication_ready": True
            }, indent=2)

        except Exception as e:
            return f"Publication plot error: {str(e)}"
