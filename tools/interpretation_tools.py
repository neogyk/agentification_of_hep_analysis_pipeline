"""
Results Interpretation and Paper Writing Tools for HEP Analysis

Tools for interpreting physics results and generating documentation.
"""

import json
import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path
from smolagents import Tool


class ResultsInterpreterTool(Tool):
    """
    Interpret physics analysis results.
    """
    name = "results_interpreter"
    description = """
    Interprets physics results in context:
    - Compare with Standard Model predictions
    - Check consistency with previous measurements
    - Identify potential new physics signatures
    - Assess significance of deviations
    """
    inputs = {
        "result_type": {
            "type": "string",
            "description": "Type: 'cross_section', 'mass', 'coupling', 'asymmetry', 'limit'"
        },
        "measured_value": {
            "type": "string",
            "description": "JSON with measured value and uncertainty"
        },
        "reference": {
            "type": "string",
            "description": "JSON with SM prediction or reference values"
        }
    }
    output_type = "string"

    def forward(self, result_type: str, measured_value: str, reference: str) -> str:
        try:
            measured = json.loads(measured_value)
            ref = json.loads(reference)
        except json.JSONDecodeError as e:
            return f"JSON parse error: {str(e)}"

        try:
            # Extract values
            obs = measured.get("value", 1.0)
            obs_stat = measured.get("stat_error", measured.get("uncertainty", 0.1))
            obs_syst = measured.get("syst_error", 0)
            obs_total = np.sqrt(obs_stat**2 + obs_syst**2)

            pred = ref.get("value", ref.get("sm_prediction", 1.0))
            pred_unc = ref.get("uncertainty", ref.get("theory_error", 0.1))

            # Calculate compatibility
            diff = obs - pred
            combined_unc = np.sqrt(obs_total**2 + pred_unc**2)
            pull = diff / combined_unc if combined_unc > 0 else 0

            # P-value for consistency
            from scipy import stats
            p_value = 2 * (1 - stats.norm.cdf(abs(pull)))

            interpretation = {
                "result_type": result_type,
                "measured": {
                    "value": obs,
                    "stat_uncertainty": obs_stat,
                    "syst_uncertainty": obs_syst,
                    "total_uncertainty": obs_total
                },
                "reference": {
                    "value": pred,
                    "uncertainty": pred_unc,
                    "source": ref.get("source", "SM prediction")
                },
                "comparison": {
                    "difference": float(diff),
                    "relative_difference": float(diff / pred) if pred != 0 else None,
                    "pull": float(pull),
                    "p_value": float(p_value),
                    "sigma_deviation": float(abs(pull))
                },
                "assessment": self._assess_result(pull, result_type)
            }

            # Add specific interpretation based on type
            if result_type == "cross_section":
                interpretation["signal_strength"] = {
                    "mu": float(obs / pred) if pred != 0 else None,
                    "uncertainty": float(obs_total / pred) if pred != 0 else None
                }
            elif result_type == "mass":
                interpretation["precision"] = {
                    "relative_precision": float(obs_total / obs) if obs != 0 else None,
                    "precision_ppm": float(obs_total / obs * 1e6) if obs != 0 else None
                }

            print(f"Result interpretation: {abs(pull):.2f} sigma from reference")

            return json.dumps(interpretation, indent=2)

        except Exception as e:
            return f"Interpretation error: {str(e)}"

    def _assess_result(self, pull: float, result_type: str) -> Dict:
        """Generate qualitative assessment"""
        abs_pull = abs(pull)

        if abs_pull < 1:
            consistency = "excellent"
            description = "Well consistent with prediction"
        elif abs_pull < 2:
            consistency = "good"
            description = "Consistent within uncertainties"
        elif abs_pull < 3:
            consistency = "moderate"
            description = "Mild tension, may warrant investigation"
        elif abs_pull < 5:
            consistency = "poor"
            description = "Significant deviation, evidence for discrepancy"
        else:
            consistency = "discovery_level"
            description = "Discovery-level significance"

        return {
            "consistency": consistency,
            "description": description,
            "recommendation": self._get_recommendation(abs_pull, result_type)
        }

    def _get_recommendation(self, pull: float, result_type: str) -> str:
        if pull < 2:
            return "Result is consistent with expectations. Consider publishing as measurement."
        elif pull < 3:
            return "Mild tension observed. Verify systematic uncertainties and look for additional data."
        elif pull < 5:
            return "Significant deviation. Investigate possible sources, cross-check with alternative methods."
        else:
            return "Discovery-level significance. Prepare detailed documentation and independent verification."


class SignificanceCalculatorTool(Tool):
    """
    Calculate various significance measures.
    """
    name = "significance_calculator"
    description = """
    Calculates statistical significance:
    - Discovery significance (p-value, sigma)
    - Local and global significance
    - Look-elsewhere effect correction
    - Expected significance from Asimov data
    """
    inputs = {
        "calculation": {
            "type": "string",
            "description": "Type: 'discovery', 'local_global', 'expected', 'asimov'"
        },
        "data": {
            "type": "string",
            "description": "JSON with observation and background"
        },
        "options": {
            "type": "string",
            "description": "JSON with calculation options"
        }
    }
    output_type = "string"

    def forward(self, calculation: str, data: str, options: str = "{}") -> str:
        try:
            data_dict = json.loads(data)
            opts = json.loads(options)
        except json.JSONDecodeError as e:
            return f"JSON parse error: {str(e)}"

        calculators = {
            "discovery": self._discovery_significance,
            "local_global": self._local_global_significance,
            "expected": self._expected_significance,
            "asimov": self._asimov_significance
        }

        if calculation not in calculators:
            return f"Unknown calculation: {calculation}"

        try:
            return calculators[calculation](data_dict, opts)
        except Exception as e:
            return f"Significance calculation error: {str(e)}"

    def _discovery_significance(self, data: Dict, opts: Dict) -> str:
        """Calculate discovery significance"""
        from scipy import stats

        observed = data.get("observed", 20)
        background = data.get("background", 10)
        background_error = data.get("background_error", np.sqrt(background))

        # Simple significance
        if background > 0:
            z_simple = (observed - background) / np.sqrt(background)

            # Profile likelihood approximation
            n, b = observed, background
            if n > b:
                z_pl = np.sqrt(2 * (n * np.log(n/b) - (n - b)))
            else:
                z_pl = 0

            # With background uncertainty
            sigma_b = background_error
            if sigma_b > 0:
                # Approximate formula
                z_with_syst = z_pl / np.sqrt(1 + (sigma_b/b)**2 * n) if b > 0 else 0
            else:
                z_with_syst = z_pl
        else:
            z_simple = z_pl = z_with_syst = 0

        p_value = stats.norm.sf(z_pl)

        result = {
            "observed": observed,
            "expected_background": background,
            "background_uncertainty": background_error,
            "excess": observed - background,
            "significance": {
                "simple_sqrt_b": float(z_simple),
                "profile_likelihood": float(z_pl),
                "with_systematics": float(z_with_syst)
            },
            "p_value": float(p_value),
            "discovery_threshold": {
                "evidence": 3.0,
                "discovery": 5.0
            },
            "is_evidence": z_with_syst >= 3.0,
            "is_discovery": z_with_syst >= 5.0
        }

        return json.dumps(result, indent=2)

    def _local_global_significance(self, data: Dict, opts: Dict) -> str:
        """Calculate local and global significance"""
        from scipy import stats

        local_z = data.get("local_significance", 3.0)
        search_range = data.get("search_range", [100, 1000])  # GeV
        resolution = data.get("resolution", 10)  # GeV

        # Look-elsewhere effect
        n_trials = (search_range[1] - search_range[0]) / resolution

        local_p = stats.norm.sf(local_z)

        # Global p-value (trial factor correction)
        global_p = 1 - (1 - local_p)**n_trials

        # Approximate global significance
        global_z = stats.norm.isf(global_p) if global_p < 0.5 else 0

        result = {
            "local_significance": float(local_z),
            "local_p_value": float(local_p),
            "search_range": search_range,
            "resolution": resolution,
            "effective_trials": float(n_trials),
            "global_p_value": float(global_p),
            "global_significance": float(global_z),
            "trial_factor": float(n_trials),
            "significance_reduction": float(local_z - global_z)
        }

        return json.dumps(result, indent=2)

    def _expected_significance(self, data: Dict, opts: Dict) -> str:
        """Calculate expected significance"""
        signal = data.get("signal", 10)
        background = data.get("background", 100)
        background_error = data.get("background_error", np.sqrt(background))

        # Median expected significance
        expected_obs = signal + background
        z_expected = np.sqrt(2 * ((expected_obs) * np.log(1 + signal/background) - signal)) if background > 0 else 0

        # +/- 1 sigma bands
        z_up = np.sqrt(2 * ((expected_obs + np.sqrt(expected_obs)) * np.log(1 + signal/background) - signal)) if background > 0 else 0
        z_down = np.sqrt(2 * ((max(0, expected_obs - np.sqrt(expected_obs))) * np.log(1 + signal/background) - signal)) if background > 0 else 0

        result = {
            "signal": signal,
            "background": background,
            "expected_observation": expected_obs,
            "expected_significance": {
                "median": float(z_expected),
                "plus_1sigma": float(z_up),
                "minus_1sigma": float(z_down)
            },
            "signal_to_background": float(signal / background) if background > 0 else None
        }

        return json.dumps(result, indent=2)

    def _asimov_significance(self, data: Dict, opts: Dict) -> str:
        """Calculate Asimov significance (expected from perfect data)"""
        signal = data.get("signal", 10)
        background = data.get("background", 100)

        # Asimov formula
        if background > 0:
            z_asimov = np.sqrt(2 * ((signal + background) * np.log(1 + signal/background) - signal))
        else:
            z_asimov = 0

        # Approximate formula for small s/b
        z_approx = signal / np.sqrt(background) if background > 0 else 0

        result = {
            "signal": signal,
            "background": background,
            "asimov_significance": float(z_asimov),
            "approximate_significance": float(z_approx),
            "approximation_valid": signal / background < 0.1 if background > 0 else False
        }

        return json.dumps(result, indent=2)


class LimitCalculatorTool(Tool):
    """
    Calculate upper limits on physics parameters.
    """
    name = "limit_calculator"
    description = """
    Calculates upper limits:
    - CLs limits on signal strength
    - Bayesian credible intervals
    - Cross section limits
    - Branching ratio limits
    """
    inputs = {
        "method": {
            "type": "string",
            "description": "Method: 'cls', 'bayesian', 'feldman_cousins'"
        },
        "data": {
            "type": "string",
            "description": "JSON with observed data and model"
        },
        "options": {
            "type": "string",
            "description": "JSON with calculation options"
        }
    }
    output_type = "string"

    def forward(self, method: str, data: str, options: str = "{}") -> str:
        try:
            data_dict = json.loads(data)
            opts = json.loads(options)
        except json.JSONDecodeError as e:
            return f"JSON parse error: {str(e)}"

        methods = {
            "cls": self._cls_limit,
            "bayesian": self._bayesian_limit,
            "feldman_cousins": self._feldman_cousins_limit
        }

        if method not in methods:
            return f"Unknown method: {method}"

        try:
            return methods[method](data_dict, opts)
        except Exception as e:
            return f"Limit calculation error: {str(e)}"

    def _cls_limit(self, data: Dict, opts: Dict) -> str:
        """CLs upper limit"""
        from scipy import stats

        observed = data.get("observed", 10)
        background = data.get("background", 10)
        signal_efficiency = data.get("signal_efficiency", 1.0)
        cl = opts.get("confidence_level", 0.95)

        # Scan signal strength
        mu_values = np.linspace(0, 50, 500)
        cls_values = []

        for mu in mu_values:
            expected_sb = mu * signal_efficiency + background
            expected_b = background

            # CLs = P(n >= n_obs | s+b) / P(n >= n_obs | b)
            p_sb = 1 - stats.poisson.cdf(observed - 1, expected_sb)
            p_b = 1 - stats.poisson.cdf(observed - 1, expected_b)

            cls_values.append(p_sb / p_b if p_b > 0 else 0)

        # Find crossing
        target = 1 - cl
        for mu, cls in zip(mu_values, cls_values):
            if cls < target:
                observed_limit = mu
                break
        else:
            observed_limit = mu_values[-1]

        # Expected limit (background-only median)
        expected_limit = 0
        for mu in mu_values:
            expected_sb = mu * signal_efficiency + background
            p_sb = 1 - stats.poisson.cdf(int(background) - 1, expected_sb)
            p_b = 1 - stats.poisson.cdf(int(background) - 1, background)
            if p_sb / p_b < target if p_b > 0 else False:
                expected_limit = mu
                break

        result = {
            "method": "CLs",
            "confidence_level": cl,
            "observed": observed,
            "background": background,
            "observed_limit": float(observed_limit),
            "expected_limit": float(expected_limit),
            "cross_section_limit": float(observed_limit / signal_efficiency) if signal_efficiency > 0 else None,
            "limit_ratio": float(observed_limit / expected_limit) if expected_limit > 0 else None
        }

        return json.dumps(result, indent=2)

    def _bayesian_limit(self, data: Dict, opts: Dict) -> str:
        """Bayesian upper limit"""
        from scipy import stats, integrate

        observed = data.get("observed", 10)
        background = data.get("background", 10)
        cl = opts.get("confidence_level", 0.95)
        prior = opts.get("prior", "flat")  # flat or jeffreys

        # Posterior for signal strength with flat prior
        def posterior(s):
            if s < 0:
                return 0
            expected = s + background
            return stats.poisson.pmf(observed, expected)

        # Normalize
        norm, _ = integrate.quad(posterior, 0, 100)

        # Find credible interval
        def cumulative(s):
            result, _ = integrate.quad(posterior, 0, s)
            return result / norm

        # Binary search for upper limit
        s_low, s_high = 0, 100
        while s_high - s_low > 0.01:
            s_mid = (s_low + s_high) / 2
            if cumulative(s_mid) < cl:
                s_low = s_mid
            else:
                s_high = s_mid

        result = {
            "method": "Bayesian",
            "prior": prior,
            "confidence_level": cl,
            "observed": observed,
            "background": background,
            "upper_limit": float(s_high),
            "mode": float(max(0, observed - background)),
            "credible_interval": [0, float(s_high)]
        }

        return json.dumps(result, indent=2)

    def _feldman_cousins_limit(self, data: Dict, opts: Dict) -> str:
        """Feldman-Cousins unified confidence interval"""
        from scipy import stats

        observed = data.get("observed", 5)
        background = data.get("background", 3)
        cl = opts.get("confidence_level", 0.90)

        # Simplified FC for Poisson
        # This is an approximation - full FC requires construction of acceptance regions

        n = observed
        b = background

        # Upper limit from Poisson
        alpha = 1 - cl

        if n == 0:
            upper = -np.log(alpha)
        else:
            # Use chi-squared relation
            upper = stats.chi2.ppf(1 - alpha, 2*(n+1)) / 2 - b

        # Lower limit
        if n > 0:
            lower = max(0, stats.chi2.ppf(alpha, 2*n) / 2 - b)
        else:
            lower = 0

        result = {
            "method": "Feldman-Cousins",
            "confidence_level": cl,
            "observed": observed,
            "background": background,
            "interval": [float(lower), float(upper)],
            "is_upper_limit": lower == 0
        }

        return json.dumps(result, indent=2)


class PaperWriterTool(Tool):
    """
    Generate physics paper sections and documentation.
    """
    name = "paper_writer"
    description = """
    Generates physics paper sections:
    - Abstract summarizing analysis and results
    - Introduction with physics motivation
    - Analysis description sections
    - Results tables and figures captions
    - Conclusion and outlook
    """
    inputs = {
        "section": {
            "type": "string",
            "description": "Section: 'abstract', 'introduction', 'analysis', 'results', 'conclusion', 'full'"
        },
        "analysis_info": {
            "type": "string",
            "description": "JSON with analysis information and results"
        },
        "style": {
            "type": "string",
            "description": "Style: 'CMS', 'ATLAS', 'PRL', 'generic'"
        }
    }
    output_type = "string"

    def forward(self, section: str, analysis_info: str, style: str = "generic") -> str:
        try:
            info = json.loads(analysis_info)
        except json.JSONDecodeError as e:
            return f"JSON parse error: {str(e)}"

        generators = {
            "abstract": self._generate_abstract,
            "introduction": self._generate_introduction,
            "analysis": self._generate_analysis,
            "results": self._generate_results,
            "conclusion": self._generate_conclusion,
            "full": self._generate_full
        }

        if section not in generators:
            return f"Unknown section: {section}"

        try:
            return generators[section](info, style)
        except Exception as e:
            return f"Paper writing error: {str(e)}"

    def _generate_abstract(self, info: Dict, style: str) -> str:
        """Generate abstract"""
        process = info.get("process", "particle production")
        channel = info.get("channel", "final state")
        energy = info.get("sqrt_s", 13)
        luminosity = info.get("luminosity", 139)
        result_type = info.get("result_type", "cross_section")
        measured = info.get("measured_value", 1.0)
        uncertainty = info.get("total_uncertainty", 0.1)
        sm_prediction = info.get("sm_prediction", 1.0)

        abstract = f"""A measurement of {process} in the {channel} channel is presented using proton-proton collision data at $\\sqrt{{s}} = {energy}$ TeV, corresponding to an integrated luminosity of {luminosity} fb$^{{-1}}$. """

        if result_type == "cross_section":
            abstract += f"""The measured cross section is $\\sigma = {measured:.2f} \\pm {uncertainty:.2f}$ pb, in agreement with the Standard Model prediction of ${sm_prediction:.2f}$ pb. """
        elif result_type == "mass":
            abstract += f"""The measured mass is $m = {measured:.3f} \\pm {uncertainty:.3f}$ GeV, consistent with previous measurements. """
        elif result_type == "limit":
            abstract += f"""No significant excess is observed, and upper limits are set on the production cross section. """

        abstract += """The results are consistent with Standard Model expectations."""

        return abstract

    def _generate_introduction(self, info: Dict, style: str) -> str:
        """Generate introduction section"""
        process = info.get("process", "the process under study")
        physics_motivation = info.get("motivation", "testing the Standard Model")

        intro = f"""\\section{{Introduction}}

The Standard Model (SM) of particle physics has been remarkably successful in describing fundamental interactions. However, testing its predictions with increasing precision remains crucial for {physics_motivation}.

This analysis presents a measurement of {process} using data collected by the experiment. The measurement provides a stringent test of SM predictions and sensitivity to potential new physics contributions.

Previous measurements of this process have been performed by various experiments, and the results presented here supersede/complement those findings with improved precision.

The paper is organized as follows: Section 2 describes the detector and data samples, Section 3 presents the event selection and reconstruction, Section 4 discusses systematic uncertainties, and Section 5 presents the results and conclusions.
"""
        return intro

    def _generate_analysis(self, info: Dict, style: str) -> str:
        """Generate analysis description"""
        channel = info.get("channel", "lepton+jets")
        selection = info.get("selection_criteria", [])
        n_events = info.get("n_selected_events", 10000)

        analysis = f"""\\section{{Analysis Strategy}}

\\subsection{{Event Selection}}

Events are selected using the following criteria optimized for the {channel} channel:

\\begin{{itemize}}
"""
        for cut in selection:
            analysis += f"\\item {cut}\n"

        if not selection:
            analysis += """\\item Trigger requirements
\\item Lepton identification and isolation
\\item Jet multiplicity and kinematics
\\item Missing transverse energy requirements
"""

        analysis += f"""\\end{{itemize}}

After all selection requirements, {n_events:,} events are selected in data.

\\subsection{{Background Estimation}}

The main backgrounds are estimated using data-driven techniques where possible, supplemented by Monte Carlo simulation for smaller contributions.

\\subsection{{Signal Extraction}}

The signal yield is extracted using a maximum likelihood fit to the discriminating observable.
"""
        return analysis

    def _generate_results(self, info: Dict, style: str) -> str:
        """Generate results section"""
        measured = info.get("measured_value", 1.0)
        stat_err = info.get("stat_uncertainty", 0.05)
        syst_err = info.get("syst_uncertainty", 0.08)
        total_err = info.get("total_uncertainty", np.sqrt(stat_err**2 + syst_err**2))
        sm_pred = info.get("sm_prediction", 1.0)
        sm_err = info.get("sm_uncertainty", 0.1)

        results = f"""\\section{{Results}}

The measured value is:

\\begin{{equation}}
\\sigma = {measured:.3f} \\pm {stat_err:.3f} \\text{{ (stat)}} \\pm {syst_err:.3f} \\text{{ (syst)}} \\text{{ pb}}
\\end{{equation}}

corresponding to a total uncertainty of $\\pm {total_err:.3f}$ pb ({total_err/measured*100:.1f}\\%).

The result is compared with the Standard Model prediction of ${sm_pred:.3f} \\pm {sm_err:.3f}$ pb, showing good agreement within uncertainties.

\\begin{{table}}[h]
\\centering
\\caption{{Summary of systematic uncertainties.}}
\\begin{{tabular}}{{lc}}
\\hline
Source & Uncertainty (\\%) \\\\
\\hline
"""
        systematics = info.get("systematics", {
            "Jet energy scale": 3.0,
            "Luminosity": 1.7,
            "Theory": 5.0
        })

        for source, value in systematics.items():
            results += f"{source} & {value:.1f} \\\\\n"

        results += f"""\\hline
Total systematic & {syst_err/measured*100:.1f} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
        return results

    def _generate_conclusion(self, info: Dict, style: str) -> str:
        """Generate conclusion"""
        measured = info.get("measured_value", 1.0)
        process = info.get("process", "the studied process")

        conclusion = f"""\\section{{Conclusion}}

A measurement of {process} has been presented using the full Run 2 dataset.

The measured value of ${measured:.3f}$ is consistent with Standard Model predictions, demonstrating the continued success of the SM in describing particle physics processes.

These results provide important input for precision tests of the Standard Model and constrain models of physics beyond the SM.

Future measurements with the High-Luminosity LHC will further improve the precision of these measurements.
"""
        return conclusion

    def _generate_full(self, info: Dict, style: str) -> str:
        """Generate full paper draft"""
        full = "\\documentclass{article}\n\\begin{document}\n\n"
        full += "\\title{" + info.get("title", "Physics Measurement") + "}\n"
        full += "\\author{The Collaboration}\n"
        full += "\\maketitle\n\n"
        full += "\\begin{abstract}\n"
        full += self._generate_abstract(info, style)
        full += "\n\\end{abstract}\n\n"
        full += self._generate_introduction(info, style)
        full += self._generate_analysis(info, style)
        full += self._generate_results(info, style)
        full += self._generate_conclusion(info, style)
        full += "\n\\end{document}"

        return full
