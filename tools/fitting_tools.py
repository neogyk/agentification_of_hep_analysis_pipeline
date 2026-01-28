"""
Fitting and Statistical Analysis Tools for HEP Analysis

Tools for histogram fitting, background modeling, and statistical inference.
"""

import json
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from smolagents import Tool


class HistogramFitterTool(Tool):
    """
    Fit histograms with various models.
    """
    name = "histogram_fitter"
    description = """
    Fits histograms using maximum likelihood or chi-square methods.
    Supported models:
    - Gaussian, Double Gaussian
    - Breit-Wigner, Voigtian
    - Crystal Ball, Double Crystal Ball
    - Exponential, Polynomial backgrounds
    - Signal + Background composite models
    """
    inputs = {
        "histogram": {
            "type": "string",
            "description": "JSON with histogram data (bin_edges, values, errors)"
        },
        "model": {
            "type": "string",
            "description": "Model: 'gaussian', 'breit_wigner', 'crystal_ball', 'exponential', 'polynomial', 'signal_plus_background'"
        },
        "initial_params": {
            "type": "string",
            "description": "JSON with initial parameter values"
        },
        "fit_range": {
            "type": "string",
            "description": "JSON array [low, high] for fit range (optional)"
        }
    }
    output_type = "string"

    def forward(self, histogram: str, model: str, initial_params: str = "{}",
                fit_range: str = "null") -> str:
        try:
            hist_data = json.loads(histogram)
            params = json.loads(initial_params)
            range_data = json.loads(fit_range) if fit_range != "null" else None
        except json.JSONDecodeError as e:
            return f"JSON parse error: {str(e)}"

        try:
            from scipy import optimize
            from scipy import stats

            # Extract histogram data
            edges = np.array(hist_data.get("bin_edges", hist_data.get("edges", [])))
            values = np.array(hist_data.get("values", hist_data.get("counts", [])))
            errors = np.array(hist_data.get("errors", np.sqrt(np.maximum(values, 1))))

            if len(edges) == 0 or len(values) == 0:
                # Generate sample histogram
                data = np.random.normal(91.2, 2.5, 10000)
                values, edges = np.histogram(data, bins=60, range=(60, 120))
                errors = np.sqrt(np.maximum(values, 1))

            centers = (edges[:-1] + edges[1:]) / 2
            bin_width = edges[1] - edges[0]

            # Apply fit range
            if range_data:
                mask = (centers >= range_data[0]) & (centers <= range_data[1])
                centers = centers[mask]
                values = values[mask]
                errors = errors[mask]

            # Get model function
            model_func = self._get_model(model)

            # Set default parameters if not provided
            if not params:
                params = self._get_default_params(model, centers, values)

            # Perform fit
            def chi2(p):
                pred = model_func(centers, *p) * bin_width
                return np.sum(((values - pred) / (errors + 1e-10))**2)

            param_names = list(params.keys())
            p0 = list(params.values())

            result = optimize.minimize(chi2, p0, method='L-BFGS-B')

            # Calculate uncertainties using Hessian
            try:
                hess_inv = result.hess_inv.todense() if hasattr(result.hess_inv, 'todense') else result.hess_inv
                param_errors = np.sqrt(np.diag(hess_inv))
            except:
                param_errors = np.zeros(len(p0))

            # Calculate fit quality
            chi2_val = result.fun
            ndf = len(values) - len(p0)
            chi2_ndf = chi2_val / ndf if ndf > 0 else 0
            p_value = 1 - stats.chi2.cdf(chi2_val, ndf) if ndf > 0 else 0

            fit_result = {
                "model": model,
                "converged": result.success,
                "parameters": {
                    name: {
                        "value": float(result.x[i]),
                        "error": float(param_errors[i]) if i < len(param_errors) else 0
                    }
                    for i, name in enumerate(param_names)
                },
                "chi2": float(chi2_val),
                "ndf": int(ndf),
                "chi2_ndf": float(chi2_ndf),
                "p_value": float(p_value),
                "fit_range": [float(centers[0]), float(centers[-1])]
            }

            print(f"Fit completed: chi2/ndf = {chi2_ndf:.2f}, p-value = {p_value:.3f}")
            for name in param_names:
                p = fit_result["parameters"][name]
                print(f"  {name}: {p['value']:.4f} +/- {p['error']:.4f}")

            return json.dumps(fit_result, indent=2)

        except Exception as e:
            return f"Fitting error: {str(e)}"

    def _get_model(self, model_name: str):
        """Return model function"""
        models = {
            "gaussian": lambda x, mu, sigma, norm: norm * np.exp(-0.5 * ((x - mu) / sigma)**2),
            "breit_wigner": lambda x, m, gamma, norm: norm * gamma**2 / ((x - m)**2 + gamma**2/4),
            "exponential": lambda x, slope, norm: norm * np.exp(slope * x),
            "polynomial": lambda x, a0, a1, a2: a0 + a1*x + a2*x**2,
            "crystal_ball": self._crystal_ball,
            "signal_plus_background": self._signal_plus_background
        }
        return models.get(model_name, models["gaussian"])

    def _crystal_ball(self, x, mu, sigma, alpha, n, norm):
        """Crystal Ball function"""
        t = (x - mu) / sigma
        abs_alpha = np.abs(alpha)

        a = (n / abs_alpha)**n * np.exp(-0.5 * abs_alpha**2)
        b = n / abs_alpha - abs_alpha

        result = np.where(
            t > -alpha,
            np.exp(-0.5 * t**2),
            a * (b - t)**(-n)
        )
        return norm * result

    def _signal_plus_background(self, x, mu, sigma, sig_norm, slope, bkg_norm):
        """Signal (Gaussian) + Background (Exponential)"""
        signal = sig_norm * np.exp(-0.5 * ((x - mu) / sigma)**2)
        background = bkg_norm * np.exp(slope * (x - mu))
        return signal + background

    def _get_default_params(self, model: str, x: np.ndarray, y: np.ndarray) -> Dict:
        """Get default parameters based on data"""
        x_mean = np.average(x, weights=y)
        x_std = np.sqrt(np.average((x - x_mean)**2, weights=y))
        y_max = np.max(y)

        defaults = {
            "gaussian": {"mu": x_mean, "sigma": x_std, "norm": y_max},
            "breit_wigner": {"m": x_mean, "gamma": x_std * 2, "norm": y_max},
            "exponential": {"slope": -0.01, "norm": y_max},
            "polynomial": {"a0": np.mean(y), "a1": 0, "a2": 0},
            "crystal_ball": {"mu": x_mean, "sigma": x_std, "alpha": 1.5, "n": 2, "norm": y_max},
            "signal_plus_background": {"mu": x_mean, "sigma": x_std, "sig_norm": y_max * 0.8,
                                       "slope": -0.01, "bkg_norm": y_max * 0.2}
        }
        return defaults.get(model, defaults["gaussian"])


class BackgroundModelTool(Tool):
    """
    Model and estimate backgrounds.
    """
    name = "background_model"
    description = """
    Creates background models and estimates:
    - Fit background in sideband regions
    - Extrapolate to signal region
    - Data-driven background estimation
    - Template fitting
    """
    inputs = {
        "data": {
            "type": "string",
            "description": "JSON histogram data"
        },
        "method": {
            "type": "string",
            "description": "Method: 'sideband', 'abcd', 'template', 'fit_and_extrapolate'"
        },
        "config": {
            "type": "string",
            "description": "JSON configuration for the method"
        }
    }
    output_type = "string"

    def forward(self, data: str, method: str, config: str = "{}") -> str:
        try:
            hist_data = json.loads(data)
            cfg = json.loads(config)
        except json.JSONDecodeError as e:
            return f"JSON parse error: {str(e)}"

        methods = {
            "sideband": self._sideband_method,
            "abcd": self._abcd_method,
            "template": self._template_method,
            "fit_and_extrapolate": self._fit_extrapolate
        }

        if method not in methods:
            return f"Unknown method: {method}. Available: {list(methods.keys())}"

        try:
            return methods[method](hist_data, cfg)
        except Exception as e:
            return f"Background estimation error: {str(e)}"

    def _sideband_method(self, hist_data: Dict, cfg: Dict) -> str:
        """Estimate background from sideband regions"""
        edges = np.array(hist_data.get("bin_edges", []))
        values = np.array(hist_data.get("values", []))

        if len(edges) == 0:
            # Generate sample
            data = np.concatenate([
                np.random.normal(91, 2.5, 8000),
                np.random.exponential(20, 2000) + 60
            ])
            values, edges = np.histogram(data, bins=60, range=(60, 120))

        centers = (edges[:-1] + edges[1:]) / 2

        # Define regions
        signal_low = cfg.get("signal_low", 86)
        signal_high = cfg.get("signal_high", 96)
        sideband_low = cfg.get("sideband_low", [60, 80])
        sideband_high = cfg.get("sideband_high", [100, 120])

        # Get sideband data
        mask_sb_low = (centers >= sideband_low[0]) & (centers < sideband_low[1])
        mask_sb_high = (centers >= sideband_high[0]) & (centers < sideband_high[1])
        mask_signal = (centers >= signal_low) & (centers < signal_high)

        sb_values = np.concatenate([values[mask_sb_low], values[mask_sb_high]])
        sb_centers = np.concatenate([centers[mask_sb_low], centers[mask_sb_high]])

        # Fit sideband with polynomial
        from scipy import optimize

        def poly(x, a, b, c):
            return a + b*x + c*x**2

        popt, _ = optimize.curve_fit(poly, sb_centers, sb_values, p0=[100, -1, 0.01])

        # Extrapolate to signal region
        signal_centers = centers[mask_signal]
        bkg_prediction = poly(signal_centers, *popt)
        bkg_total = np.sum(bkg_prediction)

        # Observed in signal region
        signal_observed = np.sum(values[mask_signal])

        result = {
            "method": "sideband",
            "signal_region": [signal_low, signal_high],
            "sideband_regions": [sideband_low, sideband_high],
            "background_estimate": float(bkg_total),
            "background_error": float(np.sqrt(bkg_total)),
            "observed_signal_region": float(signal_observed),
            "excess": float(signal_observed - bkg_total),
            "fit_parameters": {"a": float(popt[0]), "b": float(popt[1]), "c": float(popt[2])}
        }

        print(f"Background estimate: {bkg_total:.1f} +/- {np.sqrt(bkg_total):.1f}")
        print(f"Observed: {signal_observed:.0f}, Excess: {signal_observed - bkg_total:.1f}")

        return json.dumps(result, indent=2)

    def _abcd_method(self, hist_data: Dict, cfg: Dict) -> str:
        """ABCD method for background estimation"""
        # Regions: A (signal), B, C, D (control)
        # B/A = D/C -> A = B*C/D

        n_A = cfg.get("n_A", 100)  # Signal region (to predict)
        n_B = cfg.get("n_B", 500)  # Control region 1
        n_C = cfg.get("n_C", 200)  # Control region 2
        n_D = cfg.get("n_D", 1000)  # Control region 3

        # Background prediction
        bkg_A = n_B * n_C / n_D if n_D > 0 else 0

        # Uncertainty (assuming Poisson)
        rel_err_B = 1/np.sqrt(n_B) if n_B > 0 else 1
        rel_err_C = 1/np.sqrt(n_C) if n_C > 0 else 1
        rel_err_D = 1/np.sqrt(n_D) if n_D > 0 else 1
        rel_err_A = np.sqrt(rel_err_B**2 + rel_err_C**2 + rel_err_D**2)
        bkg_A_err = bkg_A * rel_err_A

        result = {
            "method": "ABCD",
            "regions": {
                "A_observed": n_A,
                "B": n_B,
                "C": n_C,
                "D": n_D
            },
            "background_estimate": float(bkg_A),
            "background_error": float(bkg_A_err),
            "excess": float(n_A - bkg_A),
            "transfer_factor": float(n_C / n_D) if n_D > 0 else 0
        }

        return json.dumps(result, indent=2)

    def _template_method(self, hist_data: Dict, cfg: Dict) -> str:
        """Template fitting for background estimation"""
        from scipy import optimize

        data_values = np.array(hist_data.get("data", []))
        signal_template = np.array(cfg.get("signal_template", []))
        background_template = np.array(cfg.get("background_template", []))

        if len(data_values) == 0:
            # Generate sample
            n_bins = 50
            sig = np.exp(-0.5 * ((np.arange(n_bins) - 25) / 5)**2) * 100
            bkg = np.exp(-0.02 * np.arange(n_bins)) * 500
            data_values = np.random.poisson(sig + bkg)
            signal_template = sig / np.sum(sig)
            background_template = bkg / np.sum(bkg)

        # Normalize templates
        signal_template = signal_template / np.sum(signal_template)
        background_template = background_template / np.sum(background_template)

        # Fit for signal and background yields
        def negative_log_likelihood(params):
            n_sig, n_bkg = params
            expected = n_sig * signal_template + n_bkg * background_template
            expected = np.maximum(expected, 1e-10)
            return -np.sum(data_values * np.log(expected) - expected)

        result = optimize.minimize(negative_log_likelihood, [100, 1000], method='L-BFGS-B',
                                   bounds=[(0, None), (0, None)])

        n_sig, n_bkg = result.x

        fit_result = {
            "method": "template",
            "signal_yield": float(n_sig),
            "background_yield": float(n_bkg),
            "total_observed": float(np.sum(data_values)),
            "negative_log_likelihood": float(result.fun),
            "converged": result.success
        }

        return json.dumps(fit_result, indent=2)

    def _fit_extrapolate(self, hist_data: Dict, cfg: Dict) -> str:
        """Fit in control region and extrapolate"""
        from scipy import optimize

        edges = np.array(hist_data.get("bin_edges", []))
        values = np.array(hist_data.get("values", []))

        if len(edges) == 0:
            data = np.random.exponential(30, 5000) + 60
            values, edges = np.histogram(data, bins=60, range=(60, 120))

        centers = (edges[:-1] + edges[1:]) / 2

        control_region = cfg.get("control_region", [60, 80])
        signal_region = cfg.get("signal_region", [85, 95])
        model = cfg.get("model", "exponential")

        # Fit in control region
        mask_ctrl = (centers >= control_region[0]) & (centers < control_region[1])

        def exp_model(x, a, b):
            return a * np.exp(b * x)

        popt, pcov = optimize.curve_fit(
            exp_model,
            centers[mask_ctrl],
            values[mask_ctrl],
            p0=[1000, -0.05]
        )

        # Extrapolate to signal region
        mask_sig = (centers >= signal_region[0]) & (centers < signal_region[1])
        bkg_prediction = exp_model(centers[mask_sig], *popt)
        bkg_total = np.sum(bkg_prediction)

        result = {
            "method": "fit_and_extrapolate",
            "control_region": control_region,
            "signal_region": signal_region,
            "fit_parameters": {"a": float(popt[0]), "b": float(popt[1])},
            "background_estimate": float(bkg_total),
            "background_error": float(np.sqrt(bkg_total)),
            "observed_signal": float(np.sum(values[mask_sig]))
        }

        return json.dumps(result, indent=2)


class SignalModelTool(Tool):
    """
    Create and configure signal models.
    """
    name = "signal_model"
    description = """
    Creates signal models for various physics processes:
    - Resonance shapes (Z, W, Higgs)
    - New particle hypotheses
    - Signal shape systematics
    - Resolution effects
    """
    inputs = {
        "particle": {
            "type": "string",
            "description": "Particle: 'Z', 'W', 'H', 'custom'"
        },
        "decay_channel": {
            "type": "string",
            "description": "Decay channel: 'ee', 'mumu', 'tautau', 'bb', 'gammagamma'"
        },
        "parameters": {
            "type": "string",
            "description": "JSON with model parameters"
        }
    }
    output_type = "string"

    def forward(self, particle: str, decay_channel: str, parameters: str = "{}") -> str:
        try:
            params = json.loads(parameters)
        except json.JSONDecodeError:
            return f"Invalid JSON: {parameters}"

        # Standard masses and widths
        particles = {
            "Z": {"mass": 91.1876, "width": 2.4952},
            "W": {"mass": 80.379, "width": 2.085},
            "H": {"mass": 125.25, "width": 0.00407},  # SM Higgs
            "custom": {"mass": params.get("mass", 100), "width": params.get("width", 1)}
        }

        if particle not in particles:
            return f"Unknown particle: {particle}"

        p_info = particles[particle]

        # Detector resolution by channel
        resolutions = {
            "ee": 1.5,    # GeV
            "mumu": 2.0,
            "tautau": 10.0,
            "bb": 15.0,
            "gammagamma": 1.7
        }

        resolution = params.get("resolution", resolutions.get(decay_channel, 2.0))

        # Generate signal shape
        x = np.linspace(p_info["mass"] - 30, p_info["mass"] + 30, 200)

        # Breit-Wigner convolved with Gaussian (Voigtian)
        from scipy.special import voigt_profile

        # Convert to Voigt parameters
        sigma = resolution / np.sqrt(2 * np.log(2)) / 2.355  # FWHM to sigma
        gamma = p_info["width"] / 2

        # Voigt profile
        signal_shape = voigt_profile(x - p_info["mass"], sigma, gamma)
        signal_shape = signal_shape / np.max(signal_shape)  # Normalize

        result = {
            "particle": particle,
            "decay_channel": decay_channel,
            "mass": p_info["mass"],
            "natural_width": p_info["width"],
            "detector_resolution": resolution,
            "effective_width": float(np.sqrt(p_info["width"]**2 + (2.355*resolution)**2)),
            "signal_shape": {
                "x": x.tolist(),
                "y": signal_shape.tolist()
            },
            "fit_parameters": {
                "mass": {"initial": p_info["mass"], "limits": [p_info["mass"]-5, p_info["mass"]+5]},
                "width": {"initial": p_info["width"], "limits": [0.1, 10]},
                "resolution": {"initial": resolution, "limits": [0.5, 5]}
            }
        }

        print(f"Signal model for {particle} -> {decay_channel}")
        print(f"  Mass: {p_info['mass']:.3f} GeV, Width: {p_info['width']:.4f} GeV")
        print(f"  Resolution: {resolution:.2f} GeV")

        return json.dumps(result, indent=2)


class StatisticalFitTool(Tool):
    """
    Perform statistical fits and hypothesis tests.
    """
    name = "statistical_fit"
    description = """
    Performs statistical analysis:
    - Profile likelihood fits
    - Hypothesis testing (p-values, significance)
    - Confidence intervals
    - Upper limits (CLs method)
    - Goodness of fit tests
    """
    inputs = {
        "analysis_type": {
            "type": "string",
            "description": "Type: 'fit', 'significance', 'upper_limit', 'goodness_of_fit'"
        },
        "data": {
            "type": "string",
            "description": "JSON with observed data and model"
        },
        "options": {
            "type": "string",
            "description": "JSON with analysis options"
        }
    }
    output_type = "string"

    def forward(self, analysis_type: str, data: str, options: str = "{}") -> str:
        try:
            data_dict = json.loads(data)
            opts = json.loads(options)
        except json.JSONDecodeError as e:
            return f"JSON parse error: {str(e)}"

        methods = {
            "fit": self._profile_fit,
            "significance": self._calculate_significance,
            "upper_limit": self._calculate_upper_limit,
            "goodness_of_fit": self._goodness_of_fit
        }

        if analysis_type not in methods:
            return f"Unknown analysis type: {analysis_type}"

        try:
            return methods[analysis_type](data_dict, opts)
        except Exception as e:
            return f"Statistical analysis error: {str(e)}"

    def _profile_fit(self, data: Dict, opts: Dict) -> str:
        """Profile likelihood fit"""
        from scipy import optimize, stats

        observed = np.array(data.get("observed", [100]))
        signal = np.array(data.get("signal", [20]))
        background = np.array(data.get("background", [80]))

        # Ensure arrays
        if not isinstance(observed, np.ndarray):
            observed = np.array([observed])
        if not isinstance(signal, np.ndarray):
            signal = np.array([signal])
        if not isinstance(background, np.ndarray):
            background = np.array([background])

        # Simple signal strength fit
        def neg_log_likelihood(mu):
            expected = mu * signal + background
            expected = np.maximum(expected, 1e-10)
            return np.sum(expected - observed * np.log(expected))

        # Fit
        result = optimize.minimize_scalar(neg_log_likelihood, bounds=(0, 10), method='bounded')
        mu_hat = result.x

        # Calculate uncertainty from likelihood scan
        nll_min = neg_log_likelihood(mu_hat)
        mu_range = np.linspace(max(0, mu_hat - 2), mu_hat + 2, 100)
        nll_values = [neg_log_likelihood(m) for m in mu_range]

        # Find 1-sigma interval (delta NLL = 0.5)
        mask_1sigma = np.array(nll_values) < nll_min + 0.5
        if np.any(mask_1sigma):
            mu_lo = mu_range[mask_1sigma][0]
            mu_hi = mu_range[mask_1sigma][-1]
            mu_err = (mu_hi - mu_lo) / 2
        else:
            mu_err = 0.5

        fit_result = {
            "parameter_of_interest": "mu",
            "best_fit": float(mu_hat),
            "uncertainty": float(mu_err),
            "confidence_interval_68": [float(mu_hat - mu_err), float(mu_hat + mu_err)],
            "nll_minimum": float(nll_min),
            "observed_total": float(np.sum(observed)),
            "expected_at_mu1": float(np.sum(signal + background))
        }

        print(f"Signal strength: mu = {mu_hat:.3f} +/- {mu_err:.3f}")

        return json.dumps(fit_result, indent=2)

    def _calculate_significance(self, data: Dict, opts: Dict) -> str:
        """Calculate discovery significance"""
        from scipy import stats

        observed = data.get("observed", 150)
        background = data.get("background", 100)
        background_error = data.get("background_error", np.sqrt(background))

        # Simple significance (Poisson)
        if background > 0:
            # Using approximation: Z = sqrt(2 * (n * ln(n/b) - (n-b)))
            n = observed
            b = background
            if n > b:
                z_simple = np.sqrt(2 * (n * np.log(n/b) - (n - b)))
            else:
                z_simple = 0

            # With background uncertainty
            sigma_b = background_error
            z_with_syst = z_simple * b / np.sqrt(b**2 + (n * sigma_b)**2)
        else:
            z_simple = 0
            z_with_syst = 0

        # Convert to p-value
        p_value = stats.norm.sf(z_simple)

        result = {
            "observed": observed,
            "background": background,
            "background_error": background_error,
            "excess": observed - background,
            "significance_simple": float(z_simple),
            "significance_with_syst": float(z_with_syst),
            "p_value": float(p_value),
            "discovery_threshold": 5.0,
            "is_discovery": z_simple >= 5.0
        }

        print(f"Significance: {z_simple:.2f} sigma (p-value: {p_value:.2e})")

        return json.dumps(result, indent=2)

    def _calculate_upper_limit(self, data: Dict, opts: Dict) -> str:
        """Calculate upper limit using CLs method"""
        from scipy import stats, optimize

        observed = data.get("observed", 10)
        background = data.get("background", 10)
        signal_efficiency = data.get("signal_efficiency", 1.0)
        cl = opts.get("confidence_level", 0.95)

        # Simple Feldman-Cousins / CLs-like calculation
        def cls(mu):
            """Calculate CLs for signal strength mu"""
            expected_s_b = mu * signal_efficiency + background
            expected_b = background

            # P(n >= observed | s+b)
            p_sb = 1 - stats.poisson.cdf(observed - 1, expected_s_b)
            # P(n >= observed | b)
            p_b = 1 - stats.poisson.cdf(observed - 1, expected_b)

            if p_b > 0:
                return p_sb / p_b
            return 0

        # Find mu where CLs = 1 - CL
        target = 1 - cl
        mu_values = np.linspace(0, 50, 500)
        cls_values = [cls(m) for m in mu_values]

        # Find crossing
        for i, (m, c) in enumerate(zip(mu_values, cls_values)):
            if c < target:
                mu_upper = m
                break
        else:
            mu_upper = mu_values[-1]

        # Expected limit (median under background-only)
        expected_obs = background
        def cls_expected(mu):
            expected_s_b = mu * signal_efficiency + background
            p_sb = 1 - stats.poisson.cdf(int(expected_obs) - 1, expected_s_b)
            p_b = 1 - stats.poisson.cdf(int(expected_obs) - 1, background)
            return p_sb / p_b if p_b > 0 else 0

        for m in mu_values:
            if cls_expected(m) < target:
                mu_expected = m
                break
        else:
            mu_expected = mu_values[-1]

        result = {
            "observed": observed,
            "background": background,
            "confidence_level": cl,
            "observed_limit": float(mu_upper),
            "expected_limit": float(mu_expected),
            "signal_efficiency": signal_efficiency,
            "cross_section_limit": float(mu_upper / signal_efficiency) if signal_efficiency > 0 else None,
            "method": "CLs"
        }

        print(f"Upper limit at {cl*100:.0f}% CL: {mu_upper:.2f}")
        print(f"Expected limit: {mu_expected:.2f}")

        return json.dumps(result, indent=2)

    def _goodness_of_fit(self, data: Dict, opts: Dict) -> str:
        """Perform goodness of fit test"""
        from scipy import stats

        observed = np.array(data.get("observed", []))
        expected = np.array(data.get("expected", []))

        if len(observed) == 0:
            # Generate sample
            expected = np.array([100, 150, 200, 180, 120, 80, 50, 30, 20, 10])
            observed = np.random.poisson(expected)

        # Chi-square test
        mask = expected > 0
        chi2 = np.sum((observed[mask] - expected[mask])**2 / expected[mask])
        ndf = np.sum(mask) - 1  # Assuming 1 free parameter

        p_value = 1 - stats.chi2.cdf(chi2, ndf)

        # Saturated model comparison
        chi2_saturated = np.sum((observed[mask] - expected[mask])**2 / observed[mask])

        result = {
            "test": "chi2",
            "chi2": float(chi2),
            "ndf": int(ndf),
            "chi2_ndf": float(chi2 / ndf) if ndf > 0 else 0,
            "p_value": float(p_value),
            "chi2_saturated": float(chi2_saturated),
            "good_fit": p_value > 0.05,
            "n_bins": len(observed)
        }

        print(f"Goodness of fit: chi2/ndf = {chi2/ndf:.2f}, p-value = {p_value:.3f}")

        return json.dumps(result, indent=2)
