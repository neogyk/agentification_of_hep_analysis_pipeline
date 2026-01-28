"""
Uncertainty Quantification Tools for HEP Analysis

Tools for calculating, combining, and propagating uncertainties.
"""

import json
import numpy as np
from typing import Optional, List, Dict, Any
from smolagents import Tool


class SystematicUncertaintyTool(Tool):
    """
    Calculate systematic uncertainties.
    """
    name = "systematic_uncertainty"
    description = """
    Calculates systematic uncertainties from various sources:
    - Experimental: JES, JER, b-tagging, lepton ID/isolation
    - Theoretical: PDF, scale variations, modeling
    - Luminosity and normalization uncertainties
    """
    inputs = {
        "source": {
            "type": "string",
            "description": "Systematic source: 'jes', 'jer', 'btag', 'lepton', 'pdf', 'scale', 'lumi', 'custom'"
        },
        "nominal": {
            "type": "string",
            "description": "JSON with nominal values"
        },
        "variations": {
            "type": "string",
            "description": "JSON with up/down variations or variation set"
        }
    }
    output_type = "string"

    def forward(self, source: str, nominal: str, variations: str) -> str:
        try:
            nom_data = json.loads(nominal)
            var_data = json.loads(variations)
        except json.JSONDecodeError as e:
            return f"JSON parse error: {str(e)}"

        try:
            if source == "pdf":
                return self._calculate_pdf_uncertainty(nom_data, var_data)
            elif source == "scale":
                return self._calculate_scale_uncertainty(nom_data, var_data)
            else:
                return self._calculate_standard_uncertainty(source, nom_data, var_data)

        except Exception as e:
            return f"Systematic calculation error: {str(e)}"

    def _calculate_standard_uncertainty(self, source: str, nominal: Dict, variations: Dict) -> str:
        """Calculate standard up/down systematic"""
        nom = np.array(nominal.get("values", [nominal.get("value", 100)]))
        up = np.array(variations.get("up", nom * 1.1))
        down = np.array(variations.get("down", nom * 0.9))

        # Calculate relative uncertainties
        rel_up = (up - nom) / nom
        rel_down = (nom - down) / nom

        # Symmetrize if requested
        rel_sym = (np.abs(rel_up) + np.abs(rel_down)) / 2

        result = {
            "source": source,
            "type": "shape" if len(nom) > 1 else "normalization",
            "nominal": nom.tolist() if len(nom) > 1 else float(nom[0]),
            "relative_up": rel_up.tolist() if len(rel_up) > 1 else float(rel_up[0]),
            "relative_down": rel_down.tolist() if len(rel_down) > 1 else float(rel_down[0]),
            "symmetrized": rel_sym.tolist() if len(rel_sym) > 1 else float(rel_sym[0]),
            "is_symmetric": bool(np.allclose(np.abs(rel_up), np.abs(rel_down), rtol=0.1)),
            "max_variation": float(np.max([np.max(np.abs(rel_up)), np.max(np.abs(rel_down))]))
        }

        print(f"Systematic {source}: max variation = {result['max_variation']*100:.1f}%")

        return json.dumps(result, indent=2)

    def _calculate_pdf_uncertainty(self, nominal: Dict, variations: Dict) -> str:
        """Calculate PDF uncertainty from replica set"""
        nom = np.array(nominal.get("values", [100]))
        replicas = np.array(variations.get("replicas", [nom * (1 + 0.01*np.random.randn()) for _ in range(100)]))

        if replicas.ndim == 1:
            replicas = replicas.reshape(-1, 1)

        # Standard deviation method (for MC replicas)
        std = np.std(replicas, axis=0)
        rel_uncertainty = std / nom

        # Hessian method (if eigenvector variations provided)
        if "eigenvectors" in variations:
            eigenvectors = np.array(variations["eigenvectors"])
            # Sum in quadrature of (up_i - down_i)/2
            hessian_unc = np.sqrt(np.sum(((eigenvectors[::2] - eigenvectors[1::2])/2)**2, axis=0)) / nom

        result = {
            "source": "pdf",
            "method": "replicas" if "replicas" in variations else "hessian",
            "n_variations": len(replicas),
            "nominal": nom.tolist() if len(nom) > 1 else float(nom[0]),
            "relative_uncertainty": rel_uncertainty.tolist() if len(rel_uncertainty) > 1 else float(rel_uncertainty[0]),
            "mean_uncertainty": float(np.mean(rel_uncertainty))
        }

        print(f"PDF uncertainty: {np.mean(rel_uncertainty)*100:.1f}%")

        return json.dumps(result, indent=2)

    def _calculate_scale_uncertainty(self, nominal: Dict, variations: Dict) -> str:
        """Calculate QCD scale uncertainty from envelope"""
        nom = np.array(nominal.get("values", [100]))

        # Standard 7-point scale variation
        scale_vars = variations.get("variations", {})
        var_names = ["muR_up", "muR_down", "muF_up", "muF_down", "muR_up_muF_up", "muR_down_muF_down"]

        all_variations = [nom]
        for name in var_names:
            if name in scale_vars:
                all_variations.append(np.array(scale_vars[name]))

        if len(all_variations) == 1:
            # Generate sample variations
            for _ in range(6):
                all_variations.append(nom * (1 + 0.1 * np.random.randn()))

        all_variations = np.array(all_variations)

        # Envelope method
        up_env = np.max(all_variations, axis=0)
        down_env = np.min(all_variations, axis=0)

        rel_up = (up_env - nom) / nom
        rel_down = (nom - down_env) / nom

        result = {
            "source": "scale",
            "method": "envelope",
            "n_variations": len(all_variations) - 1,
            "nominal": nom.tolist() if len(nom) > 1 else float(nom[0]),
            "relative_up": rel_up.tolist() if len(rel_up) > 1 else float(rel_up[0]),
            "relative_down": rel_down.tolist() if len(rel_down) > 1 else float(rel_down[0]),
            "mean_uncertainty": float((np.mean(np.abs(rel_up)) + np.mean(np.abs(rel_down))) / 2)
        }

        print(f"Scale uncertainty: +{np.mean(rel_up)*100:.1f}% / -{np.mean(rel_down)*100:.1f}%")

        return json.dumps(result, indent=2)


class StatisticalUncertaintyTool(Tool):
    """
    Calculate statistical uncertainties.
    """
    name = "statistical_uncertainty"
    description = """
    Calculates statistical uncertainties:
    - Poisson uncertainties for counting experiments
    - Binomial uncertainties for efficiencies
    - Bootstrap uncertainties
    - MC statistical uncertainties
    """
    inputs = {
        "data_type": {
            "type": "string",
            "description": "Type: 'poisson', 'binomial', 'bootstrap', 'mc_stat'"
        },
        "data": {
            "type": "string",
            "description": "JSON with input data"
        },
        "options": {
            "type": "string",
            "description": "JSON with options"
        }
    }
    output_type = "string"

    def forward(self, data_type: str, data: str, options: str = "{}") -> str:
        try:
            data_dict = json.loads(data)
            opts = json.loads(options)
        except json.JSONDecodeError as e:
            return f"JSON parse error: {str(e)}"

        calculators = {
            "poisson": self._poisson_uncertainty,
            "binomial": self._binomial_uncertainty,
            "bootstrap": self._bootstrap_uncertainty,
            "mc_stat": self._mc_stat_uncertainty
        }

        if data_type not in calculators:
            return f"Unknown data type: {data_type}"

        try:
            return calculators[data_type](data_dict, opts)
        except Exception as e:
            return f"Statistical uncertainty error: {str(e)}"

    def _poisson_uncertainty(self, data: Dict, opts: Dict) -> str:
        """Poisson uncertainty for counting experiments"""
        counts = np.array(data.get("counts", [10, 20, 30]))

        # Standard sqrt(N) uncertainty
        uncertainty = np.sqrt(counts)

        # Feldman-Cousins intervals for low statistics
        fc_intervals = []
        for n in counts:
            if n < 10:
                # Approximate FC intervals
                from scipy import stats
                # Use exact Poisson intervals
                low = stats.chi2.ppf(0.16, 2*n) / 2 if n > 0 else 0
                high = stats.chi2.ppf(0.84, 2*(n+1)) / 2
                fc_intervals.append([float(low), float(high)])
            else:
                fc_intervals.append([float(n - np.sqrt(n)), float(n + np.sqrt(n))])

        result = {
            "type": "poisson",
            "counts": counts.tolist(),
            "uncertainty": uncertainty.tolist(),
            "relative_uncertainty": (uncertainty / np.maximum(counts, 1)).tolist(),
            "feldman_cousins_68": fc_intervals,
            "total_counts": float(np.sum(counts)),
            "total_uncertainty": float(np.sqrt(np.sum(counts)))
        }

        return json.dumps(result, indent=2)

    def _binomial_uncertainty(self, data: Dict, opts: Dict) -> str:
        """Binomial uncertainty for efficiencies"""
        passed = np.array(data.get("passed", [80]))
        total = np.array(data.get("total", [100]))

        efficiency = passed / total

        # Standard binomial uncertainty
        uncertainty = np.sqrt(efficiency * (1 - efficiency) / total)

        # Clopper-Pearson exact intervals
        from scipy import stats
        cp_low = []
        cp_high = []
        for k, n in zip(passed, total):
            if k == 0:
                low = 0
            else:
                low = stats.beta.ppf(0.16, k, n - k + 1)
            if k == n:
                high = 1
            else:
                high = stats.beta.ppf(0.84, k + 1, n - k)
            cp_low.append(float(low))
            cp_high.append(float(high))

        result = {
            "type": "binomial",
            "passed": passed.tolist(),
            "total": total.tolist(),
            "efficiency": efficiency.tolist(),
            "uncertainty": uncertainty.tolist(),
            "clopper_pearson_68": {
                "low": cp_low,
                "high": cp_high
            }
        }

        return json.dumps(result, indent=2)

    def _bootstrap_uncertainty(self, data: Dict, opts: Dict) -> str:
        """Bootstrap uncertainty estimation"""
        values = np.array(data.get("values", np.random.normal(100, 10, 1000)))
        n_bootstrap = opts.get("n_bootstrap", 1000)
        statistic = opts.get("statistic", "mean")

        # Bootstrap resampling
        n = len(values)
        bootstrap_stats = []

        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=n, replace=True)
            if statistic == "mean":
                bootstrap_stats.append(np.mean(sample))
            elif statistic == "median":
                bootstrap_stats.append(np.median(sample))
            elif statistic == "std":
                bootstrap_stats.append(np.std(sample))

        bootstrap_stats = np.array(bootstrap_stats)

        result = {
            "type": "bootstrap",
            "statistic": statistic,
            "n_bootstrap": n_bootstrap,
            "original_value": float(np.mean(values) if statistic == "mean" else np.median(values)),
            "bootstrap_mean": float(np.mean(bootstrap_stats)),
            "bootstrap_std": float(np.std(bootstrap_stats)),
            "confidence_interval_68": [float(np.percentile(bootstrap_stats, 16)),
                                       float(np.percentile(bootstrap_stats, 84))],
            "confidence_interval_95": [float(np.percentile(bootstrap_stats, 2.5)),
                                       float(np.percentile(bootstrap_stats, 97.5))]
        }

        return json.dumps(result, indent=2)

    def _mc_stat_uncertainty(self, data: Dict, opts: Dict) -> str:
        """MC statistical uncertainty from weighted events"""
        weights = np.array(data.get("weights", np.ones(1000)))
        values = np.array(data.get("values", np.random.normal(91, 3, 1000)))

        # Sum of weights
        sum_w = np.sum(weights)
        # Sum of weights squared (for variance)
        sum_w2 = np.sum(weights**2)

        # Effective number of events
        n_eff = sum_w**2 / sum_w2 if sum_w2 > 0 else 0

        # Weighted mean and its uncertainty
        weighted_mean = np.average(values, weights=weights)
        weighted_var = np.average((values - weighted_mean)**2, weights=weights)
        mean_uncertainty = np.sqrt(weighted_var / n_eff) if n_eff > 0 else 0

        result = {
            "type": "mc_stat",
            "n_events": len(weights),
            "n_effective": float(n_eff),
            "sum_weights": float(sum_w),
            "weighted_mean": float(weighted_mean),
            "weighted_std": float(np.sqrt(weighted_var)),
            "mean_uncertainty": float(mean_uncertainty),
            "relative_stat_uncertainty": float(1 / np.sqrt(n_eff)) if n_eff > 0 else 1.0
        }

        return json.dumps(result, indent=2)


class UncertaintyCombinerTool(Tool):
    """
    Combine multiple uncertainty sources.
    """
    name = "uncertainty_combiner"
    description = """
    Combines multiple uncertainty sources:
    - Quadrature sum (uncorrelated)
    - Linear sum (fully correlated)
    - Custom correlation handling
    - Blue (Best Linear Unbiased Estimate) combination
    """
    inputs = {
        "uncertainties": {
            "type": "string",
            "description": "JSON array of uncertainty values/components"
        },
        "method": {
            "type": "string",
            "description": "Method: 'quadrature', 'linear', 'correlation', 'blue'"
        },
        "correlation": {
            "type": "string",
            "description": "JSON correlation matrix (for 'correlation' and 'blue' methods)"
        }
    }
    output_type = "string"

    def forward(self, uncertainties: str, method: str = "quadrature",
                correlation: str = "null") -> str:
        try:
            unc_data = json.loads(uncertainties)
            corr_matrix = json.loads(correlation) if correlation != "null" else None
        except json.JSONDecodeError as e:
            return f"JSON parse error: {str(e)}"

        try:
            # Extract uncertainty values
            if isinstance(unc_data, list):
                if isinstance(unc_data[0], dict):
                    unc_values = [u.get("value", u.get("uncertainty", 0)) for u in unc_data]
                    unc_names = [u.get("name", u.get("source", f"unc_{i}")) for i, u in enumerate(unc_data)]
                else:
                    unc_values = unc_data
                    unc_names = [f"unc_{i}" for i in range(len(unc_data))]
            else:
                unc_values = list(unc_data.values())
                unc_names = list(unc_data.keys())

            unc_values = np.array(unc_values)

            if method == "quadrature":
                total = np.sqrt(np.sum(unc_values**2))
                result = {
                    "method": "quadrature",
                    "assumption": "uncorrelated",
                    "components": dict(zip(unc_names, unc_values.tolist())),
                    "total_uncertainty": float(total),
                    "dominant_source": unc_names[np.argmax(np.abs(unc_values))]
                }

            elif method == "linear":
                total = np.sum(np.abs(unc_values))
                result = {
                    "method": "linear",
                    "assumption": "fully_correlated",
                    "components": dict(zip(unc_names, unc_values.tolist())),
                    "total_uncertainty": float(total)
                }

            elif method == "correlation":
                if corr_matrix is None:
                    corr_matrix = np.eye(len(unc_values))
                else:
                    corr_matrix = np.array(corr_matrix)

                cov_matrix = np.outer(unc_values, unc_values) * corr_matrix
                total = np.sqrt(np.sum(cov_matrix))

                result = {
                    "method": "correlation",
                    "components": dict(zip(unc_names, unc_values.tolist())),
                    "correlation_matrix": corr_matrix.tolist(),
                    "covariance_matrix": cov_matrix.tolist(),
                    "total_uncertainty": float(total)
                }

            elif method == "blue":
                # BLUE combination of measurements
                values = np.array([u.get("value", 100) for u in unc_data]) if isinstance(unc_data[0], dict) else np.ones(len(unc_values)) * 100

                if corr_matrix is None:
                    corr_matrix = np.eye(len(unc_values))
                else:
                    corr_matrix = np.array(corr_matrix)

                cov_matrix = np.outer(unc_values, unc_values) * corr_matrix

                # BLUE weights
                inv_cov = np.linalg.inv(cov_matrix)
                ones = np.ones(len(unc_values))
                weights = inv_cov @ ones / (ones @ inv_cov @ ones)

                # Combined value and uncertainty
                combined_value = np.sum(weights * values)
                combined_unc = np.sqrt(1 / (ones @ inv_cov @ ones))

                result = {
                    "method": "BLUE",
                    "input_values": values.tolist(),
                    "input_uncertainties": unc_values.tolist(),
                    "weights": weights.tolist(),
                    "combined_value": float(combined_value),
                    "combined_uncertainty": float(combined_unc),
                    "chi2": float((values - combined_value) @ inv_cov @ (values - combined_value)),
                    "ndf": len(values) - 1
                }

            else:
                return f"Unknown method: {method}"

            # Add breakdown
            if method in ["quadrature", "linear", "correlation"]:
                breakdown = {}
                for name, val in zip(unc_names, unc_values):
                    contribution = val**2 / total**2 if method == "quadrature" else np.abs(val) / total
                    breakdown[name] = {
                        "value": float(val),
                        "contribution": float(contribution * 100)  # percentage
                    }
                result["breakdown"] = breakdown

            print(f"Combined uncertainty ({method}): {result.get('total_uncertainty', result.get('combined_uncertainty', 0)):.4f}")

            return json.dumps(result, indent=2)

        except Exception as e:
            return f"Uncertainty combination error: {str(e)}"


class CovarianceMatrixTool(Tool):
    """
    Build and analyze covariance matrices.
    """
    name = "covariance_matrix"
    description = """
    Builds and analyzes covariance matrices:
    - From systematic variations
    - From toy experiments
    - Decomposition and eigenanalysis
    - Correlation matrix extraction
    """
    inputs = {
        "action": {
            "type": "string",
            "description": "Action: 'build', 'decompose', 'correlate', 'regularize'"
        },
        "data": {
            "type": "string",
            "description": "JSON with input data (variations, samples, or matrix)"
        },
        "options": {
            "type": "string",
            "description": "JSON with options"
        }
    }
    output_type = "string"

    def forward(self, action: str, data: str, options: str = "{}") -> str:
        try:
            data_dict = json.loads(data)
            opts = json.loads(options)
        except json.JSONDecodeError as e:
            return f"JSON parse error: {str(e)}"

        actions = {
            "build": self._build_covariance,
            "decompose": self._decompose_covariance,
            "correlate": self._extract_correlation,
            "regularize": self._regularize_matrix
        }

        if action not in actions:
            return f"Unknown action: {action}"

        try:
            return actions[action](data_dict, opts)
        except Exception as e:
            return f"Covariance matrix error: {str(e)}"

    def _build_covariance(self, data: Dict, opts: Dict) -> str:
        """Build covariance matrix from variations"""
        method = opts.get("method", "variations")

        if method == "variations":
            # Build from systematic variations
            nominal = np.array(data.get("nominal", [100, 90, 80]))
            variations = data.get("variations", {})

            n_bins = len(nominal)
            cov = np.zeros((n_bins, n_bins))

            for name, var in variations.items():
                up = np.array(var.get("up", nominal * 1.1))
                down = np.array(var.get("down", nominal * 0.9))
                delta = (up - down) / 2
                cov += np.outer(delta, delta)

        elif method == "toys":
            # Build from toy experiments
            toys = np.array(data.get("toys", np.random.multivariate_normal([100, 90, 80], np.eye(3)*10, 1000)))
            cov = np.cov(toys.T)

        else:
            # Direct matrix input
            cov = np.array(data.get("matrix", np.eye(3)))

        result = {
            "covariance_matrix": cov.tolist(),
            "diagonal": np.diag(cov).tolist(),
            "uncertainties": np.sqrt(np.diag(cov)).tolist(),
            "is_positive_definite": bool(np.all(np.linalg.eigvalsh(cov) > 0)),
            "condition_number": float(np.linalg.cond(cov))
        }

        return json.dumps(result, indent=2)

    def _decompose_covariance(self, data: Dict, opts: Dict) -> str:
        """Eigenvalue decomposition of covariance matrix"""
        cov = np.array(data.get("matrix", data.get("covariance_matrix", np.eye(3))))

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Cumulative variance explained
        total_variance = np.sum(eigenvalues)
        variance_explained = np.cumsum(eigenvalues) / total_variance

        result = {
            "eigenvalues": eigenvalues.tolist(),
            "eigenvectors": eigenvectors.tolist(),
            "variance_explained": variance_explained.tolist(),
            "n_components_90pct": int(np.argmax(variance_explained >= 0.9) + 1),
            "total_variance": float(total_variance)
        }

        return json.dumps(result, indent=2)

    def _extract_correlation(self, data: Dict, opts: Dict) -> str:
        """Extract correlation matrix from covariance"""
        cov = np.array(data.get("matrix", data.get("covariance_matrix", np.eye(3))))

        # Correlation = Cov / (sigma_i * sigma_j)
        sigma = np.sqrt(np.diag(cov))
        corr = cov / np.outer(sigma, sigma)

        result = {
            "correlation_matrix": corr.tolist(),
            "max_off_diagonal": float(np.max(np.abs(corr - np.diag(np.diag(corr))))),
            "highly_correlated_pairs": []
        }

        # Find highly correlated pairs
        n = len(corr)
        for i in range(n):
            for j in range(i+1, n):
                if np.abs(corr[i, j]) > 0.5:
                    result["highly_correlated_pairs"].append({
                        "bins": [i, j],
                        "correlation": float(corr[i, j])
                    })

        return json.dumps(result, indent=2)

    def _regularize_matrix(self, data: Dict, opts: Dict) -> str:
        """Regularize covariance matrix"""
        cov = np.array(data.get("matrix", data.get("covariance_matrix", np.eye(3))))
        method = opts.get("method", "ridge")
        alpha = opts.get("alpha", 0.01)

        if method == "ridge":
            # Ridge regularization
            reg_cov = cov + alpha * np.eye(len(cov)) * np.trace(cov) / len(cov)
        elif method == "shrinkage":
            # Ledoit-Wolf shrinkage
            n = len(cov)
            mu = np.trace(cov) / n
            delta = np.sum((cov - mu * np.eye(n))**2) / n
            reg_cov = (1 - alpha) * cov + alpha * mu * np.eye(n)
        else:
            reg_cov = cov

        result = {
            "original_condition": float(np.linalg.cond(cov)),
            "regularized_condition": float(np.linalg.cond(reg_cov)),
            "regularization_method": method,
            "alpha": alpha,
            "regularized_matrix": reg_cov.tolist(),
            "is_positive_definite": bool(np.all(np.linalg.eigvalsh(reg_cov) > 0))
        }

        return json.dumps(result, indent=2)
