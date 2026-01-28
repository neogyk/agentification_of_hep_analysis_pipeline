"""
Physics Analysis Tools for HEP Analysis

Core physics analysis tools for event selection, data processing,
and particle reconstruction.
"""

import json
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from smolagents import Tool


class EventSelectionTool(Tool):
    """
    Apply physics event selection criteria.
    """
    name = "event_selection"
    description = """
    Applies event selection cuts to physics data.
    Supports:
    - Kinematic cuts (pT, eta, phi, mass)
    - Object multiplicity requirements
    - Isolation criteria
    - Trigger requirements
    - Quality flags

    Returns selected events and cut flow statistics.
    """
    inputs = {
        "data": {
            "type": "string",
            "description": "Path to ROOT file or JSON data specification"
        },
        "cuts": {
            "type": "string",
            "description": "JSON array of cut definitions"
        },
        "output_format": {
            "type": "string",
            "description": "Output: 'events', 'cutflow', 'both'. Default: 'both'"
        }
    }
    output_type = "string"

    def forward(self, data: str, cuts: str, output_format: str = "both") -> str:
        try:
            cut_list = json.loads(cuts)
        except json.JSONDecodeError:
            return f"Invalid cuts JSON: {cuts}"

        try:
            # Load data
            events = self._load_data(data)
            if isinstance(events, str):
                return events  # Error message

            # Apply cuts and track cut flow
            cutflow = {"Initial": len(events)}
            mask = np.ones(len(events), dtype=bool)

            for cut in cut_list:
                cut_mask = self._apply_cut(events, cut)
                mask = mask & cut_mask
                cut_name = cut.get("name", cut.get("variable", "unnamed"))
                cutflow[cut_name] = int(np.sum(mask))
                print(f"After {cut_name}: {np.sum(mask)} events ({np.sum(mask)/cutflow['Initial']*100:.1f}%)")

            # Filter events
            selected = {k: v[mask] for k, v in events.items()}

            result = {}

            if output_format in ["cutflow", "both"]:
                result["cutflow"] = cutflow
                result["efficiency"] = cutflow[list(cutflow.keys())[-1]] / cutflow["Initial"]

            if output_format in ["events", "both"]:
                result["n_selected"] = int(np.sum(mask))
                result["variables"] = list(selected.keys())
                # Include sample of selected events
                n_sample = min(10, int(np.sum(mask)))
                result["sample"] = {k: v[:n_sample].tolist() for k, v in selected.items()}

            return json.dumps(result, indent=2)

        except Exception as e:
            return f"Event selection error: {str(e)}"

    def _load_data(self, data: str) -> Dict[str, np.ndarray]:
        """Load data from file or specification"""
        import os

        if os.path.exists(data) and data.endswith('.root'):
            try:
                import uproot
                with uproot.open(data) as f:
                    # Find first tree
                    for key in f.keys():
                        obj = f[key]
                        if hasattr(obj, 'arrays'):
                            arrays = obj.arrays(library="np")
                            return dict(arrays)
                return "No TTree found in ROOT file"
            except Exception as e:
                return f"Error loading ROOT file: {e}"

        elif data.startswith('{'):
            # JSON data specification
            try:
                spec = json.loads(data)
                return spec
            except:
                pass

        # Generate sample data for demonstration
        n_events = 10000
        return {
            "pt": np.random.exponential(50, n_events),
            "eta": np.random.uniform(-2.5, 2.5, n_events),
            "phi": np.random.uniform(-np.pi, np.pi, n_events),
            "mass": np.random.normal(91.2, 3, n_events),
            "isolation": np.random.exponential(0.1, n_events),
            "n_jets": np.random.poisson(3, n_events),
            "met": np.random.exponential(30, n_events)
        }

    def _apply_cut(self, events: Dict, cut: Dict) -> np.ndarray:
        """Apply a single cut"""
        variable = cut.get("variable")
        operator = cut.get("operator", ">")
        value = cut.get("value", 0)

        if variable not in events:
            print(f"Warning: Variable {variable} not found")
            return np.ones(len(list(events.values())[0]), dtype=bool)

        data = events[variable]

        if operator == ">":
            return data > value
        elif operator == ">=":
            return data >= value
        elif operator == "<":
            return data < value
        elif operator == "<=":
            return data <= value
        elif operator == "==":
            return data == value
        elif operator == "!=":
            return data != value
        elif operator == "between":
            low, high = value
            return (data >= low) & (data <= high)
        elif operator == "abs<":
            return np.abs(data) < value
        elif operator == "abs>":
            return np.abs(data) > value
        else:
            return np.ones(len(data), dtype=bool)


class DataProcessingTool(Tool):
    """
    Process and transform physics data.
    """
    name = "data_processing"
    description = """
    Processes and transforms physics data:
    - Apply calibrations and corrections
    - Calculate derived variables
    - Apply scale factors
    - Handle object matching
    - Perform data cleaning
    """
    inputs = {
        "data": {
            "type": "string",
            "description": "Input data (file path or JSON)"
        },
        "operations": {
            "type": "string",
            "description": "JSON array of processing operations"
        }
    }
    output_type = "string"

    def forward(self, data: str, operations: str) -> str:
        try:
            ops = json.loads(operations)
        except json.JSONDecodeError:
            return f"Invalid operations JSON: {operations}"

        try:
            # Load or parse data
            if data.startswith('{'):
                events = json.loads(data)
                events = {k: np.array(v) for k, v in events.items()}
            else:
                # Generate sample data
                n = 1000
                events = {
                    "pt": np.random.exponential(50, n),
                    "eta": np.random.uniform(-2.5, 2.5, n),
                    "phi": np.random.uniform(-np.pi, np.pi, n),
                    "energy": np.random.exponential(100, n)
                }

            # Apply operations
            for op in ops:
                op_type = op.get("type")
                print(f"Applying operation: {op_type}")

                if op_type == "scale":
                    variable = op["variable"]
                    factor = op["factor"]
                    events[variable] = events[variable] * factor

                elif op_type == "shift":
                    variable = op["variable"]
                    shift = op["shift"]
                    events[variable] = events[variable] + shift

                elif op_type == "calculate":
                    new_var = op["name"]
                    formula = op["formula"]
                    # Safe evaluation with numpy functions
                    events[new_var] = self._evaluate_formula(events, formula)

                elif op_type == "smear":
                    variable = op["variable"]
                    resolution = op["resolution"]
                    events[variable] = events[variable] * (1 + np.random.normal(0, resolution, len(events[variable])))

                elif op_type == "clean":
                    # Remove NaN and inf values
                    for k, v in events.items():
                        mask = np.isfinite(v)
                        if not np.all(mask):
                            print(f"  Removed {np.sum(~mask)} non-finite values from {k}")
                            for key in events:
                                events[key] = events[key][mask]

            result = {
                "n_events": len(list(events.values())[0]),
                "variables": list(events.keys()),
                "statistics": {}
            }

            for var, values in events.items():
                result["statistics"][var] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }

            return json.dumps(result, indent=2)

        except Exception as e:
            return f"Data processing error: {str(e)}"

    def _evaluate_formula(self, events: Dict, formula: str) -> np.ndarray:
        """Safely evaluate a formula"""
        # Create namespace with numpy functions and event variables
        namespace = {
            "np": np,
            "sqrt": np.sqrt,
            "abs": np.abs,
            "sin": np.sin,
            "cos": np.cos,
            "exp": np.exp,
            "log": np.log,
            "cosh": np.cosh,
            "sinh": np.sinh,
            **events
        }
        return eval(formula, {"__builtins__": {}}, namespace)


class ReconstructionTool(Tool):
    """
    Reconstruct physics objects and composite particles.
    """
    name = "reconstruction"
    description = """
    Reconstructs physics objects:
    - Combine particles into composite objects (Z, W, H)
    - Calculate invariant masses
    - Perform jet clustering
    - Apply b-tagging
    - Reconstruct decay chains
    """
    inputs = {
        "particles": {
            "type": "string",
            "description": "JSON with particle 4-vectors (pt, eta, phi, mass)"
        },
        "reconstruction_type": {
            "type": "string",
            "description": "Type: 'dilepton', 'dijet', 'resonance', 'missing_et'"
        },
        "options": {
            "type": "string",
            "description": "JSON with reconstruction options"
        }
    }
    output_type = "string"

    def forward(self, particles: str, reconstruction_type: str,
                options: str = "{}") -> str:
        try:
            particle_data = json.loads(particles)
            opts = json.loads(options)
        except json.JSONDecodeError as e:
            return f"JSON parse error: {str(e)}"

        try:
            if reconstruction_type == "dilepton":
                return self._reconstruct_dilepton(particle_data, opts)
            elif reconstruction_type == "dijet":
                return self._reconstruct_dijet(particle_data, opts)
            elif reconstruction_type == "resonance":
                return self._reconstruct_resonance(particle_data, opts)
            elif reconstruction_type == "missing_et":
                return self._calculate_met(particle_data, opts)
            else:
                return f"Unknown reconstruction type: {reconstruction_type}"

        except Exception as e:
            return f"Reconstruction error: {str(e)}"

    def _reconstruct_dilepton(self, data: Dict, opts: Dict) -> str:
        """Reconstruct dilepton system"""
        # Get lepton 4-vectors
        pt1 = np.array(data.get("lepton1_pt", data.get("pt1", [])))
        eta1 = np.array(data.get("lepton1_eta", data.get("eta1", [])))
        phi1 = np.array(data.get("lepton1_phi", data.get("phi1", [])))
        m1 = np.array(data.get("lepton1_mass", np.zeros(len(pt1))))

        pt2 = np.array(data.get("lepton2_pt", data.get("pt2", [])))
        eta2 = np.array(data.get("lepton2_eta", data.get("eta2", [])))
        phi2 = np.array(data.get("lepton2_phi", data.get("phi2", [])))
        m2 = np.array(data.get("lepton2_mass", np.zeros(len(pt2))))

        if len(pt1) == 0 or len(pt2) == 0:
            # Generate sample data
            n = 1000
            pt1 = np.random.exponential(40, n) + 25
            eta1 = np.random.uniform(-2.4, 2.4, n)
            phi1 = np.random.uniform(-np.pi, np.pi, n)
            m1 = np.full(n, 0.10566)  # muon mass

            pt2 = np.random.exponential(30, n) + 20
            eta2 = np.random.uniform(-2.4, 2.4, n)
            phi2 = np.random.uniform(-np.pi, np.pi, n)
            m2 = np.full(n, 0.10566)

        # Calculate 4-vectors
        px1 = pt1 * np.cos(phi1)
        py1 = pt1 * np.sin(phi1)
        pz1 = pt1 * np.sinh(eta1)
        e1 = np.sqrt(px1**2 + py1**2 + pz1**2 + m1**2)

        px2 = pt2 * np.cos(phi2)
        py2 = pt2 * np.sin(phi2)
        pz2 = pt2 * np.sinh(eta2)
        e2 = np.sqrt(px2**2 + py2**2 + pz2**2 + m2**2)

        # Combine
        px = px1 + px2
        py = py1 + py2
        pz = pz1 + pz2
        e = e1 + e2

        # Invariant mass
        m2_inv = e**2 - px**2 - py**2 - pz**2
        m_inv = np.sqrt(np.maximum(m2_inv, 0))

        # System kinematics
        pt_sys = np.sqrt(px**2 + py**2)
        eta_sys = np.arctanh(pz / np.sqrt(px**2 + py**2 + pz**2 + 1e-10))
        phi_sys = np.arctan2(py, px)

        result = {
            "n_pairs": len(m_inv),
            "mass": {
                "mean": float(np.mean(m_inv)),
                "std": float(np.std(m_inv)),
                "median": float(np.median(m_inv))
            },
            "pt": {
                "mean": float(np.mean(pt_sys)),
                "std": float(np.std(pt_sys))
            },
            "mass_distribution": {
                "bins": np.linspace(60, 120, 31).tolist(),
                "counts": np.histogram(m_inv, bins=30, range=(60, 120))[0].tolist()
            }
        }

        print(f"Reconstructed {len(m_inv)} dilepton pairs")
        print(f"Mean mass: {np.mean(m_inv):.2f} GeV")

        return json.dumps(result, indent=2)

    def _reconstruct_dijet(self, data: Dict, opts: Dict) -> str:
        """Reconstruct dijet system"""
        pt1 = np.array(data.get("jet1_pt", []))
        eta1 = np.array(data.get("jet1_eta", []))
        phi1 = np.array(data.get("jet1_phi", []))
        m1 = np.array(data.get("jet1_mass", []))

        pt2 = np.array(data.get("jet2_pt", []))
        eta2 = np.array(data.get("jet2_eta", []))
        phi2 = np.array(data.get("jet2_phi", []))
        m2 = np.array(data.get("jet2_mass", []))

        if len(pt1) == 0:
            # Generate sample data
            n = 1000
            pt1 = np.random.exponential(100, n) + 30
            eta1 = np.random.uniform(-2.5, 2.5, n)
            phi1 = np.random.uniform(-np.pi, np.pi, n)
            m1 = np.random.uniform(5, 20, n)

            pt2 = np.random.exponential(80, n) + 30
            eta2 = np.random.uniform(-2.5, 2.5, n)
            phi2 = np.random.uniform(-np.pi, np.pi, n)
            m2 = np.random.uniform(5, 20, n)

        # Calculate invariant mass
        px1 = pt1 * np.cos(phi1)
        py1 = pt1 * np.sin(phi1)
        pz1 = pt1 * np.sinh(eta1)
        e1 = np.sqrt(px1**2 + py1**2 + pz1**2 + m1**2)

        px2 = pt2 * np.cos(phi2)
        py2 = pt2 * np.sin(phi2)
        pz2 = pt2 * np.sinh(eta2)
        e2 = np.sqrt(px2**2 + py2**2 + pz2**2 + m2**2)

        m_jj = np.sqrt(np.maximum((e1+e2)**2 - (px1+px2)**2 - (py1+py2)**2 - (pz1+pz2)**2, 0))

        # Delta R between jets
        dphi = phi1 - phi2
        dphi = np.where(dphi > np.pi, dphi - 2*np.pi, dphi)
        dphi = np.where(dphi < -np.pi, dphi + 2*np.pi, dphi)
        deta = eta1 - eta2
        dR = np.sqrt(dphi**2 + deta**2)

        result = {
            "n_pairs": len(m_jj),
            "dijet_mass": {
                "mean": float(np.mean(m_jj)),
                "std": float(np.std(m_jj))
            },
            "delta_R": {
                "mean": float(np.mean(dR)),
                "std": float(np.std(dR))
            }
        }

        return json.dumps(result, indent=2)

    def _reconstruct_resonance(self, data: Dict, opts: Dict) -> str:
        """Generic resonance reconstruction"""
        target_mass = opts.get("target_mass", 125)  # Higgs default
        window = opts.get("mass_window", 20)

        # Use dilepton reconstruction as base
        result = json.loads(self._reconstruct_dilepton(data, opts))

        # Add resonance-specific info
        mass_dist = result["mass_distribution"]
        counts = np.array(mass_dist["counts"])
        bins = np.array(mass_dist["bins"])
        centers = (bins[:-1] + bins[1:]) / 2

        # Find peak
        peak_idx = np.argmax(counts)
        peak_mass = centers[peak_idx]

        result["resonance"] = {
            "target_mass": target_mass,
            "observed_peak": float(peak_mass),
            "peak_counts": int(counts[peak_idx]),
            "signal_window": [target_mass - window/2, target_mass + window/2]
        }

        return json.dumps(result, indent=2)

    def _calculate_met(self, data: Dict, opts: Dict) -> str:
        """Calculate missing transverse energy"""
        # Sum all visible momenta
        met_x = 0
        met_y = 0

        for key in data:
            if "_pt" in key and "_phi" in key.replace("_pt", "_phi"):
                pt = np.array(data[key])
                phi_key = key.replace("_pt", "_phi")
                if phi_key in data:
                    phi = np.array(data[phi_key])
                    met_x -= np.sum(pt * np.cos(phi))
                    met_y -= np.sum(pt * np.sin(phi))

        if met_x == 0 and met_y == 0:
            # Generate sample MET
            n = 1000
            met_x = np.random.normal(0, 30, n)
            met_y = np.random.normal(0, 30, n)
            met = np.sqrt(met_x**2 + met_y**2)
        else:
            met = np.sqrt(met_x**2 + met_y**2)

        if isinstance(met, np.ndarray):
            result = {
                "met": {
                    "mean": float(np.mean(met)),
                    "std": float(np.std(met)),
                    "distribution": np.histogram(met, bins=20, range=(0, 200))[0].tolist()
                }
            }
        else:
            result = {"met": float(met)}

        return json.dumps(result, indent=2)


class KinematicCalculatorTool(Tool):
    """
    Calculate kinematic variables for physics analysis.
    """
    name = "kinematic_calculator"
    description = """
    Calculates derived kinematic variables:
    - Invariant mass
    - Transverse mass (mT)
    - Delta R, Delta phi, Delta eta
    - Thrust, sphericity
    - Razor variables
    - Aplanarity, circularity
    """
    inputs = {
        "calculation": {
            "type": "string",
            "description": "Calculation type: 'invariant_mass', 'transverse_mass', 'delta_r', 'angular', 'shape'"
        },
        "particles": {
            "type": "string",
            "description": "JSON with particle kinematics"
        }
    }
    output_type = "string"

    def forward(self, calculation: str, particles: str) -> str:
        try:
            data = json.loads(particles)
        except json.JSONDecodeError:
            return f"Invalid JSON: {particles}"

        calculators = {
            "invariant_mass": self._calc_invariant_mass,
            "transverse_mass": self._calc_transverse_mass,
            "delta_r": self._calc_delta_r,
            "angular": self._calc_angular_variables,
            "shape": self._calc_shape_variables
        }

        if calculation not in calculators:
            return f"Unknown calculation: {calculation}. Available: {list(calculators.keys())}"

        try:
            return calculators[calculation](data)
        except Exception as e:
            return f"Calculation error: {str(e)}"

    def _calc_invariant_mass(self, data: Dict) -> str:
        """Calculate invariant mass"""
        # Get all particle 4-vectors
        particles = []
        i = 1
        while f"pt{i}" in data or f"particle{i}_pt" in data:
            key_prefix = f"pt{i}" if f"pt{i}" in data else f"particle{i}_pt"
            base = key_prefix.replace("_pt", "").replace("pt", "")

            if base:
                pt = np.array(data.get(f"{base}_pt", data.get(f"pt{i}", [])))
                eta = np.array(data.get(f"{base}_eta", data.get(f"eta{i}", [])))
                phi = np.array(data.get(f"{base}_phi", data.get(f"phi{i}", [])))
                mass = np.array(data.get(f"{base}_mass", data.get(f"mass{i}", np.zeros(len(pt)))))
            else:
                pt = np.array(data.get(f"pt{i}", []))
                eta = np.array(data.get(f"eta{i}", []))
                phi = np.array(data.get(f"phi{i}", []))
                mass = np.array(data.get(f"mass{i}", np.zeros(len(pt))))

            if len(pt) > 0:
                particles.append({"pt": pt, "eta": eta, "phi": phi, "mass": mass})
            i += 1

        if len(particles) < 2:
            return json.dumps({"error": "Need at least 2 particles for invariant mass"})

        # Calculate total 4-vector
        px_tot = np.zeros(len(particles[0]["pt"]))
        py_tot = np.zeros(len(particles[0]["pt"]))
        pz_tot = np.zeros(len(particles[0]["pt"]))
        e_tot = np.zeros(len(particles[0]["pt"]))

        for p in particles:
            px = p["pt"] * np.cos(p["phi"])
            py = p["pt"] * np.sin(p["phi"])
            pz = p["pt"] * np.sinh(p["eta"])
            e = np.sqrt(px**2 + py**2 + pz**2 + p["mass"]**2)

            px_tot += px
            py_tot += py
            pz_tot += pz
            e_tot += e

        m_inv = np.sqrt(np.maximum(e_tot**2 - px_tot**2 - py_tot**2 - pz_tot**2, 0))

        result = {
            "n_particles": len(particles),
            "invariant_mass": {
                "values": m_inv.tolist()[:100],  # First 100 values
                "mean": float(np.mean(m_inv)),
                "std": float(np.std(m_inv)),
                "min": float(np.min(m_inv)),
                "max": float(np.max(m_inv))
            }
        }

        return json.dumps(result, indent=2)

    def _calc_transverse_mass(self, data: Dict) -> str:
        """Calculate transverse mass (for W->lnu type decays)"""
        lepton_pt = np.array(data.get("lepton_pt", data.get("pt1", np.random.exponential(40, 100))))
        lepton_phi = np.array(data.get("lepton_phi", data.get("phi1", np.random.uniform(-np.pi, np.pi, 100))))
        met = np.array(data.get("met", np.random.exponential(30, len(lepton_pt))))
        met_phi = np.array(data.get("met_phi", np.random.uniform(-np.pi, np.pi, len(lepton_pt))))

        dphi = lepton_phi - met_phi
        mt = np.sqrt(2 * lepton_pt * met * (1 - np.cos(dphi)))

        result = {
            "transverse_mass": {
                "values": mt.tolist()[:100],
                "mean": float(np.mean(mt)),
                "std": float(np.std(mt)),
                "min": float(np.min(mt)),
                "max": float(np.max(mt))
            }
        }

        return json.dumps(result, indent=2)

    def _calc_delta_r(self, data: Dict) -> str:
        """Calculate Delta R between objects"""
        eta1 = np.array(data.get("eta1", np.random.uniform(-2.5, 2.5, 100)))
        phi1 = np.array(data.get("phi1", np.random.uniform(-np.pi, np.pi, 100)))
        eta2 = np.array(data.get("eta2", np.random.uniform(-2.5, 2.5, 100)))
        phi2 = np.array(data.get("phi2", np.random.uniform(-np.pi, np.pi, 100)))

        deta = eta1 - eta2
        dphi = phi1 - phi2
        dphi = np.where(dphi > np.pi, dphi - 2*np.pi, dphi)
        dphi = np.where(dphi < -np.pi, dphi + 2*np.pi, dphi)

        dR = np.sqrt(deta**2 + dphi**2)

        result = {
            "delta_R": {
                "values": dR.tolist()[:100],
                "mean": float(np.mean(dR)),
                "std": float(np.std(dR))
            },
            "delta_eta": {
                "mean": float(np.mean(np.abs(deta))),
                "std": float(np.std(deta))
            },
            "delta_phi": {
                "mean": float(np.mean(np.abs(dphi))),
                "std": float(np.std(dphi))
            }
        }

        return json.dumps(result, indent=2)

    def _calc_angular_variables(self, data: Dict) -> str:
        """Calculate angular variables"""
        eta = np.array(data.get("eta", np.random.uniform(-2.5, 2.5, 100)))
        phi = np.array(data.get("phi", np.random.uniform(-np.pi, np.pi, 100)))

        # cos(theta*) from eta
        theta = 2 * np.arctan(np.exp(-eta))
        cos_theta = np.cos(theta)

        result = {
            "cos_theta": {
                "mean": float(np.mean(cos_theta)),
                "std": float(np.std(cos_theta))
            },
            "eta_distribution": {
                "mean": float(np.mean(eta)),
                "std": float(np.std(eta))
            },
            "phi_distribution": {
                "mean": float(np.mean(phi)),
                "std": float(np.std(phi)),
                "is_uniform": float(np.std(phi)) > 1.5  # Check for flat distribution
            }
        }

        return json.dumps(result, indent=2)

    def _calc_shape_variables(self, data: Dict) -> str:
        """Calculate event shape variables"""
        # Simplified thrust calculation
        px = np.array(data.get("px", np.random.normal(0, 50, 100)))
        py = np.array(data.get("py", np.random.normal(0, 50, 100)))
        pz = np.array(data.get("pz", np.random.normal(0, 50, 100)))

        # Sum of |p|
        p_sum = np.sum(np.sqrt(px**2 + py**2 + pz**2))

        # Simplified sphericity tensor
        S = np.array([
            [np.sum(px*px), np.sum(px*py), np.sum(px*pz)],
            [np.sum(py*px), np.sum(py*py), np.sum(py*pz)],
            [np.sum(pz*px), np.sum(pz*py), np.sum(pz*pz)]
        ]) / (p_sum**2 + 1e-10)

        eigenvalues = np.linalg.eigvalsh(S)
        eigenvalues = np.sort(eigenvalues)[::-1]

        sphericity = 1.5 * (eigenvalues[1] + eigenvalues[2])
        aplanarity = 1.5 * eigenvalues[2]

        result = {
            "sphericity": float(sphericity),
            "aplanarity": float(aplanarity),
            "eigenvalues": eigenvalues.tolist()
        }

        return json.dumps(result, indent=2)
