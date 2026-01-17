Simulation Quality Analysis - Code Logic Overview This script is designed to act as a Quality Assurance Engineer for simulation data. It takes raw logs from ANSYS and OpenFOAM and filters out bad results so engineers don't waste time analyzing broken simulations.

We use a "Belt and Suspenders" approach:

Hard Rules (The Belt): Explicit engineering limits (like Yield Strength).

Machine Learning (The Suspenders): Statistical analysis to catch weird patterns that don't technically break a rule but look suspicious.

🛠️ The Logic Behind The Code

Data Cleaning & Preprocessing The Problem: Real-world data is messy. Text might say "Converged" while the numbers are missing, or a number might be stored as text. The Solution:
pd.to_numeric(..., errors='coerce'): This forces everything into numbers. If a cell contains garbage text, it turns into NaN (Not a Number) instead of crashing the script.

isnull().any(axis=1): We immediately flag any row with missing data. If we don't know the stress value, we can't trust the simulation, period.

Machine Learning (Anomaly Detection) The Tool: IsolationForest The Why:
Rules are great, but they are rigid. What if a simulation has low stress but massive displacement? A simple "Max Stress" rule might miss that.

Isolation Forest works by trying to "isolate" points. Weird data points (anomalies) are easy to isolate because they sit far away from everyone else.

contamination=0.25: We tell the model, "Assume roughly 25% of our data might be bad." This makes the model aggressive enough to catch the outliers in this small dataset.

The "Explain Run" Function The Why: A flag saying "FAIL" isn't helpful. An engineer needs to know why it failed so they can fix it. The Logic:
We check against Physical Limits:

Yield Strength (450 MPa): If stress is higher than this, the part has permanently bent or broken.

Max Displacement (2.5mm): If it moves more than this, it might be hitting something else.

We check Solver Health:

Iterations (>40): If the computer had to guess 40+ times to find an answer, the answer is likely unstable.

Status Text: We trust the solver's own error messages (e.g., "Did not converge").

The Rule Engine (Severity Levels) The Why: Not all failures are equal.
FAIL: The simulation is useless (e.g., The part broke, or the data is missing).

WARNING: The simulation technically finished, but it looks shaky (e.g., it took a long time to converge). These might still be useful but need a human double-check.

PASS: Clean, healthy data.

Edge Cases Handling The Problem: Sometimes the data lies.
The "Converged but Broken" Case: Several runs (like R002) say "Converged Successfully" but have physically impossible stress values (890 MPa).

The Fix: Our code prioritizes numbers over text. Even if the status says "Success," if the stress is too high, we mark it as a FAIL.
