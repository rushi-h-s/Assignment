import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Parse simulation data
data = """run_id,solver_type,mesh_count,max_stress_MPa,displacement_mm,convergence_iters,status_text,timestamp
R001,ANSYS,250000,320,1.2,18,Converged successfully,2025-01-05 10:15
R002,ANSYS,180000,890,5.6,22,Converged successfully,2025-01-05 10:32
R003,OpenFOAM,300000,100,0.9,45,Did not converge,2025-01-05 10:39
R004,ANSYS,150000,,2.1,15,Converged successfully,2025-01-05 11:18
R005,OpenFOAM,95000,310,12.5,9,Converged successfully,2025-01-05 11:42
R006,ANSYS,190000,280,1.8,9,Converged successfully,2025-01-05 12:18
R007,OpenFOAM,120000,3000,0.4,12,Converged successfully,2025-01-05 12:45
R008,ANSYS,200000,410,1.6,,Converged successfully,2025-01-05 13:05
R009,ANSYS,175000,390,1.5,16,Warning: near-yield,2025-01-05 13:25
R010,OpenFOAM,80000,95,0.2,6,Converged successfully,2025-01-05 13:50
R011,ANSYS,220000,720,4.9,19,Converged successfully,2025-01-05 14:10
R012,ANSYS,160000,590,2.0,30,Converged successfully,2025-01-05 14:28
R013,OpenFOAM,140000,480,1.9,20,Did not converge,2025-01-05 14:55
R014,ANSYS,300000,350,1.1,14,Converged successfully,2025-01-05 15:20
R015,OpenFOAM,60000,1500,8.2,9,Converged successfully,2025-01-05 15:45"""

from io import StringIO
df = pd.read_csv(StringIO(data))

# ============================================================================
# 1. CLEAN AND PREPROCESS DATA
# ============================================================================
print("="*70)
print("1. DATA CLEANING & PREPROCESSING")
print("="*70)

df['max_stress_MPa'] = pd.to_numeric(df['max_stress_MPa'], errors='coerce')
df['displacement_mm'] = pd.to_numeric(df['displacement_mm'], errors='coerce')
df['convergence_iters'] = pd.to_numeric(df['convergence_iters'], errors='coerce')
df['has_missing'] = df[['max_stress_MPa','displacement_mm','convergence_iters']].isnull().any(axis=1)
df['converged'] = df['status_text'].str.contains('Converged successfully', case=False, na=False)

print(f"Total runs: {len(df)}")
print(f"Missing data runs: {df['has_missing'].sum()}")
print("\nStatistics:\n", df[['max_stress_MPa','displacement_mm','convergence_iters']].describe())

# ============================================================================
# 2. APPLY ML MODEL (ANOMALY DETECTION)
# ============================================================================
print("\n" + "="*70)
print("2. ML-BASED ANOMALY DETECTION")
print("="*70)

ml_data = df[['max_stress_MPa','displacement_mm','convergence_iters']].dropna()
scaler = StandardScaler()
features_scaled = scaler.fit_transform(ml_data)

iso_forest = IsolationForest(contamination=0.25, random_state=42)
predictions = iso_forest.fit_predict(features_scaled)
scores = iso_forest.score_samples(features_scaled)

ml_data['ml_anomaly'] = (predictions == -1)
ml_data['ml_score'] = scores
df = df.merge(ml_data[['ml_anomaly','ml_score']], left_index=True, right_index=True, how='left')

print(f"Samples analyzed: {len(ml_data)}")
print(f"ML anomalies detected: {ml_data['ml_anomaly'].sum()}")

# ============================================================================
# 3. EXPLAIN WHY SPECIFIC RUNS ARE FLAGGED
# ============================================================================
print("\n" + "="*70)
print("3. FLAGGED RUNS WITH EXPLANATIONS")
print("="*70)

YIELD = 450
MAX_DISP = 2.5
MAX_ITER = 40

def explain_run(row):
    reasons = []
    if pd.notna(row['max_stress_MPa']) and row['max_stress_MPa'] > YIELD:
        reasons.append(f"Stress {row['max_stress_MPa']:.0f} > {YIELD} MPa (exceeds yield)")
    if pd.notna(row['displacement_mm']) and row['displacement_mm'] > MAX_DISP:
        reasons.append(f"Displacement {row['displacement_mm']:.1f} > {MAX_DISP} mm (exceeds limit)")
    if pd.notna(row['convergence_iters']) and row['convergence_iters'] > MAX_ITER:
        reasons.append(f"Iterations {row['convergence_iters']} > {MAX_ITER} (poor convergence)")
    if not row['converged']:
        reasons.append(f"Non-convergence: '{row['status_text']}'")
    if row['has_missing']:
        reasons.append("Missing critical data")
    if row.get('ml_anomaly', False):
        reasons.append(f"ML flagged (score={row['ml_score']:.2f})")
    return reasons

df['reasons'] = df.apply(explain_run, axis=1)
df['flagged'] = df['reasons'].apply(len) > 0

flagged = df[df['flagged']].sort_values('run_id')
for _, row in flagged.iterrows():
    print(f"\n{row['run_id']} ({row['solver_type']}):")
    print(f"  Values: Stress={row['max_stress_MPa']}, Disp={row['displacement_mm']}, Iter={row['convergence_iters']}")
    for reason in row['reasons']:
        print(f"  ❌ {reason}")

# ============================================================================
# 4. DESIGN RULE ENGINE BASED ON VALIDATION RULES
# ============================================================================
print("\n" + "="*70)
print("4. RULE-BASED VALIDATION ENGINE")
print("="*70)

def validate_run(row):
    severity = 'PASS'
    # Hard fails
    if (pd.notna(row['max_stress_MPa']) and row['max_stress_MPa'] > YIELD) or \
       (pd.notna(row['displacement_mm']) and row['displacement_mm'] > MAX_DISP) or \
       (pd.notna(row['convergence_iters']) and row['convergence_iters'] > MAX_ITER) or \
       not row['converged'] or row['has_missing']:
        severity = 'FAIL'
    # Soft warnings
    elif (pd.notna(row['convergence_iters']) and 20 <= row['convergence_iters'] <= MAX_ITER) or \
         (row['converged'] and len(row['reasons']) > 0):
        severity = 'WARNING'
    return severity

df['severity'] = df.apply(validate_run, axis=1)

print("\nValidation Rules Applied:")
print(f"  - Max Stress: {YIELD} MPa")
print(f"  - Max Displacement: {MAX_DISP} mm")
print(f"  - Max Iterations: {MAX_ITER}")
print(f"  - Warning Range: 20-{MAX_ITER} iterations")

print("\nValidation Results:")
print(df['severity'].value_counts().to_dict())

# ============================================================================
# 5. DISCUSS EDGE CASES AND LIMITATIONS
# ============================================================================
print("\n" + "="*70)
print("5. EDGE CASES & LIMITATIONS")
print("="*70)

print("\nEDGE CASES:")
print("  • R002,R011,R015: 'Converged' but violate physical limits")
print("    → Convergence ≠ Physical validity")
print("  • R004,R008: Missing data → Cannot validate → Treated as FAIL")
print("  • R007: Extreme stress (3000 MPa) → Likely simulation error")
print("  • R012: Borderline (30 iter, 590 MPa) → Needs manual review")

print("\nLIMITATIONS:")
print("  1. Small dataset (15 runs) → Limited ML reliability")
print("  2. Binary thresholds → No gradual degradation modeling")
print("  3. Isolation Forest contamination=0.25 is arbitrary")
print("  4. No physics-based relationships (stress-strain curves)")
print("  5. Missing context: geometry, load cases, safety factors")
print("  6. No time-series analysis for drift detection")
print("  7. ML cannot explain WHY run is anomalous")
print("  8. No solver-specific failure pattern analysis")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Total: {len(df)} | Pass: {(df['severity']=='PASS').sum()} | " + 
      f"Warning: {(df['severity']=='WARNING').sum()} | Fail: {(df['severity']=='FAIL').sum()}")
print("="*70)