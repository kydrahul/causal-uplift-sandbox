"""
segment_analysis.py
-------------------
Cross-references Causal Uplift (CATE) against Predicted Lifetime Value.
Segments the population into a 2x2 grid to identify "Mismatch" users:
Users who have high uplift (we can save them) but low value (they aren't 
worth saving).
"""

import numpy as np
import pandas as pd

def run_segment_analysis(df: pd.DataFrame, feature_cols: list[str], cate_scores: np.ndarray, value_model) -> dict:
    """
    Given the DataFrame, CATE scores (from DoubleML), and the Value model,
    assigns users to quadrants.
    """
    print("\n[segment_analysis] Running Value-Aware Mismatch Analysis ...")
    
    X = df[feature_cols].values
    
    # 1. Predict Potential Value for EVERYONE
    predicted_value = value_model.predict(X)
    
    # 2. Define Thresholds (Medians are a robust default)
    uplift_threshold = np.median(cate_scores)
    value_threshold = np.median(predicted_value)
    
    # 3. Create Quadrant Labels
    segments = []
    for u, v in zip(cate_scores, predicted_value):
        if u >= uplift_threshold and v >= value_threshold:
            segments.append("Star Users (High Uplift, High Value)")
        elif u >= uplift_threshold and v < value_threshold:
            segments.append("Mismatch (High Uplift, Low Value)")
        elif u < uplift_threshold and v >= value_threshold:
            segments.append("Sure Things (Low Uplift, High Value)")
        else:
            segments.append("Lost Causes (Low Uplift, Low Value)")
            
    df_segments = pd.DataFrame({
        "uplift": cate_scores,
        "predicted_value": predicted_value,
        "segment": segments
    })
    
    # 4. Quantify the Mismatch
    mismatch_frac = (df_segments["segment"] == "Mismatch (High Uplift, Low Value)").mean()
    print(f"   Mismatch Fraction: {mismatch_frac*100:.1f}% of users are 'treatable' but low-value.")
    
    return {
        "uplift_threshold": float(uplift_threshold),
        "value_threshold": float(value_threshold),
        "mismatch_fraction": float(mismatch_frac),
        "predicted_value": predicted_value,
        "segments": np.array(segments)
    }
