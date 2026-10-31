from pathlib import Path; ROOT = Path(__file__).resolve().parents[2].as_posix()  # pipeline_run root
"""
Added in this project (no equivalent in the original package): merges the two partial SMOTE
result files into smote_all_results.json, the file name used by the shipped results.
"""
import json
W = ROOT + "/work/07_imbalance_smote/"
merged = json.load(open(W + "smote_results_partial1.json"))
merged.update(json.load(open(W + "smote_results_partial2.json")))
json.dump(merged, open(W + "smote_all_results.json", "w"), indent=2)
print({k: v["test_metrics"]["f1"] for k, v in merged.items()})
