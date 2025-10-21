# Quick fix script
import re

# Read the file
with open('phase2/training/evaluators/model_evaluator.py', 'r') as f:
    content = f.read()

# Replace all float values in to_dict method
replacements = [
    ('self.mse,', 'float(self.mse),'),
    ('self.mae,', 'float(self.mae),'),
    ('self.rmse,', 'float(self.rmse),'),
    ('self.r2_score,', 'float(self.r2_score),'),
    ('self.correlation,', 'float(self.correlation),'),
    ('self.max_error,', 'float(self.max_error),'),
    ('self.min_error,', 'float(self.min_error),'),
    ('self.std_error,', 'float(self.std_error),'),
    ('self.p25_error,', 'float(self.p25_error),'),
    ('self.p50_error,', 'float(self.p50_error),'),
    ('self.p75_error,', 'float(self.p75_error),'),
    ('self.p90_error,', 'float(self.p90_error),'),
    ('self.p95_error,', 'float(self.p95_error),'),
    ('self.inference_time_ms,', 'float(self.inference_time_ms),'),
    ('self.throughput_samples_per_sec,', 'float(self.throughput_samples_per_sec),'),
]

for old, new in replacements:
    content = content.replace(old, new)

# Write back
with open('phase2/training/evaluators/model_evaluator.py', 'w') as f:
    f.write(content)

print("✅ Fixed all float32 serialization issues!")
