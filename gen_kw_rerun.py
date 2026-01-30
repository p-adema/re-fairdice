
import os
import math
import sys
import scipy.stats as stats

# Load scores
sys.path.append(os.getcwd())
try:
    from scores_v2 import scores
except ImportError:
    with open('scores_v2.py', 'r') as f:
        content = f.read()
    context = {'nan': float('nan')}
    exec(content, context)
    scores = context['scores']

data = {}
for key, val in scores.items():
    parts = key.split('/')
    if len(parts) != 4: continue
    env, type_, method, beta = parts
    
    if method != 'Rerun': continue
        
    if env not in data: data[env] = {}
    if type_ not in data[env]: data[env][type_] = {}
    
    data[env][type_][beta] = val

print(f"{'Group (Env/Type)':<40} | {'H-Statistic':<12} | {'P-Value':<12} | {'Significance':<15}")
print("-" * 90)

for env in sorted(data.keys()):
    for dtype in sorted(data[env].keys()):
        betas = sorted(data[env][dtype].keys())
        groups = [data[env][dtype][b] for b in betas]
        
        # Filter out NaNs and empty groups
        filter_groups = []
        for g in groups:
            params = [x for x in g if not math.isnan(x)]
            if params:
                filter_groups.append(params)
        
        if len(filter_groups) < 2:
            print(f"{f'{env}/{dtype}':<40} | {'N/A':<12} | {'N/A':<12} | {'Not Enough Data':<15}")
            continue
            
        try:
            H, p_value = stats.kruskal(*filter_groups)
            sig_str = "SIGNIFICANT" if p_value < 0.05 else "Not Significant"
            
            # Format P-value
            if p_value < 0.001:
                p_str = "< 0.001"
            else:
                p_str = f"{p_value:.4f}"
            
            print(f"{f'{env}/{dtype}':<40} | {H:<12.4f} | {p_str:<12} | {sig_str:<15}")
        except ValueError:
             print(f"{f'{env}/{dtype}':<40} | {'Error':<12} | {'Error':<12} | {'Error':<15}")
