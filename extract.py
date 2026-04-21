import json

with open("evaluate.py", "r") as f:
    code = f.read()

# Replace the plotting part with JSON dump
code = code.replace(
    "fig, ax1 = plt.subplots(figsize=(10, 6))",
    """import json
    with open('results.json', 'w') as f:
        json.dump({
            'prune_ratios': prune_ratios,
            'actual_sparsities': actual_sparsities,
            'baseline_accs': baseline_accs,
            'hybrid_accs': hybrid_accs,
            'theoretical_speedups': theoretical_speedups
        }, f, indent=4)
        
    print('Results saved to results.json')
    import sys
    sys.exit(0)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))"""
)

with open("evaluate_dump.py", "w") as f:
    f.write(code)
