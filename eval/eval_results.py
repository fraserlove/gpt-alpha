import json, sys, os, glob
from tabulate import tabulate

models = ['__'.join(model.split('/')) for model in sys.argv[1:]]

tests = {
    'commonsenseqa_0shot': 'acc,none',
    'piqa_0shot': 'acc_norm,none',
    'siqa_0shot': 'acc,none',
    'openbookqa_0shot': 'acc_norm,none',
    'triviaqa_0shot': 'exact_match,remove_whitespace',
    'truthfulqa_0shot': 'acc,none',
    'mmlu_5shot': 'acc,none',
    'winogrande_5shot': 'acc,none',
    'arc_challenge_25shot': 'acc_norm,none',
    'hellaswag_10shot': 'acc_norm,none',
    'gsm8k_5shot': 'exact_match,flexible-extract',
}

# Initialize the results dictionary
results = {test: [] for test in tests.keys()}
average_scores = []

for model in models:
    total = 0
    model_averages = []
    for test in tests.keys():
        try:
            # Find any JSON file in the specified directory
            search_pattern = os.path.join(test, model, '*.json')
            result_files = glob.glob(search_pattern)
            if not result_files:
                raise FileNotFoundError(f'No results files found for pattern: {search_pattern}')
            
            with open(result_files[0]) as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f'File not found: {test}/{model}')
            results[test].append('N/A')
            continue
        except json.JSONDecodeError:
            print(f'Error decoding JSON in file: {test}/{model}')
            results[test].append('N/A')
            continue

        r_count = 0
        r_total = 0
        for test_name in data['results']:
            r_count += 1
            r_total += data['results'][test_name][tests[test]]
        score = (r_total * 100) / r_count
        results[test].append(f'{score:.2f}')
        total += score

    average = total / len(tests)
    average_scores.append(f'{average:.2f}')

# Add the average scores as a row in the results dictionary
results['Average Score'] = average_scores

# Prepare headers
headers = ['Benchmark'] + [model.split('__')[1] if '__' in model else model for model in models]

# Prepare data for tabulation
table_data = [[test] + scores for test, scores in results.items()]

# Display the results table
print(tabulate(table_data, headers=headers, tablefmt='github', floatfmt='.2f'))
