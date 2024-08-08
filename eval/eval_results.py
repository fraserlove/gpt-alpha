import json, sys, os, glob

model = '__'.join(sys.argv[1].split('/'))

tests = {
    'commonsenseqa_0shot': 'acc,none',
    'piqa_0shot': 'acc_norm,none',
    'openbookqa_0shot': 'acc_norm,none',
    'triviaqa_0shot': 'exact_match,remove_whitespace',
    'truthfulqa_0shot': 'acc,none',
    'mmlu_5shot': 'acc,none',
    'winogrande_5shot': 'acc,none',
    'arc_challenge_25shot': 'acc_norm,none',
    'hellaswag_10shot': 'acc_norm,none',
    'gsm8k_5shot': 'exact_match,flexible-extract',
}

total = 0
print('-' * 35)
for test in list(tests.keys()):
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
        continue
    except json.JSONDecodeError:
        print(f'Error decoding JSON in file: {test}/{model}')
        continue

    r_count = 0
    r_total = 0
    for test_name in data['results']:
        r_count += 1
        r_total += data['results'][test_name][tests[test]]
    score = (r_total * 100) / r_count
    print(f'{test:<25} : {score:.1f}')
    total += score

average = total / len(tests)
print('-' * 35)
print(f'{"Average Score":<25} : {average:.1f}')