from skopt import dump, load

with open('data\\analysis_results\\ops\\bayes_opt_results.pkl', 'rb') as f:
    data = load(f)


print(data)