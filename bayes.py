import pandas as pddef update(table):    table['unnorm'] = table['prior'] * table['likelihood']    prob_data = table['unnorm'].sum()    table['posterior'] = table['unnorm'] / prob_data    return prob_datadef prob(A):    return A.mean()def conditional(proposition, given):    return prob(proposition[given])def odd(p):    return p / (1-p)def prob_odd(o):    return o / (o+1)