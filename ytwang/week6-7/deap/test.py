import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import operator
import math
import random

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

# dir = "./data/41940samples/"
# X_train = pd.read_csv(dir + "/X_train_41940samples_138features_filtered.csv", index_col=0)
# X_test = pd.read_csv(dir + "X_test_41940samples_138features_filtered.csv", index_col="seg_id")
# y_train = pd.read_csv(dir + "y_train_41940sasmples.csv",index_col=0)

dir = "../data/4194samples/"
X_train = pd.read_csv(dir + "X_train_138features_filtered.csv", index_col=0)
X_test = pd.read_csv(dir + "X_test_138features_filtered.csv", index_col="seg_id")
y_train = pd.read_csv(dir + "y_train_4194samples.csv", index_col=0)

# 正则化X
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

print(f'{X_train_scaled.shape[0]} samples in new train data and {X_train_scaled.shape[1]} columns.')
print(X_train_scaled.head())
print(f'{X_test_scaled.shape[0]} samples in new test data and {X_test_scaled.shape[1]} columns.')
print(X_test_scaled.head())
print(f'{y_train.shape[0]} samples in y_train data and {y_train.shape[1]} columns.')
print(y_train.head)


# Define new functions
def protectedDiv(left, right):
    if right == 0  or left/right > 1e13 or left/right < -1e13:
        return 1
    else:
        return left/right
    # try:
    #     return left / right
    # except ZeroDivisionError:
    #     return 1

pset = gp.PrimitiveSet("MAIN", 138)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(np.tanh, 1)
pset.addEphemeralConstant("constant73.0", lambda : 73.0)
pset.addEphemeralConstant("constant2", lambda : 2.0)
pset.addEphemeralConstant("constant5", lambda : 5.577521)
pset.addEphemeralConstant("constant0.04", lambda : 0.04)
# pset.addEphemeralConstant("r", lambda: random.randint(-1, 1))

for index, arg in enumerate(X_train_scaled.columns):
    argument_map = {}
    temp_index = "ARG%d" % index
    argument_map[temp_index] = arg
    pset.renameArguments(**argument_map)
    # print(temp_index)

# pset.renameArguments(ARG0="ave")
# pset.renameArguments(ARG1="std")
# pset.renameArguments(ARG2="max")
# pset.renameArguments(ARG3="min")
# pset.renameArguments(ARG4="sum")
# pset.renameArguments(ARG5="skew")
# pset.renameArguments(ARG6="kurt")
# pset.renameArguments(ARG7="mad")
# pset.renameArguments(ARG8="med")
# pset.renameArguments(ARG9="mean_change_abs")
# pset.renameArguments(ARG10="mean_change_rate")
# pset.renameArguments(ARG11="abs_max")
# pset.renameArguments(ARG12="abs_min")
# pset.renameArguments(ARG13="std_first_50000")
# pset.renameArguments(ARG14="std_last_50000")
# pset.renameArguments(ARG15="std_first_10000")
# pset.renameArguments(ARG16="std_last_10000")

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=10, max_=20)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def evalSymbReg(individual, x, y): # symbolic regression
    func = toolbox.compile(expr=individual)
    predicts = []
    for index, point in x.iterrows():
        try:
            pred = func(*point[:])
        except ValueError:
            pred = 0.0
        predicts.append(pred)
    predicts = np.array(predicts)
    y = np.array(y).reshape(-1)
    score = mean_absolute_error(y, predicts)
    print(score)
    return score,

toolbox.register("evaluate", evalSymbReg, x = X_train_scaled, y = y_train)
toolbox.register("select", tools.selTournament, tournsize=10)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=5, max_=10)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=25))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=25))

random.seed(318)

pop = toolbox.population(n=50)
hof = tools.HallOfFame(3)

pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 10, halloffame=hof, verbose=True)
print(hof.items[0])
print(hof.items[1])
print(hof.items[2])
# print(log)