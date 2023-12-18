import itertools

lista = [113, 75, 282, 348, 315, 315, 282, 101, 305, 305]
listb = [265, 22, 265, 72, 22, 1, 302, 265, 22, 265, 72, 22, 1, 302]

rolls = {
    0: [101, 133, 305, 34, 59, 58, 305],
    1: [265, 22, 265, 72, 22, 1, 302],
    2: [261, 343, 262],
    4: [83, 74],
    5: [178, 22, 203],
    6: [325, 128, 326, 206, 71],
    7: [156, 313, 313, 105, 83, 105, 82, 349, 349, 272, 25, 199, 143],
    8: [309, 42, 39, 250, 101, 346, 250, 136]
}

rollsi = {
    0: [113, 75, 282, 348, 315, 315, 282],
    1: [241, 275, 241, 65, 275, 307, 199],
    2: [158, 77, 160],
    4: [333, 335],
    5: [101, 96, 84],
    6: [66, 256, 67, 180, 132],
    7: [148, 0, 0, 0, 32, 0, 32, 325, 325, 224, 283, 107, 155],
    8: [285, 348, 350, 318, 116, 241, 318, 78]
}

# comine the rollsi and rolls dictionary
rolls = {key: rolls.get(key, []) + rollsi.get(key, []) for key in set(list(rolls.keys()) + list(rollsi.keys()))}


def find_best(lista, listb):
    combinations = list(itertools.product(lista, listb))
    # minus the first element with the last element and find absolute value
    abs_combinations = [(x[1] - x[0]) for x in combinations]
    # find the index of the value that is closest to 45
    for i, x in enumerate(abs_combinations):
        if 40 < x < 50:
            print(f"index: {i}, value: {x}")
            print(f"combination: {combinations[i]}")


# run find best for each roll and the next roll
for i in range(len(rolls)):
    if i == 3 or i == 2:
        continue
    print(f"roll: {i} vs roll: {i + 1}")
    find_best(rolls[i], rolls[i + 1])
    print()
