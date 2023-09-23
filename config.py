ATR2IDX = {
    'red': 0,
    'green': 1,
    'blue': 2,
    'yellow': 3,
    'black': 4,
    'purple': 5,
}

OBJ2IDX = {
    'Circle': 0,
    'Rectangle': 1,
    'Triangle': 2,
    'Pentagon': 3,
    'Oval': 4,
    'Hexagon': 5,
}

IDX2ATR = {v : k for k, v in ATR2IDX.items()}

IDX2OBJ = {v : k for k, v in OBJ2IDX.items()}

classes = []
for va in IDX2ATR.values():
    for vo in IDX2OBJ.values():
        classes.append(f"{va} {vo}")