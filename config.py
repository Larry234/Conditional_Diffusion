import torch


ATR2IDX = {
    'red': 0,
    'green': 1,
    'blue': 2,
    'yellow' : 3,
    'black': 4,
    'purple' : 5,
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
        
CLS2IDX = {classes[i] : i for i in range(len(classes))}

class Zappo50K:
    ATR2IDX = {
        'Heel': 0,
        'Flat': 1,
    }
    
    OBJ2IDX = {
        'Boot': 0,
        'Shoe': 1,
        'Slipper': 2,
        'Sandal': 3,
    }
    
    IDX2ATR = {v : k for k, v in ATR2IDX.items()}

    IDX2OBJ = {v : k for k, v in OBJ2IDX.items()}
    
    classes = []
    for va in IDX2ATR.values():
        for vo in IDX2OBJ.values():
            classes.append(f"{va} {vo}")