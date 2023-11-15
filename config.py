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


class TripleCond:
    def __init__(self):
        
        self.SIZE2IDX = {
            'small': 0,
            'medium': 1,
            'big': 2,
        }
        
        self.ATR2IDX = {
            'red': 0,
            'green': 1,
            'blue': 2,
            'yellow' : 3,
            'black': 4,
            'purple' : 5,
        }

        self.OBJ2IDX = {
            'Circle': 0,
            'Rectangle': 1,
            'Triangle': 2,
            'Pentagon': 3,
            'Oval': 4,
            'Hexagon': 5,
        }
        
        self.IDX2SIZE = {v : k for k, v in self.SIZE2IDX.items()}
        self.IDX2ATR = {v : k for k, v in self.ATR2IDX.items()}
        self.IDX2OBJ = {v : k for k, v in self.OBJ2IDX.items()}
        
        self.classes = []
        for vs in self.IDX2SIZE.values():
            for va in self.IDX2ATR.values():
                for vo in self.IDX2OBJ.values():
                    self.classes.append(f"{vs} {va} {vo}")

        self.CLS2IDX = {self.classes[i] : i for i in range(len(self.classes))}

class CelebATriple():
    def __init__(self):
        
        self.SIZE2IDX = {
            'Straight_Hair': 0,
            'Wavy_Hair': 1,
        }
        
        self.ATR2IDX = {
            'Brown_Hair': 0,
            'Blond_Hair': 1,
            'Gray_Hair': 2,
            'Black_Hair': 3
        }

        self.OBJ2IDX = {
            'Male': 0,
            'Female': 1,
        }
        
        self.IDX2SIZE = {v : k for k, v in self.SIZE2IDX.items()}
        self.IDX2ATR = {v : k for k, v in self.ATR2IDX.items()}
        self.IDX2OBJ = {v : k for k, v in self.OBJ2IDX.items()}
        
        self.classes = []
        for vs in self.IDX2SIZE.values():
            for va in self.IDX2ATR.values():
                for vo in self.IDX2OBJ.values():
                    self.classes.append(f"{vs} {va} {vo}")

        self.CLS2IDX = {self.classes[i] : i for i in range(len(self.classes))}