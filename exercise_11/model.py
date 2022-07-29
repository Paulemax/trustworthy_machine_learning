from torch import nn

model = nn.Sequential(                            
    nn.Conv2d(1,64,3,padding=1,stride=2),
    nn.ReLU(),
    nn.Conv2d(64,128,3,padding=1,stride=2),
    nn.ReLU(),
    nn.Flatten(),
    nn.Dropout(0.5),
    nn.Linear(7*7*128,10)
)
