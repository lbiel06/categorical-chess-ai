from create_dataset import create, generate_random_positions
from predict import predict


fens = list(generate_random_positions(10, 100))

predictions = predict(fens)

for fen, pred in zip(fens, predictions):
    print(fen, end=' ')
    if pred == 0:
        print('BLACK')
    if pred == 1:
        print('BALANCED')
    if pred == 2:
        print('WHITE')