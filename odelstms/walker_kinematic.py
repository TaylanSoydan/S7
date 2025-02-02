# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.

import os
import tensorflow as tf
from node_cell import (
    LSTMCell,
    CTRNNCell,
    ODELSTM,
    VanillaRNN,
    CTGRU,
    BidirectionalRNN,
    GRUD,
    PhasedLSTM,
    GRUODE,
    HawkLSTMCell,
)
import argparse
from irregular_sampled_datasets import Walker2dImitationData


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="lstm")
parser.add_argument("--size", default=64, type=int)
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--lr", default=0.005, type=float)
args = parser.parse_args()

data = Walker2dImitationData(seq_len=64)

# Assuming 'data' is your object of the class 'PersonData'
for attribute in dir(data):
    # Filter out private attributes and methods
    if not attribute.startswith('_'):
        try:
            value = getattr(data, attribute)
            
            # Try to get the shape if it's a numpy array or similar object
            if hasattr(value, 'shape'):
                print(f'{attribute}.shape: {value.shape}')
            else:
                print(f'{attribute}: {value}')
                
        except Exception as e:
            print(f'Could not access attribute {attribute} due to: {e}')

i = 0
print(data.train_x[i,:,:])
print(data.train_times[i,:,:])
print(data.train_y[i,:])


# if args.model == "lstm":
#     cell = LSTMCell(units=args.size)
# elif args.model == "ctrnn":
#     cell = CTRNNCell(units=args.size, num_unfolds=3, method="rk4")
# elif args.model == "node":
#     cell = CTRNNCell(units=args.size, num_unfolds=3, method="rk4", tau=0)
# elif args.model == "odelstm":
#     cell = ODELSTM(units=args.size)
# elif args.model == "ctgru":
#     cell = CTGRU(units=args.size)
# elif args.model == "vanilla":
#     cell = VanillaRNN(units=args.size)
# elif args.model == "bidirect":
#     cell = BidirectionalRNN(units=args.size)
# elif args.model == "grud":
#     cell = GRUD(units=args.size)
# elif args.model == "phased":
#     cell = PhasedLSTM(units=args.size)
# elif args.model == "gruode":
#     cell = GRUODE(units=args.size)
# elif args.model == "hawk":
#     cell = HawkLSTMCell(units=args.size)
# else:
#     raise ValueError("Unknown model type '{}'".format(args.model))

# signal_input = tf.keras.Input(shape=(data.seq_len, data.input_size), name="robot")
# time_input = tf.keras.Input(shape=(data.seq_len, 1), name="time")

# rnn = tf.keras.layers.RNN(cell, time_major=False, return_sequences=True)

# output_states = rnn((signal_input, time_input))
# y = tf.keras.layers.Dense(data.input_size)(output_states)

# model = tf.keras.Model(inputs=[signal_input, time_input], outputs=[y])

# model.compile(
#     optimizer=tf.keras.optimizers.RMSprop(args.lr),
#     loss=tf.keras.losses.MeanSquaredError(),
# )
# model.summary()

# hist = model.fit(
#     x=(data.train_x, data.train_times),
#     y=data.train_y,
#     batch_size=128,
#     epochs=args.epochs,
#     validation_data=((data.valid_x, data.valid_times), data.valid_y),
#     callbacks=[
#         tf.keras.callbacks.ModelCheckpoint(
#             "/tmp/checkpoint", save_best_only=True, save_weights_only=True, mode="min"
#         )
#     ],
# )

# # Restore checkpoint with lowest validation MSE
# model.load_weights("/tmp/checkpoint")
# best_test_loss = model.evaluate(
#     x=(data.test_x, data.test_times), y=data.test_y, verbose=2
# )
# print("Best test loss: {:0.3f}".format(best_test_loss))

# # Log result in file
# base_path = "results/walker"
# os.makedirs(base_path, exist_ok=True)
# with open("{}/{}_{}.csv".format(base_path, args.model, args.size), "a") as f:
#     f.write("{:06f}\n".format(best_test_loss))