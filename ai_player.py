import json
import os
from typing import List

import keras
import keras.layers as layers
import numpy as np
import tensorflow as tf

from game import GameTurn
from game import Player


class AIPlayer(Player):
    """
    The AI player has an RNN which get as inputs the state of the game,
    list of 10 turns, each turn has a list of the guess for that turn and a tuple of 2 numbers which are the hints.
    so the dimension of the data is 10x(4+2) => 10x6
    """

    def __init__(self, max_turns=10, code_length=4, colors_amount=8):
        self.max_turns = max_turns
        self.code_length = code_length
        self.colors_amount = colors_amount

        self.model = self._build_model()

    def pickle(self, directory_path):
        self.model.save(directory_path)
        with open(os.path.join(directory_path, 'player.data'), 'w') as f:
            json.dump({
                'max_turns': self.max_turns,
                'code_length': self.code_length,
                'colors_amount': self.colors_amount
            }, f)

    @classmethod
    def unpickle(cls, saved_directory):
        with open(os.path.join(saved_directory, 'player.data'), 'r') as f:
            data = json.load(f)

        instance = cls(**data)
        model = keras.models.load_model(saved_directory)
        instance.model = model
        return instance

    def _build_model(self):
        return tf.keras.Sequential([
            layers.Dense(100, batch_input_shape=(None, self.max_turns, self.code_length + 2)),
            layers.Dense(80),
            layers.Flatten(),
            layers.Dense(32),
            layers.Reshape((self.code_length, self.colors_amount)),
            layers.Softmax()
        ])

    def guess(self, game_state: List[GameTurn]) -> List[int]:
        if len(game_state) < self.max_turns:
            turn_left = self.max_turns - len(game_state)
            padding = np.zeros((turn_left, self.code_length + 2))
            if turn_left == self.max_turns:
                game_state = padding

            else:
                game_state = [state.guess + [state.hints.reds, state.hints.yellows]
                              for state in game_state]
                game_state = np.vstack((game_state, padding))

        nn_input = np.array(game_state)
        nn_input = nn_input.reshape((1, self.max_turns, self.code_length + 2))

        nn_input = nn_input / nn_input.max() # Scale

        result = self.model.predict(nn_input)
        result = result.reshape((self.code_length, self.colors_amount))
        return list(result.argmax(axis=1) + 1)  # adding one since the numbers starting at 1 and not 0.
