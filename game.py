from collections import namedtuple
from random import randint
from typing import List

Hints = namedtuple('Hints', ['reds', 'yellows'])
GameTurn = namedtuple('GameTurn', ['guess', 'hints'])


class Player:
    def guess(self, game_state: List[GameTurn]) -> List[int]:
        return NotImplemented


class HumanPlayer(Player):
    def guess(self, game_state: List[GameTurn]) -> List[int]:
        for i, turn in enumerate(game_state):
            print(f"Turn #{i}: code={turn.guess}, reds={turn.hints.reds}, yellows={turn.hints.yellows}")

        guess = input("Enter a list of codes seperated by space:\n").strip('\n').split(" ")
        return [int(g) for g in guess]


class CrackCodeGame:
    def __init__(self, player: Player, code_length=4, colors_amount=8):
        self.turns = 10
        self.current_turn = 0
        self.code_length = code_length
        self.color_amount = colors_amount

        self.max_hints = self.code_length * self.turns

        self.player = player

        self.secret_code: List = None
        self.game_output: List[GameTurn] = []

    @property
    def won(self):
        if len(self.game_output) == 0:
            return False

        last_hints: Hints = self.game_output[-1].hints
        return last_hints.reds == self.code_length

    @property
    def game_over(self):
        return self.won or self.current_turn == self.turns

    def _generate_secret_code(self):
        return [randint(1, self.color_amount) for _ in range(self.code_length)]

    def _calculate_hints(self, players_guess: List[int]):
        # Calculate reds
        reds = 0
        secret_code_left = self.secret_code.copy()
        players_guess_left = players_guess.copy()
        for secret, guess in zip(self.secret_code, players_guess):
            if secret == guess:
                reds += 1
                secret_code_left.remove(secret)
                players_guess_left.remove(guess)

        # Calculate yellows
        yellows = 0
        while len(players_guess_left) > 0:
            guess = players_guess_left[0]
            if guess in secret_code_left:
                yellows += 1
                secret_code_left.remove(guess)

            players_guess_left.remove(guess)

        assert yellows + reds <= self.code_length
        return Hints(reds, yellows)

    def run(self):
        self.secret_code = self._generate_secret_code()

        while not self.game_over:
            players_guess = self.player.guess(self.game_output)
            reds, yellows = self._calculate_hints(players_guess)
            self.game_output.append(GameTurn(guess=players_guess, hints=Hints(reds, yellows)))
            self.current_turn += 1

        return {
            'turns': self.current_turn,
            'won': self.won,
            'total_hints': sum(turn.hints.reds + turn.hints.yellows for turn in self.game_output)
        }


if __name__ == '__main__':
    print(CrackCodeGame(HumanPlayer()).run())