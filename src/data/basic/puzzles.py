
import random
import string

chars = string.printable
vocab_list = [x for x in chars]

class Dataset:

    def generate(self):
        return None, None


class Move(Dataset):
    def __init__(self, vocab = {
        "left": "l",
        "right": "r",
        "up": "u",
        "down": "d",
        "newline": "\n",
        "blank": "0",
        "full": "1"
    }, dimensions={"min-width":3, "min-height":3, "max-width": 8, "max-height": 8}):
        self.vocab = vocab
        self.dimensions = dimensions

    def generate(self):
        direction = random.choice(["left", "right", "up", "down"])
        width = random.randint(self.dimensions["min-width"], self.dimensions["max-width"])
        height = random.randint(self.dimensions["min-width"], self.dimensions["max-width"])
        x = 0
        y = 0
        if direction == "left":
            x = random.randint(1, width - 1)
            y = random.randint(0, height - 1)
        elif direction == "right":
            x = random.randint(0, width - 2)
            y = random.randint(0, height - 1)
        elif direction == "up":
            x = random.randint(0, width - 1)
            y = random.randint(1, height - 1)
        elif direction == "down":
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 2)

        question_grid = [[self.vocab["blank"] for _ in range(width)] for _ in range(height)]
        question_grid[y][x] = self.vocab["full"]

        answer_grid = [[self.vocab["blank"] for _ in range(width)] for _ in range(height)]
        if direction == "left":
            answer_grid[y][x - 1] = self.vocab["left"]
        elif direction == "right":
            answer_grid[y][x + 1] = self.vocab["right"]
        elif direction == "up":
            answer_grid[y - 1][x] = self.vocab["up"]
        elif direction == "down":
            answer_grid[y + 1][x] = self.vocab["down"]

        question = self.vocab[direction] + "\n" + "\n".join(["".join(row) for row in question_grid])
        answer = "\n".join(["".join(row) for row in answer_grid])

        return question, answer

