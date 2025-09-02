# algorithms/base_trainer.py

class BaseTrainer:
    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = output_dir

    def train(self):
        raise NotImplementedError("train() must be implemented in subclass")

    def evaluate(self, episodes=10):
        raise NotImplementedError("evaluate() must be implemented in subclass")

    def save(self, path):
        raise NotImplementedError("save() must be implemented in subclass")

    def load(self, path):
        raise NotImplementedError("load() must be implemented in subclass")
