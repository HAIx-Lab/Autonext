import collections
import math
import pickle
from typing import List

class PPMWithDictionary:
    def __init__(self, order: int, smoothing_factor: float = 1):
        self.order = order
        self.context_dict = collections.defaultdict(lambda: collections.defaultdict(int))  # Context dictionary
        self.smoothing_factor = smoothing_factor
        self.lambda_weights = []  # Interpolation weights

    def train(self, text: str):
        """
        Train the PPM model by populating the context frequencies using a dictionary.
        """
        for i in range(len(text)):
            for j in range(1, self.order + 1):
                if i - j < 0:
                    break
                context = text[i - j:i]
                char = text[i]
                self.context_dict[context][char] += 1

    def save(self, filename: str):
        """
        Save the trained model's context dictionary to a file.
        """
        with open(filename, 'wb') as f:
            pickle.dump({
                'order': self.order,
                'context_dict': dict(self.context_dict),  # Convert defaultdict to a regular dict
                'smoothing_factor': self.smoothing_factor,
                'lambda_weights': self.lambda_weights
            }, f)
        print(f"Model saved to {filename}")

    @classmethod
    def load(cls, filename: str):
        """
        Load a trained model from a file.
        """
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        model = cls(order=data['order'], smoothing_factor=data['smoothing_factor'])
        # Reinitialize the context_dict as defaultdict
        model.context_dict = collections.defaultdict(lambda: collections.defaultdict(int), data['context_dict'])
        model.lambda_weights = data['lambda_weights']
        print(f"Model loaded from {filename}")
        return model

    def entropy(self, context: str) -> float:
        """
        Calculate the entropy of a given context.
        """
        total = sum(self.context_dict[context].values())
        if total == 0:
            return float('inf')  # Unseen context

        entropy = 0.0
        for count in self.context_dict[context].values():
            probability = count / total
            entropy -= probability * math.log2(probability)

        return entropy

    def predict(self, context: str) -> dict:
        """
        Predict the probability distribution for the next character with optimized interpolation.
        """
        probabilities = collections.defaultdict(float)
        unique_chars = set(char for sub_context in self.context_dict for char in self.context_dict[sub_context])
        total_unique = len(unique_chars)

        # Dynamically compute weights based on entropy
        entropies = [self.entropy(context[-i:]) for i in range(self.order + 1)]
        max_entropy = max(entropies)

        valid_entropies = [max_entropy - e for e in entropies if e != float('inf')]
        if sum(valid_entropies) == 0:
            self.lambda_weights = [1 / (self.order + 1)] * (self.order + 1)
        else:
            self.lambda_weights = [
                (max_entropy - entropy) / sum(valid_entropies)
                if entropy != float('inf') else 0
                for entropy in entropies
            ]

        for i in range(self.order + 1):  # Interpolating over all orders
            sub_context = context[-i:]
            total = sum(self.context_dict[sub_context].values()) + self.smoothing_factor * total_unique
            if total > 0:
                for char in unique_chars:
                    count = self.context_dict[sub_context].get(char, 0)
                    probabilities[char] += self.lambda_weights[i] * ((count + self.smoothing_factor) / total)

        return probabilities

    def next_char(self, context: str) -> str:
        """
        Predict the next character based on the current context.
        """
        probabilities = self.predict(context)
        if probabilities:
            return max(probabilities, key=probabilities.get)
        return None

    def evaluate(self, text: str) -> float:
        """
        Evaluate the model accuracy on the given text.
        """
        correct_predictions = 0
        total_predictions = 0

        for i in range(self.order, len(text)):
            context = text[i - self.order:i]
            true_char = text[i]
            predicted_char = self.next_char(context)

            if predicted_char == true_char:
                correct_predictions += 1
            total_predictions += 1

        return correct_predictions / total_predictions


# Example usage
if __name__ == "__main__":
    # Load the Hindi dataset
    with open("/home/tanmay.somkuwar/implementation/gujarati.txt", "r", encoding='utf-8') as f:
        dataset = f.read()

    # Split into training and test sets (80% training, 20% testing)
    split_index = int(0.8 * len(dataset))
    train_text = dataset[:split_index]
    test_text = dataset[split_index:]

    # Train the PPM model
    ppm_model = PPMWithDictionary(order=3) #Change the order as required
    ppm_model.train(train_text)
    ppm_model.save("ppm_model_gujarati_order3.pkl")

    # Load the trained model and evaluate
    ppm_model_loaded = PPMWithDictionary.load("ppm_model_gujarati_order3.pkl")
   
