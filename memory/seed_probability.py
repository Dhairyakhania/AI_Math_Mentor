"""
Seed probability memory with common JEE problems and traps.
"""

from memory.store import MemoryStore


def seed_probability_memory(memory: MemoryStore):
    examples = [
        {
            "problem": "A coin is tossed 3 times. What is the probability of getting exactly two heads?",
            "solution": "Sample space has 8 outcomes. Favorable outcomes = {HHT, HTH, THH}. Probability = 3/8.",
            "topic": "probability",
            "metadata": {
                "core_concepts": "sample_space,combinations",
                "difficulty": "jee",
                "trap": "order_vs_combination"
            }
        },
        {
            "problem": "Two dice are thrown. Find the probability that the sum is 7.",
            "solution": "There are 36 outcomes. Favorable outcomes = 6. Probability = 6/36 = 1/6.",
            "topic": "probability",
            "metadata": {
                "core_concepts": "sample_space,sum_distribution",
                "difficulty": "jee",
                "trap": "unordered_pairs"
            }
        },
        {
            "problem": "A card is drawn from a deck. Find probability it is a red card.",
            "solution": "There are 26 red cards out of 52. Probability = 1/2.",
            "topic": "probability",
            "metadata": {
                "core_concepts": "basic_probability",
                "difficulty": "foundation"
            }
        }
    ]

    for ex in examples:
        memory.add_entry(
            problem=ex["problem"],
            solution=ex["solution"],
            topic=ex["topic"],
            metadata=ex["metadata"]
        )
