import numpy as np
from inference import (
    competition_model,
    text_model,
    predict_competition,
    predict_text_only,
)

print("Competition classes:", competition_model.classes_)
print("Text-only classes:", text_model.classes_)
print()

tests = [
    "hello",
    "I completely disagree with this!",
    "This is a much longer comment explaining my opinion...",
    "😂😂😂",
]

print("=== TEXT-ONLY MODEL ===")
for text in tests:
    print(text)
    print(predict_text_only(text))
    print("-" * 60)

print("\n=== COMPETITION MODEL ===")
for text in tests:
    print(text)
    print(predict_competition(text))
    print("-" * 60)

assert np.array_equal(competition_model.classes_, np.array([0, 1, 2, 3]))
assert np.array_equal(text_model.classes_, np.array([0, 1, 2, 3]))

print("\nSmoke test completed successfully.")

