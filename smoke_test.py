import numpy as np
from inference import competition_model, text_model, predict_competition, predict_text_only

print("Competition classes:", competition_model.classes_)
print("Text-only classes:", text_model.classes_)
print()

tests = [
    "the us is an international bully and needs its physical threat. of course, the us will not seriously consider disarming. we are more concerned in making sure that other nations do not acqiure our abilities to kill.",
    "can trupm stop putting his foot in his mouth and stay out of it?",
    "good luck with my-way-or-the-highway stevenson there's a reason she's no longer in jeffco.",
    "because he thought dusg they were attending services there. he posted threats to his mother in law that very morning. he was after them, and their friends. a nut-job that should have never been able to own a firearm. but the nra, congress, texas and the air force made sure it was an option available to him. reaping what you have sown.",
    'why is it, that the native corporation have become million dollar entities? those "jokers" must be doing something right to offer dividends year after year.. put that in your hat and eat it robert.'
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



