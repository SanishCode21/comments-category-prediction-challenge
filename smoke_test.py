from inference import predict_comment

tests = [
    {
        "comment": "This is an example comment for testing.",
        "upvote": 10,
        "downvote": 2,
        "emoticon_1": 1,
    },
    {"comment": "hello"},
    {"comment": "I completely disagree with this!"},
    {"comment": "This is a much longer comment explaining my opinion..."},
    {"comment": "😂😂😂"},
]

for test in tests:
    result = predict_comment(**test)
    print(test["comment"])
    print(result)
    print("-" * 60)
