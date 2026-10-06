import numpy as np
import pandas as pd

MODEL_COLUMNS = [
    "post_id", "emoticon_1", "emoticon_2", "emoticon_3", "upvote", "downvote",
    "if_1", "if_2", "race", "religion", "gender", "disability", "comment",
    "char_len", "word_count", "total_emoticons", "net_score", "month", "hour",
    "day", "year", "is_weekend", "avg_word_length", "long_comment",
    "emoticon_density", "total_votes", "vote_ratio", "engagement_score",
    "emoticon_vote_interaction", "post_comment_count",
]

# Stable defaults from the engineered X_train distribution.
DEFAULT_IF1 = 0.0
DEFAULT_IF2 = 1.0804178182729878
DEFAULT_MONTH = 5
DEFAULT_HOUR = 13
DEFAULT_IS_WEEKEND = 0

DEFAULT_RACE = "unknown"
DEFAULT_RELIGION = "unknown"
DEFAULT_GENDER = "unknown"
DEFAULT_DISABILITY = 0

DEFAULT_UPVOTE = 1
DEFAULT_DOWNVOTE = 0
DEFAULT_EMOTICON_1 = 0
DEFAULT_EMOTICON_2 = 0
DEFAULT_EMOTICON_3 = 0


def build_competition_features(
    comment,
    upvote=DEFAULT_UPVOTE,
    downvote=DEFAULT_DOWNVOTE,
    emoticon_1=DEFAULT_EMOTICON_1,
    emoticon_2=DEFAULT_EMOTICON_2,
    emoticon_3=DEFAULT_EMOTICON_3,
    race=DEFAULT_RACE,
    religion=DEFAULT_RELIGION,
    gender=DEFAULT_GENDER,
    disability=DEFAULT_DISABILITY,
):
    """Create one row matching the fitted competition pipeline schema."""
    comment = "" if comment is None else str(comment)

    upvote = max(0, int(upvote or 0))
    downvote = max(0, int(downvote or 0))
    emoticon_1 = max(0, int(emoticon_1 or 0))
    emoticon_2 = max(0, int(emoticon_2 or 0))
    emoticon_3 = max(0, int(emoticon_3 or 0))
    disability = int(bool(disability))

    word_count = len(comment.split())
    char_len = len(comment)
    total_emoticons = emoticon_1 + emoticon_2 + emoticon_3
    total_votes = upvote + downvote

    row = {
        "post_id": 0,
        "emoticon_1": emoticon_1,
        "emoticon_2": emoticon_2,
        "emoticon_3": emoticon_3,
        "upvote": upvote,
        "downvote": downvote,
        # Already transformed values expected by the fitted model.
        "if_1": DEFAULT_IF1,
        "if_2": DEFAULT_IF2,
        "race": race or DEFAULT_RACE,
        "religion": religion or DEFAULT_RELIGION,
        "gender": gender or DEFAULT_GENDER,
        "disability": disability,
        "comment": comment,
        "char_len": char_len,
        "word_count": word_count,
        "total_emoticons": total_emoticons,
        "net_score": upvote - downvote,
        "month": DEFAULT_MONTH,
        "hour": DEFAULT_HOUR,
        "day": 0,
        "year": 0,
        "is_weekend": DEFAULT_IS_WEEKEND,
        "avg_word_length": char_len / max(word_count, 1),
        "long_comment": int(word_count > 100),
        "emoticon_density": total_emoticons / (word_count + 1),
        "total_votes": total_votes,
        "vote_ratio": upvote / (downvote + 1),
        "engagement_score": np.log1p(total_votes) * np.log1p(word_count + 1),
        "emoticon_vote_interaction": total_emoticons * total_votes,
        "post_comment_count": 1,
    }

    return pd.DataFrame([row], columns=MODEL_COLUMNS)

