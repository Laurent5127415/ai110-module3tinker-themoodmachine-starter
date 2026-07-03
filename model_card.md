# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

You may complete this model card for whichever version you used, or compare both if you explored them.

## 1. Model Overview

**Model type:**  
I compared both models. The main implementation is a rule based sentiment scorer in `mood_analyzer.py`, and I also trained a small bag-of-words logistic regression model in `ml_experiments.py`.

**Intended purpose:**  
The system is designed to classify short text snippets into mood labels: `positive`, `negative`, `neutral`, or `mixed`.

**How it works (brief):**  
The rule based model tokenizes input, looks up positive and negative words, applies simple negation handling, and converts the resulting score into a label. The ML model uses `CountVectorizer` to convert text into word-count vectors and trains `LogisticRegression` on the same labeled examples.

## 2. Data

**Dataset description:**  
`SAMPLE_POSTS` contains 15 short text examples after I expanded it. The original starter set had 11 posts; I added 4 new examples that include slang, emojis, mixed feelings, and a sarcastic-style sentence.

**Labeling process:**  
I labeled the new posts manually based on the intended mood signal in each sentence. Examples labeled `mixed` were chosen when the sentence contained both positive and negative emotion words. One especially tricky post was `"I absolutely love getting stuck in traffic"`, which is labeled `negative` because the overall meaning is sarcastic and the mood is intended to be bad despite the word "love." Another mixed example was `"I feel okay, not great but not awful"`.

**Important characteristics of your dataset:**  
- Contains slang and informal language like `lowkey`, `no cap`, and `not bad`
- Includes emojis such as `🤑` and `☕`
- Includes mixed-feeling examples like `"Feeling tired but kind of hopeful"`
- Includes a sarcastic or ironic example: `"I absolutely love getting stuck in traffic"`
- Mostly short social-media-style posts

**Possible issues with the dataset:**  
- Very small dataset (15 examples)
- Limited diversity in wording and topics
- Label definitions are subjective for mixed and sarcastic posts
- Some examples are highly similar to each other, which can make training accuracy optimistic

## 3. How the Rule Based Model Works (if used)

**Your scoring rules:**  
- Text is lowercased, punctuation is removed, and tokens are split on whitespace.
- The model keeps a lexicon of positive and negative words from `dataset.py`.
- A score starts at 0.
  - Positive words add `+1`
  - Negative words subtract `-1`
- Negation handling flips sentiment when a negation word appears before a sentiment word, looking ahead up to three tokens. For example, `"not a bad day"` becomes positive.
- The model also treats some emojis as positive tokens by including them in `POSITIVE_WORDS`.
- The predicted label is:
  - `positive` if score > 0
  - `negative` if score < 0
  - `neutral` if score == 0
  - `mixed` if score == 0 and the text contains both positive and negative lexicon words

**Strengths of this approach:**  
- Predictable and easy to inspect
- Works well when the sentence contains clear sentiment words from the lexicon
- Handles simple negation and explicit mixed-emotion examples when both sides are recognized
- Emoji handling and slang additions improve coverage for examples like `"Just had the best coffee ever! ☕"`

**Weaknesses of this approach:**  
- Fails when the sentence uses out-of-vocabulary positive or negative words
- Cannot reliably interpret sarcasm or irony
- Depends heavily on explicitly chosen lexicon entries
- Mixed moods still only register if both sides are lexicon-matched and score is zero
- Does not understand longer or more complex structures beyond token lookups

## 4. How the ML Model Works (if used)

**Features used:**  
The ML model uses bag-of-words features from `CountVectorizer`, which converts each sentence into counts of words.

**Training data:**  
It is trained on `SAMPLE_POSTS` and `TRUE_LABELS` from `dataset.py`.

**Training behavior:**  
When I added 4 new labeled examples, the ML model still achieved perfect training accuracy on the current dataset. This shows that it is highly sensitive to the labels and examples provided.

**Strengths and weaknesses:**  
- Strength: Learns patterns automatically from the labeled text and can capture associations not explicitly encoded in rules.
- Weakness: With only 15 examples, it easily fits the training set and may overfit, meaning it may not generalize to new sentences outside the dataset.
- Weakness: It can pick up spurious cues from very small datasets and is not robust to unseen phrasing.

## 5. Evaluation

**How you evaluated the model:**  
I evaluated both versions on the labeled examples in `dataset.py`. For the rule based model, I used `python main.py`; for the ML model, I used `python ml_experiments.py`.

**Observed accuracy:**  
- Rule-based model: `0.94` after the dataset expansion
- ML model: `1.00` training accuracy on the same dataset

**Examples of correct predictions:**  
- `"I love this class so much"` → positive, because it contains the positive lexicon word `love`.
- `"I am not happy about this"` → negative, because negation flips the positive word `happy` to a negative signal.
- `"Feeling like a million bucks today! 🤑"` → positive, because the emoji token and positive phrasing are recognized.

**Examples of incorrect predictions:**  
- Rule-based: `"I absolutely love getting stuck in traffic"` → predicted `positive`, true `negative`. The model misclassified it because it matched `love` as a positive word and had no rule for sarcastic context.
- Rule-based: before fixing lexicon entries, `"Feeling tired but kind of hopeful"` and `"Lowkey stressed but kind of proud of myself"` were wrong until `hopeful` and `proud` were added.
- ML: on this training set, the ML model did not show a failure, but that may be because it is memorizing the small labeled set.

## 6. Limitations

- The dataset is very small (15 examples) and not balanced across all possible mood expressions.
- The rule based model depends on the exact sentiment words in `POSITIVE_WORDS` and `NEGATIVE_WORDS`; missing words like `hopeful` or `proud` cause failures.
- The rule based model does not understand sarcasm. Example: `"I absolutely love getting stuck in traffic"` is labeled `negative`, but the model predicts `positive` because it sees `love` as positive.
- The rule based model still struggles with nuanced mixed feelings if one side of the sentiment is not in the lexicon.
- The ML model is sensitive to the specific labels in the training set and may overfit. Its perfect training accuracy does not guarantee generalization.
- The model is optimized for short, informal English phrases and may not work well for longer text, code-switching, or non-standard dialects.

## 7. Ethical Considerations

- Misclassifying a message with distress or sarcasm can lead to incorrect mood inference, which is risky if used in any decision-making application.
- The dataset contains slang and informal style, so the model may interpret some language communities better than others. It may misinterpret dialects or cultural references it was not trained on.
- Mood detection should not be used to make sensitive judgments about individuals without context or consent.
- This system is not a clinical tool and should not be used to infer mental health or emotional state in serious contexts.

## 8. Ideas for Improvement

- Add more labeled data, especially examples with sarcasm, mixed emotions, and varied slang.
- Use a real held-out test set instead of evaluating on training data only.
- Improve preprocessing to handle emojis and slang more robustly, perhaps with a dictionary of emoji sentiment.
- Add more lexicon entries or phrase-level rules for common constructions like `not bad`.
- Use TF-IDF or character n-grams instead of plain `CountVectorizer` for the ML model.
- Explore a small neural network or transformer if more labeled data becomes available.
- Add explicit rules for sarcasm or contextual negation if the goal is to improve the rule based model further.
