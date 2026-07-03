# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

from typing import List, Dict, Tuple, Optional
import string

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        Improvements implemented:
          - Strips leading and trailing whitespace
          - Converts everything to lowercase
          - Removes punctuation (but keeps emojis for sentiment signals)
          - Splits on whitespace
        """
        # Strip whitespace and convert to lowercase
        cleaned = text.strip().lower()
        
        # Remove punctuation except for common emoji indicators
        # This preserves emojis while removing sentence punctuation
        punctuation = string.punctuation.replace(':', '')
        for char in punctuation:
            cleaned = cleaned.replace(char, ' ')
        
        # Split on whitespace and filter out empty tokens
        tokens = [t for t in cleaned.split() if t]

        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Enhancement: Negation handling
          - When "not" or "never" precedes a word, flip the sentiment
          - Looks ahead up to 3 tokens to find the sentiment word
          - Example: "not happy" → -1, "not a bad day" → +1
        """
        tokens = self.preprocess(text)
        score = 0
        
        # Negation words that flip the sentiment of the next word
        negation_words = {'not', 'never', 'no', 'dont', 'doesn', 'can\'t', 'cant', 'won\'t', 'wont'}
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            # Check if this is a negation word
            if token in negation_words:
                # Look ahead for the next sentiment word (within next 3 tokens)
                for j in range(i + 1, min(i + 4, len(tokens))):
                    next_token = tokens[j]
                    if next_token in self.positive_words:
                        score -= 1  # Flip positive to negative
                        i = j  # Skip to the sentiment word so we don't process it again
                        break
                    elif next_token in self.negative_words:
                        score += 1  # Flip negative to positive
                        i = j  # Skip to the sentiment word so we don't process it again
                        break
                else:
                    # No sentiment word found, just move past the negation
                    pass
            
            # Regular sentiment scoring
            elif token in self.positive_words:
                score += 1
            elif token in self.negative_words:
                score -= 1
            
            i += 1
        
        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        Mapping:
          - score > 0  → "positive"
          - score < 0  → "negative"
          - score == 0 → "neutral"
        
        Also detects "mixed" emotions when text contains both positive and negative words.
        """
        score = self.score_text(text)
        tokens = self.preprocess(text)
        
        # Check if text contains both positive and negative words
        has_positive = any(token in self.positive_words for token in tokens)
        has_negative = any(token in self.negative_words for token in tokens)
        
        # If score is close to zero and has mixed signals, label as "mixed"
        if score == 0 and has_positive and has_negative:
            return "mixed"
        
        # Otherwise use simple threshold
        if score > 0:
            return "positive"
        elif score < 0:
            return "negative"
        else:
            return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        tokens = self.preprocess(text)

        positive_hits: List[str] = []
        negative_hits: List[str] = []
        score = 0

        for token in tokens:
            if token in self.positive_words:
                positive_hits.append(token)
                score += 1
            if token in self.negative_words:
                negative_hits.append(token)
                score -= 1

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
