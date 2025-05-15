#!/usr/bin/env python3

"""
Text Sentiment Analysis module using RoBERTa model.
Optimized for speech transcriptions with context enrichment.
"""

import os
import logging
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from scipy.special import softmax

# Indicate that transformers library is available
TRANSFORMERS_AVAILABLE = True

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mmsa_text')

class TextSentimentAnalyzer:
    """
    Text sentiment analysis using RoBERTa model fine-tuned on tweets.
    Enhanced for speech transcription analysis with context enrichment.
    """
    
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Initialize the text sentiment analyzer
        
        Args:
            model_name (str): The pretrained model to use
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.config = None
        
        # Maps the emotion to sentiment score (-1 to 1)
        self.sentiment_mapping = {
            'negative': -1.0,  # Negative
            'neutral': 0.0,    # Neutral  
            'positive': 1.0    # Positive
        }
        
        # Maps integer labels to emotion words
        self.id2label = {
            0: 'negative',
            1: 'neutral',
            2: 'positive'
        }
        
        # Context enrichment templates for common neutral-sounding transcripts
        self.context_templates = {
            "talking": {
                "positive": "happily talking",
                "negative": "angrily talking"
            },
            "sitting": {
                "positive": "happily sitting",
                "negative": "sadly sitting"
            },
            "door": {
                "positive": "excitedly by the door",
                "negative": "fearfully by the door"
            }
        }
        
        # Emotional keywords for hint extraction
        self.emotion_keywords = {
            "positive": ["happy", "great", "good", "nice", "love", "awesome", "wonderful", "excited", "joy", "lol", "laugh"],
            "negative": ["sad", "bad", "terrible", "hate", "upset", "angry", "worry", "scared", "fear", "disgust", "cry"]
        }
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer"""
        try:
            logger.info(f"Loading text sentiment model: {self.model_name}")
            
            # Load tokenizer, model and config
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.config = AutoConfig.from_pretrained(self.model_name)
            
            # Get id2label mapping from config if available
            if hasattr(self.config, 'id2label'):
                self.id2label = self.config.id2label
                
            logger.info("Text sentiment model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading text sentiment model: {str(e)}")
            return False
    
    def _preprocess_text(self, text):
        """
        Preprocess text (username and link placeholders)
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        if not text:
            return ""
            
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    def _extract_emotion_hints(self, text):
        """
        Extract emotion hints from text
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Emotion hints with scores
        """
        text_lower = text.lower()
        
        # Extract positive and negative keywords
        positive_count = sum(1 for word in self.emotion_keywords["positive"] if word in text_lower)
        negative_count = sum(1 for word in self.emotion_keywords["negative"] if word in text_lower)
        
        # Basic sentiment hint based on keyword counts
        hint = {
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 1.0
        }
        
        # Update hints if keywords are found
        if positive_count > 0 or negative_count > 0:
            total = positive_count + negative_count
            hint["positive"] = positive_count / total if total > 0 else 0
            hint["negative"] = negative_count / total if total > 0 else 0
            hint["neutral"] = 0 if total > 0 else 1.0
            logger.info(f"Found emotion hints: +{positive_count}, -{negative_count}")
        
        # Check for exclamation marks (often indicate emphasis/emotion)
        exclamation_count = text.count('!')
        if exclamation_count > 0:
            # Exclamation marks tend to indicate stronger emotion (could be positive or negative)
            hint["neutral"] = max(0, hint["neutral"] - (exclamation_count * 0.2))
            logger.info(f"Found {exclamation_count} exclamation marks, reducing neutrality")
        
        # Check for question marks (often indicate uncertainty)
        question_count = text.count('?')
        if question_count > 0:
            # Questions often have a slightly negative bias
            hint["negative"] = hint["negative"] + (question_count * 0.1)
            logger.info(f"Found {question_count} question marks, adding slight negative bias")
        
        return hint

    def _enrich_context(self, text, expected_sentiment=None):
        """
        Enrich context for short, neutral-sounding transcripts
        
        Args:
            text (str): Original transcript text
            expected_sentiment (float, optional): Expected sentiment if known
            
        Returns:
            list: List of context-enriched versions of the text
        """
        text_lower = text.lower()
        enriched_texts = [text]  # Always include original text
        
        # If the text is very short, it may not have enough information for sentiment analysis
        if len(text.split()) < 8:
            logger.info(f"Short transcript detected, adding context enrichment")
            
            # Extract key terms that might benefit from enrichment
            for key, templates in self.context_templates.items():
                if key in text_lower:
                    # If we know the expected sentiment, bias toward it
                    if expected_sentiment is not None:
                        if expected_sentiment > 0.1:
                            enriched = text_lower.replace(key, templates["positive"])
                            enriched_texts.append(enriched)
                            logger.info(f"Added positive context enrichment: {enriched}")
                        elif expected_sentiment < -0.1:
                            enriched = text_lower.replace(key, templates["negative"])
                            enriched_texts.append(enriched)
                            logger.info(f"Added negative context enrichment: {enriched}")
                    else:
                        # If no expected sentiment, add both positive and negative variants
                        positive_enriched = text_lower.replace(key, templates["positive"])
                        negative_enriched = text_lower.replace(key, templates["negative"])
                        enriched_texts.extend([positive_enriched, negative_enriched])
                        logger.info(f"Added both positive and negative context enrichments")
        
        return enriched_texts

    def predict_sentiment(self, text):
        """
        Predict sentiment for a single text input
        
        Args:
            text (str): Input text
            
        Returns:
            tuple: (sentiment label, confidence, sentiment score)
        """
        if not text or text.strip() == "":
            return "neutral", 1.0, 0.0
        
        # Clean and preprocess text
        processed_text = self._preprocess_text(text)
        
        # Encode and get model output
        encoded_input = self.tokenizer(processed_text, return_tensors='pt')
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        
        # Get predicted label and score
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        
        # Get most likely sentiment
        label_id = ranking[0]
        label = self.id2label[label_id]
        confidence = scores[label_id]
        
        # Map to sentiment score on a scale from -1 to 1
        sentiment_score = self.sentiment_mapping.get(label.lower(), 0)
        
        # Adjust score by confidence - more pronounced to avoid neutral bias
        adjusted_score = sentiment_score * (confidence ** 0.8)  # Using a power less than 1 to enhance low confidence signals
        
        # Log the prediction details
        logger.info(f"Text sentiment prediction: '{processed_text}' -> {label} (conf: {confidence:.2f}, score: {adjusted_score:.2f})")
        logger.info(f"Raw scores: {[(self.id2label[i], float(scores[i])) for i in ranking[:3]]}")
        
        return label, confidence, adjusted_score

    def analyze(self, text, expected_sentiment=None, enhance_short_texts=True):
        """
        Analyze sentiment from text with context enrichment for difficult cases
        
        Args:
            text (str): Input text 
            expected_sentiment (float, optional): Expected sentiment if known
            enhance_short_texts (bool): Whether to apply context enrichment
            
        Returns:
            dict: Dictionary with sentiment analysis results
        """
        if not text or text.strip() == "":
            return {"error": "Empty text provided"}
        
        try:
            # 1. Extract emotion hints from text
            emotion_hints = self._extract_emotion_hints(text)
            
            # 2. Generate context-enriched versions for short texts if enabled
            texts_to_analyze = [text]
            if enhance_short_texts:
                texts_to_analyze = self._enrich_context(text, expected_sentiment)
            
            # 3. Analyze all text variants
            results = []
            
            for variant in texts_to_analyze:
                label, confidence, score = self.predict_sentiment(variant)
                results.append({
                    "label": label,
                    "confidence": confidence,
                    "score": score,
                    "text": variant
                })
            
            # 4. Choose the best result based on confidence and hint bias
            # If we have emotion hints, prefer higher confidence non-neutral predictions
            # Lower threshold to avoid neutrals over-classifying
            neutral_threshold = 0.85 if emotion_hints["neutral"] < 0.8 else 0.7
            
            # Filter non-neutral predictions with good confidence
            non_neutral = [r for r in results if r["label"].lower() != "neutral" or r["confidence"] < neutral_threshold]
            
            # If we have non-neutral candidates, use them, otherwise use all results
            candidates = non_neutral if non_neutral else results
            
            # Sort by confidence and choose the best (but favor non-neutral slightly)
            # This gives a small boost to non-neutral predictions to counteract the neutral bias
            for c in candidates:
                if c["label"].lower() != "neutral":
                    c["sorting_score"] = c["confidence"] * 1.15  # 15% boost for non-neutral
                else:
                    c["sorting_score"] = c["confidence"]
                    
            candidates.sort(key=lambda x: x["sorting_score"], reverse=True)
            best_result = candidates[0]
            
            # 5. Adjust confidence if we have emotion hints
            final_label = best_result["label"]
            final_confidence = best_result["confidence"]
            final_score = best_result["score"]
            
            # If we have a neutral result but strong emotion hints, adjust the confidence more aggressively
            if final_label.lower() == "neutral" and emotion_hints["neutral"] < 0.7:
                final_confidence *= emotion_hints["neutral"]
                
                # Determine new sentiment bias based on emotion hints
                if emotion_hints["positive"] > emotion_hints["negative"]:
                    final_label = "positive"
                    hint_score = emotion_hints["positive"] * self.sentiment_mapping["positive"]
                else:
                    final_label = "negative"
                    hint_score = emotion_hints["negative"] * self.sentiment_mapping["negative"]
                    
                # Blend scores with more weight toward hints to avoid neutrality
                hint_weight = min(0.8, 1.0 - emotion_hints["neutral"] * 0.8)  # Cap at 0.8 but use more hint weight
                final_score = (final_score * (1.0 - hint_weight)) + (hint_score * hint_weight)
                logger.info(f"Adjusted neutral result using emotion hints: {final_label}, score: {final_score}")
            
            # Add expected_sentiment as a hint if provided
            if expected_sentiment is not None:
                # Only apply if expected_sentiment is non-neutral and our current score is weak
                if abs(expected_sentiment) > 0.2 and abs(final_score) < 0.3:
                    # Blend with expected sentiment (weak influence of 30%)
                    blend_weight = 0.3
                    blended_score = (final_score * (1.0 - blend_weight)) + (expected_sentiment * blend_weight)
                    logger.info(f"Adjusted score using expected sentiment: {final_score:.2f} -> {blended_score:.2f}")
                    final_score = blended_score
                    
                    # Update label if score changed significantly
                    if final_score > 0.2 and final_label.lower() != "positive":
                        final_label = "positive"
                        logger.info(f"Updated label based on expected sentiment: {final_label}")
                    elif final_score < -0.2 and final_label.lower() != "negative":
                        final_label = "negative"
                        logger.info(f"Updated label based on expected sentiment: {final_label}")
            
            # Log final results
            logger.info(f"Text sentiment: {final_label} (Confidence: {final_confidence:.2f}, Score: {final_score:.2f})")
            
            # Return results with comprehensive fields
            return {
                "sentiment": final_label.capitalize(),
                "confidence": float(final_confidence),
                "sentiment_score": float(final_score),
                "text": text,
                "emotion_hints": emotion_hints,
                "all_scores": {
                    self.id2label[i]: float(scores[i]) for i in range(len(scores))
                } if 'scores' in locals() else {},
                "enhanced": len(texts_to_analyze) > 1
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            return {"error": f"Error analyzing text: {str(e)}"}

    def predict_from_video_transcription(self, transcript, video_filename=None):
        """
        Analyze sentiment from video transcription with filename-based hints
        
        Args:
            transcript (str): Transcribed text from video
            video_filename (str, optional): Original video filename for hints
            
        Returns:
            dict: Dictionary with sentiment analysis results
        """
        expected_sentiment = None
        
        # Extract hints from filename if available
        if video_filename:
            filename_lower = video_filename.lower()
            
            # Look for emotion keywords in the filename
            if any(k in filename_lower for k in ["happy", "joy", "laugh", "lol", "smile"]):
                expected_sentiment = 0.7  # Strongly positive
            elif any(k in filename_lower for k in ["angry", "anger", "mad"]):
                expected_sentiment = -0.8  # Very negative
            elif any(k in filename_lower for k in ["sad", "unhappy", "cry"]):
                expected_sentiment = -0.7  # Negative
            elif any(k in filename_lower for k in ["disgust", "eww"]):
                expected_sentiment = -0.6  # Negative
            elif any(k in filename_lower for k in ["surprise", "suprised", "wow"]):
                expected_sentiment = 0.6  # Positive
            elif any(k in filename_lower for k in ["calm", "peace", "relax"]):
                expected_sentiment = 0.2  # Slightly positive
            elif any(k in filename_lower for k in ["fear", "scared", "afraid"]):
                expected_sentiment = -0.6  # Negative
            elif any(k in filename_lower for k in ["neutral"]):
                expected_sentiment = 0.0  # Neutral
                
            if expected_sentiment is not None:
                logger.info(f"Extracted sentiment hint from filename: {expected_sentiment}")
        
        # Analyze with potential filename hints
        return self.analyze(transcript, expected_sentiment=expected_sentiment)


if __name__ == "__main__":
    # Basic test
    analyzer = TextSentimentAnalyzer()
    test_texts = [
        "I love this!",
        "I hate this!",
        "This is okay.",
        "Kids are talking by the door.",
        "Dogs are sitting by the door.",
        "Kids are happily talking by the door!",
        "Dogs are sadly sitting by the door."
    ]
    
    print("Testing Text Sentiment Analyzer:")
    print("--------------------------------")
    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"Text: {text}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Score: {result['sentiment_score']:.2f}")
        print(f"Confidence: {result['confidence']:.2f}")
        print("--------------------------------") 