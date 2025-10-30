import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, Pipeline
from typing import Dict


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


COMPLAINT = "complaint_type"
POSITIVE_FEEDBACK = "positive_feedback_type"
REQUEST_INFO = "information_type"

NEGATIVE, POSITIVE, NEUTRAL = "NEGATIVE", "POSITIVE", "NEUTRAL"

APOLOGIZE_MSG = "We apologize for the inconvenience. Our team is already working on a solution."
POSITIVE_FEEDBACK_MSG = "Thank you for your feedback! We are very pleased that you are satisfied."
FORWARD_MSG = "Thank you for your message. We will forward it to the appropriate department."

NEGATIVE_WORDS = ["badly", "terrible", "hate", "error", "dissatisfied"]
POSITIVE_WORDS = ["good", "excellent", "love", "satisfied", "thanks"]


app = FastAPI(title="Customer Sentiment Orchestrator API")

sentiment_analyzer: Pipeline = None

def get_analyzer() -> Pipeline:
    """Return a singleton sentiment analysis pipeline (lazy-loaded)."""
    global sentiment_analyzer
    if sentiment_analyzer is None:
        try:
            sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            logger.info("Sentiment model loaded successfully")
        except Exception as e:
            sentiment_analyzer = None
            logger.error(f"Failed to load sentiment model: {e}")
    return sentiment_analyzer

class Request(BaseModel):
    """Schema for incoming user messages.

    :param message: text provided by user or system to analyze
    :type message: str
    """
    message: str

# --- Agent 1 — Request Classification ---
def classify_request(message: str) -> str:
    """Agent 1 — Classifies message into complaint, feedback, or info request.

    :param message: text content of the user request
    :type message: str
    :return: one of COMPLAINT, POSITIVE_FEEDBACK, or REQUEST_INFO
    :rtype: str
    """
    message_lower = message.lower()

    complaint_keywords = [
        "complaint", "problem", "error", "issue", "fault", "bug",
        "fail", "failure", "broken", "dissatisfied", "delay", "refund", "wrong"
    ]

    positive_keywords = [
        "thanks", "thank you", "good", "excellent", "love", "happy",
        "satisfied", "great", "awesome", "appreciate", "pleased", "fantastic"
    ]

    if any(word in message_lower for word in complaint_keywords):
        return COMPLAINT

    if any(word in message_lower for word in positive_keywords):
        return POSITIVE_FEEDBACK

    return REQUEST_INFO

# --- Rule-Based Sentiment ---
def analyze_sentiment_simple(message: str) -> str:
    """
    Perform simple sentiment analysis based on keyword matching.

    Returns:
        str: POSITIVE, NEGATIVE, or NEUTRAL.
    """
    message_lower = message.lower()

    if any(word in message_lower for word in NEGATIVE_WORDS):
        return NEGATIVE
    elif any(word in message_lower for word in POSITIVE_WORDS):
        return POSITIVE
    return NEUTRAL

# --- Agent 2: Transformer Sentiment Model ---
def analyze_sentiment(text: str) -> str:
    """Agent 2b — Performs sentiment analysis with lazy-loaded transformer model (1–5 star model).
    Falls back to simple keyword analysis if model is unavailable.

    :param text: message text to analyze
    :type text: str
    :return: sentiment label (POSITIVE, NEGATIVE, or NEUTRAL)
    :rtype: str
    """
    try:
        analyzer = get_analyzer()
        if analyzer is None:
            logger.warning("No model available, using simple keyword-based sentiment")
            return analyze_sentiment_simple(text)

        result = analyzer(text)[0] # [{'label': 'POSITIVE', 'score': 0.99}, ...]
        label = result.get("label", "")
        if NEGATIVE in label.upper():
            return NEGATIVE
        elif POSITIVE in label.upper():
            return POSITIVE
        return NEUTRAL
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return NEUTRAL

# --- Agent 3: Response Generator ---
def generate_response(classification: str, sentiment: str) -> str:
    """Agent 3 — Generates response message based on classification and sentiment.

    :param classification: request category (e.g., complaint or feedback)
    :type classification: str
    :param sentiment: detected sentiment from the text
    :type sentiment: str
    :return: message text suitable for end-user response
    :rtype: str
    """
    if classification == COMPLAINT and sentiment == NEGATIVE:
        return APOLOGIZE_MSG
    elif classification == POSITIVE_FEEDBACK:
        return POSITIVE_FEEDBACK_MSG
    return FORWARD_MSG

@app.post("/process")
def process(req: Request) -> Dict[str, str]:
    """Main orchestration endpoint.

    Handles request classification, sentiment analysis, and response synthesis.

    :param req: input request containing message text
    :type req: Request
    :return: dictionary containing classification, sentiment, and response text
    :rtype: dict
    :raises HTTPException: if processing fails or unhandled errors occur
    """
    try:
        logger.info(f"Processing message: {req.message}")

        classification = classify_request(req.message)
        sentiment = analyze_sentiment(req.message)
        response = generate_response(classification, sentiment)

        result = {
            "classification": classification,
            "sentiment": sentiment,
            "responseText": response
        }

        logger.info(f"Response generated: {result}")
        return result

    except Exception as e:
        logger.exception(f"Error in /process endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    from uvicorn import run
    run("app:app", host="0.0.0.0", port=port)
