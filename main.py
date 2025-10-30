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

NEGATIVE = "Negative"
POSITIVE = "Positive"
NEUTRAL = "Neutral"

APOLOGIZE_MSG = "We apologize for the inconvenience. Our team is already working on a solution."
POSITIVE_FEEDBACK_MSG = "Thank you for your feedback! We are very pleased that you are satisfied."
FORWARD_MSG = "Thank you for your message. We will forward it to the appropriate department."

NEGATIVE_WORDS = ["badly", "terrible", "hate", "error", "dissatisfied"]
POSITIVE_WORDS = ["good", "excellent", "love", "satisfied", "thanks"]


app = FastAPI(title="Customer Sentiment Orchestrator API")

try:
    sentiment_analyzer: Pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    logger.info("Sentiment analysis model loaded successfully.")
except Exception as e:
    sentiment_analyzer = None
    logger.error(f"Failed to load sentiment model: {e}")

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

# --- Agent 2 (Option 1): Rule-Based Sentiment ---
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

# --- Agent 2 (Option 2): Transformer Sentiment Model ---
def analyze_sentiment_three_class(text: str) -> str:
    """Agent 2b — Performs transformer-based sentiment analysis (1–5 star model).
    Falls back to simple keyword analysis if model is unavailable.

    :param text: message text to analyze
    :type text: str
    :return: sentiment label (POSITIVE, NEGATIVE, or NEUTRAL)
    :rtype: str
    """
    if not sentiment_analyzer:
        logger.warning("Sentiment model unavailable. Falling back to simple rules.")
        return analyze_sentiment_simple(text)

    try:
        result = sentiment_analyzer(text)[0]
        label = result.get("label", "")
        stars = int(label.split()[0]) if label and label[0].isdigit() else 3  # default neutral

        if stars <= 2:
            return NEGATIVE
        elif stars == 3:
            return NEUTRAL
        else:
            return POSITIVE
    except Exception as e:
        logger.error(f"Error during sentiment analysis: {e}")
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
        sentiment = analyze_sentiment_three_class(req.message)
        response = generate_response(classification, sentiment)

        result = {
            "classification": classification,
            "sentiment": sentiment,
            "responseText": response
        }

        logger.info(f"Result: {result}")
        return result

    except Exception as e:
        logger.exception(f"Error in /process endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
