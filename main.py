from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class RequestBody(BaseModel):
    message: str

@app.post("/process")
def process_request(body: RequestBody):
    response = f"Python received your message: {body.message}"
    return {"responseText": response}
