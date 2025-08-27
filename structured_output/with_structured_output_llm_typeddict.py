from dotenv import load_dotenv
from typing import TypedDict
import os

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(dotenv_path="/home/tst_imperial/langchain/.env")

gemini_key = os.getenv("GOOGLE_API_KEY")
if not gemini_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Load Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_key,
    temperature=0.7,
    max_output_tokens=1000
)

# schema
class Review(TypedDict):
    summary: str
    sentiment: str

structured_model = model.with_structured_output(Review)

result = structured_model.invoke(
    """I didn’t just buy a coffee maker — I adopted a wizard that lives on my kitchen counter. 
    Every morning, it performs a tiny miracle: water goes in, bliss comes out. 
    The aroma sneaks through the house like a secret melody, and by the time the mug is in my hand, 
    I swear I hear applause from the universe itself.

    The design? Sleek enough to make my toaster jealous. 
    The buttons? So intuitive, my cat almost brewed herself a latte. 
    And the taste… let’s just say I now understand why poets compared love to a warm cup.

    If happiness had a flavor, this machine would serve it at 7 AM sharp. 
    Absolute 10/10 — I’m in a committed relationship with my coffee maker now."""
)

print(result)
