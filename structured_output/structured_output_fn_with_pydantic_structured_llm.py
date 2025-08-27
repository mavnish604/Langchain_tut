from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from typing import Optional,Literal,Annotated

load_dotenv()
model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
class Review(BaseModel):
    key_themes : list[str] = Field(description="write all key themes discussed in the review")
    summary : str = Field(description="a beif summary of review")
    sentiment: Literal["pos","neg"] = Field(description="Return sentiment of the user")
    pros: Optional[list[str]] = Field(default=None,description="return all the pros mentioned")
    cons : Optional[list[str]] =Field(default=None,description="return all the cons mentioned in review")
    name : Optional[str]=Field(default=None,description="return name")
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
    Absolute 10/10 — I’m in a committed relationship with my coffee maker now.
    --Avnish Mishra
    """
)

print(dict(result))
