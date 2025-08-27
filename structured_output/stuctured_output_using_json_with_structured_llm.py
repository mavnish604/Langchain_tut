from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

#schema
json_schema={
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "write all key themes discussed in the review"
    },
    "summary": {
      "type": "string",
      "description": "a beif summary of review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Return sentiment of the user"
    },
    "pros": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "return all the pros mentioned"
    },
    "cons": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "return all the cons mentioned in review"
    },
    "name": {
      "type": "string",
      "description": "return name"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}

str_model = model.with_structured_output(json_schema)
res=str_model.invoke(    """I didn’t just buy a coffee maker — I adopted a wizard that lives on my kitchen counter. 
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
print(res)
