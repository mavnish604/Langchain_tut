from langchain_openai import ChatOpenAI 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings,ChatHuggingFace
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal
load_dotenv()
import os

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

usr_in=""""Upon a rigorous and comprehensive evaluation, the Motorola G62 5G presents itself not merely as another contender in a crowded market, but as a definitive statement of purposeful design and astute engineering. This device is, without reservation, an exemplary offering that decisively transcends its price point, delivering an experience that is both exceptionally refined and commendably robust.

The very essence of its appeal lies in the unadulterated purity of its software experience. Motorola's near-stock Android 12 implementation is a masterclass in efficiency, shedding the superfluous bloatware that plagues many of its contemporaries. The result is an interface that is breathtakingly fluid, responsive, and intuitively navigable. Every swipe, every application launch, every transition speaks of an underlying optimization that prioritizes the user's interaction above all else. This isn't just a phone with Android; it's Android as it was conceptually intended – swift, clean, and utterly dependable.

Performance, often a compromise in this segment, is here an outright triumph. The Snapdragon 480+ 5G chipset, far from being merely adequate, orchestrates a consistently smooth and lag-free operation. Multitasking is handled with an impressive grace, and day-to-day applications glide effortlessly. This device doesn't merely *cope* with demands; it actively *excels* in handling them, exhibiting a foundational stability that is immensely reassuring. The inclusion of true 5G capability is not just a bullet point; it is a fully realized promise of blistering data speeds and unwavering connectivity, fundamentally enhancing every aspect of digital interaction.

The visual and auditory experiences are equally compelling. The 6.5-inch Full HD+ display, with its magnificent 120Hz refresh rate, presents an arresting tapestry of vibrant colours and pristine clarity. Scrolling is a veritable delight, and media consumption becomes an engaging, immersive affair, free from the visual compromises often associated with devices in this category. Complementing this visual feast, the stereo speakers produce an audio output that is surprisingly rich, well-balanced, and projects with an authoritative presence, culminating in a truly satisfying multimedia ensemble.

Furthermore, the physical embodiment of the Motorola G62 5G is one of intelligent pragmatism. Its design eschews fleeting trends for an understated elegance and an ergonomic sensibility that ensures a comfortable and secure grip. The build quality conveys a palpable sense of durability and thoughtful construction, cementing its status as a reliable daily companion. Battery longevity, a critical metric for the discerning consumer, is addressed with an authoritative capacity, providing sustained power that easily endures prolonged usage, thereby liberating the user from the incessant anxiety of depletion.

In conclusive summation, the Motorola G62 5G is not merely a purchase; it is an investment in uncompromised quality and a superior user experience. It stands as an unequivocally decisive recommendation for those who demand excellence, efficiency, and steadfast reliability without venturing into the realm of excessive expenditure. This device unequivocally sets a new standard; it doesn't just meet expectations, it decisively and demonstrably exceeds them in every conceivable metric."""
parser = StrOutputParser()


class Feedback(BaseModel):
    sentiment : Literal["positive","negative"]=Field(description="give the sentiment of the feedback")
    summary : str = Field(description="give the summary of the user review in 5 lines")
parser2=PydanticOutputParser(pydantic_object=Feedback)

prompt1=PromptTemplate(
    template="classify the following {feedback} as positive or negative {format_ins}",
    input_variables=["feedback"],
    partial_variables={"format_ins":parser2.get_format_instructions()}
)


classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template="write an equal response to this positive  feedback {feedback}",
    input_variables=["feedback"]
)


prompt3 = PromptTemplate(
    template="write an appropriate response to this negative feedback {feedback}",
    input_variables=["feedback"]
)

branch_chain=RunnableBranch(
    (lambda x:x.sentiment == "positive",prompt2.invoke|model|parser),
    (lambda x:x.sentiment == "negative",prompt3.invoke|model|parser),
    RunnableLambda(lambda x:"could not find sentiment")
)

chain = classifier_chain | branch_chain
 
print(chain.invoke({"feedback":usr_in}))

chain.get_graph().print_ascii()