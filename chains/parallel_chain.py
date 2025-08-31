from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace,HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from langchain_core.prompts import PromptTemplate 
from langchain_openai import ChatOpenAI
import os

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

llm = HuggingFaceEndpoint(
    repo_id="NousResearch/Hermes-4-405B",
    task="text-generation"
)
model1 = ChatHuggingFace(llm=llm)

model2 = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

model3 = ChatOpenAI(
    api_key=groq_key,
    base_url="https://api.groq.com/openai/v1",
    model="llama3-70b-8192",
    temperature=2,
    max_completion_tokens=1000
)

text =""""Gemma 3 model card
Model Page: Gemma

Resources and Technical Documentation:

Gemma 3 Technical Report
Responsible Generative AI Toolkit
Gemma on Kaggle
Gemma on Vertex Model Garden
Terms of Use: Terms

Authors: Google DeepMind

Model Information
Summary description and brief definition of inputs and outputs.

Description
Gemma is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models. Gemma 3 models are multimodal, handling text and image input and generating text output, with open weights for both pre-trained variants and instruction-tuned variants. Gemma 3 has a large, 128K context window, multilingual support in over 140 languages, and is available in more sizes than previous versions. Gemma 3 models are well-suited for a variety of text generation and image understanding tasks, including question answering, summarization, and reasoning. Their relatively small size makes it possible to deploy them in environments with limited resources such as laptops, desktops or your own cloud infrastructure, democratizing access to state of the art AI models and helping foster innovation for everyone.

Inputs and outputs
Input:

Text string, such as a question, a prompt, or a document to be summarized
Images, normalized to 896 x 896 resolution and encoded to 256 tokens each, for the 4B, 12B, and 27B sizes.
Total input context of 128K tokens for the 4B, 12B, and 27B sizes, and 32K tokens for the 1B and 270M sizes.
Output:

Generated text in response to the input, such as an answer to a question, analysis of image content, or a summary of a document
Total output context up to 128K tokens for the 4B, 12B, and 27B sizes, and 32K tokens for the 1B and 270M sizes per request, subtracting the request input tokens
Citation
@article{gemma_2025,
    title={Gemma 3},
    url={https://arxiv.org/abs/2503.19786},
    publisher={Google DeepMind},
    author={Gemma Team},
    year={2025}
}

Model Data
Data used for model training and how the data was processed.

Training Dataset
These models were trained on a dataset of text data that includes a wide variety of sources. The 27B model was trained with 14 trillion tokens, the 12B model was trained with 12 trillion tokens, 4B model was trained with 4 trillion tokens, the 1B with 2 trillion tokens, and the 270M with 6 trillion tokens. The knowledge cutoff date for the training data was August 2024. Here are the key components:

Web Documents: A diverse collection of web text ensures the model is exposed to a broad range of linguistic styles, topics, and vocabulary. The training dataset includes content in over 140 languages.
Code: Exposing the model to code helps it to learn the syntax and patterns of programming languages, which improves its ability to generate code and understand code-related questions.
Mathematics: Training on mathematical text helps the model learn logical reasoning, symbolic representation, and to address mathematical queries.
Images: A wide range of images enables the model to perform image analysis and visual data extraction tasks.
The combination of these diverse data sources is crucial for training a powerful multimodal model that can handle a wide variety of different tasks and data formats.

Data Preprocessing
Here are the key data cleaning and filtering methods applied to the training data:

CSAM Filtering: Rigorous CSAM (Child Sexual Abuse Material) filtering was applied at multiple stages in the data preparation process to ensure the exclusion of harmful and illegal content.
Sensitive Data Filtering: As part of making Gemma pre-trained models safe and reliable, automated techniques were used to filter out certain personal information and other sensitive data from training sets.
Additional methods: Filtering based on content quality and safety in line with our policies.
Implementation Information
Details about the model internals.

Hardware
Gemma was trained using Tensor Processing Unit (TPU) hardware (TPUv4p, TPUv5p and TPUv5e). Training vision-language models (VLMS) requires significant computational power. TPUs, designed specifically for matrix operations common in machine learning, offer several advantages in this domain:

Performance: TPUs are specifically designed to handle the massive computations involved in training VLMs. They can speed up training considerably compared to CPUs.
Memory: TPUs often come with large amounts of high-bandwidth memory, allowing for the handling of large models and batch sizes during training. This can lead to better model quality.
Scalability: TPU Pods (large clusters of TPUs) provide a scalable solution for handling the growing complexity of large foundation models. You can distribute training across multiple TPU devices for faster and more efficient processing.
Cost-effectiveness: In many scenarios, TPUs can provide a more cost-effective solution for training large models compared to CPU-based infrastructure, especially when considering the time and resources saved due to faster training.
These advantages are aligned with Google's commitments to operate sustainably.
Software
Training was done using JAX and ML Pathways.

JAX allows researchers to take advantage of the latest generation of hardware, including TPUs, for faster and more efficient training of large models. ML Pathways is Google's latest effort to build artificially intelligent systems capable of generalizing across multiple tasks. This is specially suitable for foundation models, including large language models like these ones.

Together, JAX and ML Pathways are used as described in the paper about the Gemini family of models; "the 'single controller' programming model of Jax and Pathways allows a single Python process to orchestrate the entire training run, dramatically simplifying the development workflow."

Evaluation
Model evaluation metrics and results.

Benchmark Results
These models were evaluated against a large collection of different datasets and metrics to cover different aspects of text generation. Evaluation results marked with IT are for instruction-tuned models. Evaluation results marked with PT are for pre-trained models.

Gemma 3 270M
Benchmark	n-shot	Gemma 3 PT 270M
HellaSwag	10-shot	40.9
BoolQ	0-shot	61.4
PIQA	0-shot	67.7
TriviaQA	5-shot	15.4
ARC-c	25-shot	29.0
ARC-e	0-shot	57.7
WinoGrande	5-shot	52.0
Benchmark	n-shot	Gemma 3 IT 270m
HellaSwag	0-shot	37.7
PIQA	0-shot	66.2
ARC-c	0-shot	28.2
WinoGrande	0-shot	52.3
BIG-Bench Hard	few-shot	26.7
IF Eval	0-shot	51.2
Gemma 3 1B, 4B, 12B & 27B
Reasoning and factuality
Benchmark	n-shot	Gemma 3 IT 1B	Gemma 3 IT 4B	Gemma 3 IT 12B	Gemma 3 IT 27B
GPQA Diamond	0-shot	19.2	30.8	40.9	42.4
SimpleQA	0-shot	2.2	4.0	6.3	10.0
FACTS Grounding	-	36.4	70.1	75.8	74.9
BIG-Bench Hard	0-shot	39.1	72.2	85.7	87.6
BIG-Bench Extra Hard	0-shot	7.2	11.0	16.3	19.3
IFEval	0-shot	80.2	90.2	88.9	90.4
Benchmark	n-shot	Gemma 3 PT 1B	Gemma 3 PT 4B	Gemma 3 PT 12B	Gemma 3 PT 27B
HellaSwag	10-shot	62.3	77.2	84.2	85.6
BoolQ	0-shot	63.2	72.3	78.8	82.4
PIQA	0-shot	73.8	79.6	81.8	83.3
SocialIQA	0-shot	48.9	51.9	53.4	54.9
TriviaQA	5-shot	39.8	65.8	78.2	85.5
Natural Questions	5-shot	9.48	20.0	31.4	36.1
ARC-c	25-shot	38.4	56.2	68.9	70.6
ARC-e	0-shot	73.0	82.4	88.3	89.0
WinoGrande	5-shot	58.2	64.7	74.3	78.8
BIG-Bench Hard	few-shot	28.4	50.9	72.6	77.7
DROP	1-shot	42.4	60.1	72.2	77.2
STEM and code
Benchmark	n-shot	Gemma 3 IT 1B	Gemma 3 IT 4B	Gemma 3 IT 12B	Gemma 3 IT 27B
MMLU (Pro)	0-shot	14.7	43.6	60.6	67.5
LiveCodeBench	0-shot	1.9	12.6	24.6	29.7
Bird-SQL (dev)	-	6.4	36.3	47.9	54.4
Math	0-shot	48.0	75.6	83.8	89.0
HiddenMath	0-shot	15.8	43.0	54.5	60.3
MBPP	3-shot	35.2	63.2	73.0	74.4
HumanEval	0-shot	41.5	71.3	85.4	87.8
Natural2Code	0-shot	56.0	70.3	80.7	84.5
GSM8K	0-shot	62.8	89.2	94.4	95.9
Benchmark	n-shot	Gemma 3 PT 4B	Gemma 3 PT 12B	Gemma 3 PT 27B
MMLU	5-shot	59.6	74.5	78.6
MMLU (Pro COT)	5-shot	29.2	45.3	52.2
AGIEval	3-5-shot	42.1	57.4	66.2
MATH	4-shot	24.2	43.3	50.0
GSM8K	8-shot	38.4	71.0	82.6
GPQA	5-shot	15.0	25.4	24.3
MBPP	3-shot	46.0	60.4	65.6
HumanEval	0-shot	36.0	45.7	48.8
Multilingual
Benchmark	n-shot	Gemma 3 IT 1B	Gemma 3 IT 4B	Gemma 3 IT 12B	Gemma 3 IT 27B
Global-MMLU-Lite	0-shot	34.2	54.5	69.5	75.1
ECLeKTic	0-shot	1.4	4.6	10.3	16.7
WMT24++	0-shot	35.9	46.8	51.6	53.4
Benchmark	Gemma 3 PT 1B	Gemma 3 PT 4B	Gemma 3 PT 12B	Gemma 3 PT 27B
MGSM	2.04	34.7	64.3	74.3
Global-MMLU-Lite	24.9	57.0	69.4	75.7
WMT24++ (ChrF)	36.7	48.4	53.9	55.7
FloRes	29.5	39.2	46.0	48.8
XQuAD (all)	43.9	68.0	74.5	76.8
ECLeKTic	4.69	11.0	17.2	24.4
IndicGenBench	41.4	57.2	61.7	63.4
Multimodal
Benchmark	Gemma 3 IT 4B	Gemma 3 IT 12B	Gemma 3 IT 27B
MMMU (val)	48.8	59.6	64.9
DocVQA	75.8	87.1	86.6
InfoVQA	50.0	64.9	70.6
TextVQA	57.8	67.7	65.1
AI2D	74.8	84.2	84.5
ChartQA	68.8	75.7	78.0
VQAv2 (val)	62.4	71.6	71.0
MathVista (testmini)	50.0	62.9	67.6
Benchmark	Gemma 3 PT 4B	Gemma 3 PT 12B	Gemma 3 PT 27B
COCOcap	102	111	116
DocVQA (val)	72.8	82.3	85.6
InfoVQA (val)	44.1	54.8	59.4
MMMU (pt)	39.2	50.3	56.1
TextVQA (val)	58.9	66.5	68.6
RealWorldQA	45.5	52.2	53.9
ReMI	27.3	38.5	44.8
AI2D	63.2	75.2	79.0
ChartQA	63.6	74.7	76.3
VQAv2	63.9	71.2	72.9
BLINK	38.0	35.9	39.6
OKVQA	51.0	58.7	60.2
TallyQA	42.5	51.8	54.3
SpatialSense VQA	50.9	60.0	59.4
CountBenchQA	26.1	17.8	68.0
Ethics and Safety
Ethics and safety evaluation approach and results."""

prompt1 = PromptTemplate(
    template="generate a 100-200 words summary based on {text}",
    input_variables=["text"]
)
prompt2 = PromptTemplate(
    template="generate a 5-7 questions quiz based on {text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="combine both the {summary} and {quizes} in a single document in a proper formated manner",
    input_variables=["summary","quizes"]
)

parser=StrOutputParser()

parallel = RunnableParallel({
    "summary": prompt1 | model1 | parser,
    "quizes": prompt2 | model2 | parser
})

chain = parallel |prompt3 |model3 | parser
print(chain.invoke({"text":text}))

chain.get_graph().print_ascii()