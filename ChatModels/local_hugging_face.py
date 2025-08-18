#!/home/tst_imperial/langchain/venv/bin/python
from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
llm=HuggingFacePipeline.from_model_id(
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v0.6",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.7,
        max_new_tokens=1000
)
)
model = ChatHuggingFace(llm=llm)
r=model.invoke("what is tthe capital of india")
print(r.content)