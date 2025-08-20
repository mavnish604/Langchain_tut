from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template="""
You are an expert AI researcher explaining the paper: "{usr_in_paper}".
Guidelines:
1. Use only the actual content and contributions from the paper "{usr_in_paper}".
2. If a detail is not in the paper, say "This detail is not explicitly mentioned in the paper."
3. Tailor your explanation style to: "{usr_in_style}".
4. Explanation length should match: "{usr_in_length}".
5. Avoid speculation, hallucinations, or unrelated details.
6. If needed, include equations, pseudocode, or code snippets ONLY if they directly help the explanation style "{usr_in_style}".
7. Keep clarity and factual accuracy above all.Now, explain "{usr_in_paper}" in a {usr_in_style} style with {usr_in_length} length.
""",
    input_variables=['usr_in_paper', 'usr_in_style', 'usr_in_length'],
    validate_template=True
)
template.save("template.json")