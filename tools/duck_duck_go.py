from langchain_community.tools import DuckDuckGoSearchResults

search_multi = DuckDuckGoSearchResults(num_results=5)
print(search_multi.invoke("latest news India English"))
