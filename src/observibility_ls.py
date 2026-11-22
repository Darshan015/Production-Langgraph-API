from langsmith import traceable
from langsmith.run_trees import RunTree
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
import os

#Enable tracing
os.environ["LANGSMITH_TRACING"] = "true"

@traceable(name="Run observibility")
def runObservibility():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature="0.2")
    prompt = ChatPromptTemplate.from_template("""
    
    Summarize in one line for this {query}
    
    """)
    chain = prompt | llm | StrOutputParser()
    res = chain.invoke("Evals in RAG")
    print(res)



if __name__ =="__main__":
    runObservibility()