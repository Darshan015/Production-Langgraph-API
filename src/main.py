from dotenv import load_dotenv
load_dotenv()
from importlib.metadata import version

lang_core = version("langchain-core")
lg_graph = version("langgraph")

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

print(f"version of langchain-core {lang_core}")
print(f"version of langgraph {lg_graph}")


def main():
    llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)
    response = llm.invoke("Say the setup completed in your style")
    print(f"The final response from open AI lib {response.content}")

    llm_anth = ChatAnthropic(model_name="claude-haiku-5-5")
    response_ant = llm_anth.invoke("Say the setup completed in your style")
    print(f"The final response from Anthropic lib {response_ant.text}")


if __name__ =="__main__":
    main()

