from dotenv import load_dotenv
load_dotenv()
import os
import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from pdf import canada_engine, south_korea_engine


assert os.getenv("OPENAI_API_KEY") is not None
population_path = os.path.join("data", "population.csv")
population_df = pd.read_csv(population_path)


population_query_engine = PandasQueryEngine(
    df=population_df,
    verbose=True,
    instruction_str=instruction_str,
)

population_query_engine.update_prompts({"pandas_prompt": new_prompt})

tools = [
    note_engine,
    QueryEngineTool(
        query_engine=population_query_engine,
        metadata=ToolMetadata(
            name="population_data",
            description="This gives information about the world population and demographics.",
        ),
    ),
    QueryEngineTool(
        query_engine=canada_engine,
        metadata=ToolMetadata(
            name="canada_data",
            description="This gives detailed information about canada the country.",
        ),
    ),
    QueryEngineTool(
        query_engine=south_korea_engine,
        metadata=ToolMetadata(
            name="south_korea_data",
            description="This gives detailed information about South Korea the country.",
        ),
    )
]

llm = OpenAI(
    model="gpt-4o-mini-2024-07-18",
    )
agent = ReActAgent.from_tools(
    tools=tools,
    llm=llm,
    verbose=True,
    context=context
)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)