import os

import langchain
from dotenv import find_dotenv, load_dotenv
from langchain import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

sample_text = """

A car is a motor vehicle with four wheels that is designed for the transportation of people. It is one of the most common modes of transportation worldwide and has become an integral part of modern society.
Cars typically have seating for two to five people, although some larger models can accommodate more passengers. They are powered by internal combustion engines, which burn fuel to generate power. However, there are also electric cars that use electric motors and batteries for propulsion.
Cars come in various types and sizes, ranging from compact cars to SUVs, sedans, coupes, and sports cars. Each type has its own unique features and characteristics, catering to different needs and preferences.
The main components of a car include the engine, transmission, suspension, brakes, steering system, and electrical system. These components work together to provide a smooth and safe driving experience. Cars also have various safety features such as airbags, seat belts, anti-lock braking systems, and stability control systems to ensure the well-being of the occupants.
In addition to transportation, cars offer convenience and freedom. They allow individuals to travel long distances, commute to work, run errands, and explore new places at their own pace. Cars have also become a symbol of status and personal style, with many people choosing vehicles that reflect their personality and preferences.
However, cars also have environmental impacts. The burning of fossil fuels in traditional cars contributes to air pollution and greenhouse gas emissions, which are major concerns for the environment. As a result, there has been a growing interest in electric and hybrid cars, which are more environmentally friendly alternatives.
Overall, cars have revolutionized transportation and have become an essential part of modern life. They offer convenience, freedom, and the ability to explore the world around us. However, it is important to use cars responsibly and consider their environmental impact.

"""

def first_test(llm):
    # Map
    map_template = """The following is a set of documents
    {docs}
    Based on this list of docs, please identify the main themes 
    Helpful Answer:"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)
    return map_chain


def second_test(llm):
    chain = load_summarize_chain(llm, chain_type="stuff")
    return chain


def third_test(llm):
    template = "summarize this text {thing} in one line"

    prompt_template = PromptTemplate(input_variables=["thing"], template=template)
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain


def main():
    load_dotenv(find_dotenv())
    os.environ.get("OPENAI_API_KEY")

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-16k",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    chain = third_test(llm)
    output = chain.run(sample_text)
    print(output)


if __name__ == "__main__":
    main()
