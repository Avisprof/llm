from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

bloom = HuggingFacePipeline.from_model_id(
    model_id="bigscience/bloom-1b7",
    task="text-generation",
    model_kwargs={"max_length": 64},
    #device=0,  # Номер GPU карточки, если есть!
)

template = """Question: {question}.

Answer:"""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(
    prompt=prompt,
    llm=bloom
)

question = "When did man first fly into space?"

print(llm_chain.invoke(question)['text'])