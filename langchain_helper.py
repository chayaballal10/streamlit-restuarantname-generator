from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

def generate_restaurant_name_and_items(cuisine: str):
    # Prompt 1
    name_prompt = PromptTemplate.from_template(
        "I want to open a restaurant for {cuisine} food. Suggest a fancy name for this."
    )

    # Prompt 2
    menu_prompt = PromptTemplate.from_template(
        "Suggest some menu items for {restaurant_name}. Return it as a comma separated string."
    )

    # LCEL chains
    name_chain = name_prompt | llm
    menu_chain = menu_prompt | llm

    # Run chain 1
    restaurant_name = name_chain.invoke({"cuisine": cuisine}).content

    # Run chain 2
    menu_items = menu_chain.invoke(
        {"restaurant_name": restaurant_name}
    ).content

    return {
        "restaurant_name": restaurant_name,
        "menu_items": menu_items
    }
