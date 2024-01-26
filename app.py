import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp

load_dotenv() #loads open ai Key

# 1. Vectorise the sales response csv data
loader = CSVLoader(file_path="customer_emails_responses_man_withouttable_copy.csv")
documents = loader.load()

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

db = FAISS.from_documents(documents, embeddings)

# 2. Function for similarity search

def retrieve_info(query):
    similar_response = db.similarity_search(query, k=1)

    page_contents_array = [doc.page_content for doc in similar_response]

    # print(page_contents_array)

    return page_contents_array

prompt_template = """
You are a world class business development representative. 
I will share a prospect's message with you and you will give me the best answer that 
I should send to this prospect based on past best practies, 
and you will follow ALL of the rules below:

1/ Response should be very similar or even identical to the past best practies, 
in terms of length, ton of voice, logical arguments and other details

2/ If the best practice are irrelevant, then try to mimic the style of the best practice to prospect's message

Below is a message I received from the prospect:
{message}

Here is a list of best practies of how we normally respond to prospect in similar scenarios:
{best_practice}

Please write the best response that I should send to this prospect:
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=prompt_template
)

llm = LlamaCpp(
    model_path="darevox-7b.Q5_K_S.gguf",
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    verbose=True,  # Verbose is required to pass to the callback manager
)

chain = LLMChain(llm=llm, prompt=prompt)

# 4. Retrieval augmented generation
def generate_response(message):
    best_practice = retrieve_info(message)
    # Prepare the input for the LLMChain
    input_dict = {
        "message": message,
        "best_practice": best_practice
    }
    # Invoke the LLMChain with the prepared input
    response = chain.invoke(input= input_dict)
    return response


###### TESTING
# samplePrompt = """
# Dear Apple,

# I hope this email finds you well. I am writing to inquire about the possibility of placing a bulk order with your company and to learn more about any potential discounts or special pricing 
# that may be available for such orders.

# Sincerely,
# Prag
# """

# similaritySearchOutput = retrieve_info(samplePrompt)
# print(similaritySearchOutput)

# generatedResponse = generate_response(samplePrompt)
# print(generatedResponse)

def main():
    st.set_page_config(
        page_title="Customer response generator", page_icon=":bird:")

    st.header("Customer response generator :bird:")
    message = st.text_area("customer message")

    if message:
        st.write("Generating best practice message...")

        result = generate_response(message)

        st.info(result)


if __name__ == '__main__':
    main()