import streamlit as st
from langchain_core.documents import Document
from langchain_community.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.base import BaseCallbackHandler
import re
import os


check_prompt = ChatPromptTemplate.from_template(
    """
        I'll give you history of questions and a new question.

        Calculate the similarities(from 0 to 1) of each previous questions comparing with the new one 
        and select the question that has the highest similarity.

        In case that more than 2 questions have the highest similarity,
        then choose the latest question based on the index number of question.

        If the similarity of the question you select is lower than 0.5,
        then return "Never been asked".

        Else, return the index number of question.
        Do NOT append any other comment.

        Examples:

        QnA History: 
        No history

        New question: How far away is the moon?
        Answer: Never been asked.

        ======================================

        QnA History: 
        0: What is the color of the ocean?
        1: What is the price of GPT-4 Turbo?
        2: Tell me the color of the ocean.

        New question: let me know the color of the ocean.
        Answer: 2
        

        Your turn!

        QnA History: 
        {query_hist}

        New question: {question}
    """
)

answers_prompt = ChatPromptTemplate.from_template(
        """
            Using ONLY the following context answer the user's question.
            If you can't just say you don't know, don't make anything up.

            Then, give a score to the answer between 0 and 5. 0 being not helpful to
            the user and 5 being helpful to the user.

            Make sure to include the answer's score.

            Context: {context}

            Examples:

            Question: How far away is the moon?
            Answer: The moon is 384,400 km away.
            Score: 5

            Question: How far away is the sun?
            Answer: I don't know.
            Score: 0

            Your turn!

            Question: {question}
        """
)

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's
            question.

            Use the answers that have the highest score (more helpful).

            Cite sources. Do not modify the source, keep it as a link.

            Answers: {answers}
            """
        ),
        (
            "human",
            "{question}"
        )
    ]
)


def save_query_hist(question, answer):
    st.session_state["query_hist"].append({"question": question, "answer": answer})

def check_previous_queries(query):
    check_chain = check_prompt | llm
    condensed_hist = "No history"
    if st.session_state["query_hist"]:
        query_hist = st.session_state["query_hist"]
        condensed_hist = "\n\n".join(f"{i}: {query_hist[i]['question']}\n"
                        for i in range(len(query_hist)))
    return check_chain.invoke({"query_hist": condensed_hist, "question": query})

def paint_answer(query, result):
    refined = result.replace("$", "\$")
    st.markdown(refined)
    save_query_hist(query, refined)

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]

    choose_chain = choose_prompt | llm

    condensed = "\n\n".join(f"Answer: {answer['answer']}\n\
                            Source: {answer['source']}\n" for answer in answers) #   Date: {answer['date']}\n
    
    return choose_chain.invoke({"question": question,"answers": condensed})

def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]

    answers_chain = answers_prompt | llm
 
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"context": doc.page_content, "question": question}
                    ).content,
                "source": doc.metadata["source"],
            } for doc in docs
        ]
    }

def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    navbar = soup.find("div", "navbar_container")
   
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    if navbar:
        navbar.decompose()
    
    refined = re.sub("([\n\s\t]|\xa0| {2,})", " ",str(soup.get_text()))
    return refined

@st.cache_resource(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )

    loader = SitemapLoader(
        url,
        parsing_function=parse_page
    )
    loader.requests_per_second = 5
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🔦"
)

st.title("SiteGPT")

with st.sidebar:
    with st.form("OpenAI API Key Setting"):
        user_openai_api_key = st.text_input("Enter your OpenAI API key.")
        submitted = st.form_submit_button("Set")
        if submitted:
            os.environ['OPENAI_API_KEY'] = user_openai_api_key

    url = st.text_input("Write down a URL", 
                        placeholder="https://example.com")
    
    st.link_button("Go to Github Repo", "https://github.com/hihighhye/SiteGPT")

# test_url = "https://developers.cloudflare.com/sitemap-0.xml"

if user_openai_api_key:
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-4.1-nano-2025-04-14",
        api_key=user_openai_api_key
    )

if not url:
    st.markdown(
                """
                Ask questions about the content of a website.

                Start by writing the URL of the website on the sidebar.
            """)
    
else:
    # async chromium loader
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")

    else:
        retriever = load_website(url)
        # docs = retriever.invoke("Can I use langchain in production?")
 
        query = st.text_input("Ask a question to the website.")
        
        if query:
            response = check_previous_queries(query)
            if response.content != "Never been asked":
               answer = st.session_state["query_hist"][int(response.content)]["answer"]
               paint_answer(query, answer)

            else:
                chain = {
                    "docs": retriever, 
                    "question": RunnablePassthrough(),
                } | RunnableLambda(get_answers) | RunnableLambda(choose_answer)
    
                response = chain.invoke(query)
                paint_answer(query, response.content)
         
        else:
            st.session_state["query_hist"] = []

           

