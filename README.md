# SiteGPT

<img src="https://github.com/user-attachments/assets/e7941c80-b675-4998-908c-531956a0bdba" width="400" />

<br>
<br>

## Description

A Chatbot which is possible to crawl the website with given sitemap url and find the answer for the user's questions based on the documents from the website.

<br>

## The Structure of Chain

> - **retriever(Vectorstore: FAISS) + Map Re-rank prompt + LLM(gpt-4.1-nano-2025-04-14)** <br>
> - **Map Re-rank Chain** : answers_chain + choose_chain

<br>

## Map Re-rank Prompt

```
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
                """)

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
```

<br>
<br>
<br>
