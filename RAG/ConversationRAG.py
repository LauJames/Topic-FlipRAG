from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

class ConversationRAGChain(object):
    def __init__(self, llm, prompt_llm, retriever):
        self.llm = llm  
        self.prompt_llm = prompt_llm
        self.retriever = retriever

        self.contextualize_q_system_prompt = """Given a chat history and the latest user input \
which might reference context in the chat history, extract a standalone question \
which can be understood without the chat history and ignore the command. Do NOT answer the question \
just reformulate it between <<< and >>> if needed and otherwise return it as is."""
        self.prompt_qa = """
        You are a neutral and objective assistant. 
        Answer the question using only the information provided in the retrieved context. 
        Avoid adding personal opinions, assumptions, or any unsupported information. 
        Base your response strictly on the context.
        Keep the answer concise:

        Context: {context}.
        """
        self.prompt_qa = """
Use the following pieces of retrieved context to answer the question. Keep the answer concise:

Context: {context}.
Put the whole answer in [[ ]]

"""


    def run(self, query):
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.prompt_llm, self.retriever, contextualize_q_prompt
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.prompt_qa),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        answer = rag_chain.invoke({"input": query, "chat_history": [""]})

        return answer

    def direct_answer(self, query):
        direct_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant.Keep the answer concise"),
                ("human", "{input}"),
            ]
        )

        prompt_messages = direct_prompt.format_prompt(input=query).to_messages()
        answer = self.llm.invoke(prompt_messages)

        return answer

    
    def mid_output(self, input):
        print(input)
        return input