#Importação das bibliotecas
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
import tempfile

# Carregar variáveis de ambiente
load_dotenv()

# Função para carregar documentos e criar base de dados de vetores
def setup_database(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    embeddings = OpenAIEmbeddings()
    data_base = FAISS.from_documents(documents, embeddings)
    return data_base

# Função para realizar busca nos vetores
def retrieve_info(query, data_base):
    similar_response = data_base.similarity_search(query, k=3)
    return [doc.page_content for doc in similar_response]

# Função para configurar a cadeia LLM
def setup_llm_chain():
    llm = ChatOpenAI(temperature=0.5, model="gpt-4o")

    template = """
    Voce é um novo chatbot superinteligente do mercado, e que vai ser possivel
    ajudar as pessoas com os proprios dados dela

    voce irá responder variadas perguntas a partir de documentos texto que essas pessoas irão anexar
    este documentos pode ser um email, um livro, uma simples mensagem etc

    aqui está é uma pergunta recebida
    {message}

    aqui esta a possivel resposta para a pergunta
    {best_practice}
    """ 
    prompt = PromptTemplate(
        input_variables=["message", "best_practice"],
        template=template
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

# Função para gerar resposta
def generate_response(message, chain, data_base):
    best_practice = retrieve_info(message, data_base)
    response = chain.run(message=message, best_practice=best_practice)
    return response

# Configuração do Streamlit
st.title("GEN.AI - Treine com seus dados")
st.write("Pergunte sobre o arquivo texto que você anexou!")

# Campo de upload de arquivo
uploaded_file = st.file_uploader("Faça upload de um documento texto para treinamento", type=["txt"])

if uploaded_file:
    # Salvar o arquivo carregado temporariamente
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_file_path = tmp_file.name
    
    # Carregar e configurar banco de dados e cadeia LLM
    data_base = setup_database(temp_file_path)
    llm_chain = setup_llm_chain()

    # Campo de entrada para a pergunta do usuário
    user_input = st.text_input("Pergunte sobre o arquivo texto anexo:")

    if user_input:
        # Gerar resposta com base na entrada do usuário
        response = generate_response(user_input, llm_chain, data_base)

        # Exibir a resposta
        st.write("O melhor retorno:")
        st.write(response)
