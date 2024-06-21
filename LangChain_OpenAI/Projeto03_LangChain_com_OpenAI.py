from langchain_openai import ChatOpenAI

# Variaiveis da viagem
numero_de_dias = 7
numero_de_criancas = 2
atividade = "Praia"
pais = "Estados Unidos"

# Prompt do roteiro de viagem
prompt = f"Crie um roteiro de viagem de {numero_de_dias} dias, para uma familia com {numero_de_criancas} crianças que gostam de {atividade}, o destino será {pais}"

# Instanciando o ChatOpenAI, usando o langChain a construção do chat se torna mais rapida e direta.
llm = ChatOpenAI(model="gpt-4",
                temperature=0.5,
                api_key="Cole aqui sua chave privada da OpenAI")

# Retorno do roteiro de viagem
resposta = llm.invoke(prompt)
print(resposta.content)

