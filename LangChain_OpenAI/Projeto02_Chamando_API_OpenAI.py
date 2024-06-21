import openai

# Configurar a chave da API
openai.api_key = "sk-proj-SJwZITYfQ43n0sw9LUl7T3BlbkFJhEIaRRVdFXdkhBC1P3gj"

# Variaiveis da viagem
numero_de_dias = 7
numero_de_criancas = 2
atividade = "Praia"
pais = "Estados Unidos"

# Prompt do roteiro de viagem
prompt = f"Crie um roteiro de viagem de {numero_de_dias} dias, para uma familia com {numero_de_criancas} crianças que gostam de {atividade}, o destino será {pais}"

# Realizar a chamada à API OpenAI
resposta = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Você é um assistente que está criando um roteiro de viagem para uma família."},
        {"role": "user", "content": prompt}
    ]
)

# Retorno do roteiro de viagem
roteiro_de_viagem = resposta.choices[0].message.content
print(roteiro_de_viagem)
