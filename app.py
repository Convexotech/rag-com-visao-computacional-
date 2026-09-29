import os
import base64
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import AzureOpenAI, APIError
import streamlit as st
import pandas as pd
from pypdf import PdfReader
import sqlite3


banco = sqlite3.connect('banco_imb.db')
cursor = banco.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS img(
               
               id INTEGER,
               dados TEXT,
               nome TEXT
                              
               
               )
''')
banco.commit()
# banco.close()


# Carrega variáveis locais se existirem (para execução local com .env)
load_dotenv(override=True)

# Compatibilidade entre st.secrets e os nomes de variáveis de ambiente solicitados
def get_env_variable(key, default=None):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)

AZURE_ENDPOINT = get_env_variable("ENDPOINT")
AZURE_API_KEY = get_env_variable("API_KEY")
AZURE_API_VERSION = get_env_variable("API_VERSION", "2025-04-01-preview")
DEPLOYMENT_NAME = get_env_variable("GPT5_MODEL")

# Inicialização do Cliente Azure OpenAI
client = None
if AZURE_ENDPOINT and AZURE_API_KEY:
    try:
        client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            api_version=AZURE_API_VERSION
        )
    except Exception:
        pass

@st.cache_data(ttl=3600)
def raspar_site_dados():
    url = "https://gratuitos.netlify.app/"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove elementos indesejados
        for script in soup(["script", "style"]):
            script.extract()
            
        texto_limpo = soup.get_text(separator="\n")
        linhas = [linha.strip() for linha in texto_limpo.splitlines() if linha.strip()]
        return "\n".join(linhas)
    except Exception as e:
        return f"Erro ao acessar o site: {e}"

@st.cache_data
def carregar_dados_locais():
    conteudo_csv = "dados.csv"
    conteudo_pdf = "escola.pdf"
    
    # Leitura de arquivo CSV se existir no diretório
    try:
        if os.path.exists("dados.csv"):
            df = pd.read_csv("dados.csv")
            conteudo_csv = df.to_string(index=False)
    except Exception as e:
        conteudo_csv = f"Erro ao ler CSV: {e}"

    # Leitura de arquivo PDF se existir no diretório
    try:
        if os.path.exists("escola.pdf"):
            reader = PdfReader("escola.pdf")
            texto_pdf_list = []
            for pagina in reader.pages:
                texto_pagina = pagina.extract_text()
                if texto_pagina:
                    texto_pdf_list.append(texto_pagina)
            conteudo_pdf = "\n".join(texto_pdf_list)
    except Exception as e:
        conteudo_pdf = f"Erro ao ler PDF: {e}"

    return conteudo_csv, conteudo_pdf

def validar_configuracao():
    faltando = []
    if not AZURE_ENDPOINT:
        faltando.append("ENDPOINT")
    if not AZURE_API_KEY:
        faltando.append("API_KEY")
    if not DEPLOYMENT_NAME:
        faltando.append("GPT5_MODEL")
    return faltando

def image_to_base64(uploaded_file):
    """Converte o arquivo de imagem carregado para string base64."""
    bytes_data = uploaded_file.getvalue()
    return base64.b64encode(bytes_data).decode("utf-8")

def main():
    st.set_page_config(page_title="Consulta de Unidades e Visão", page_icon="🏫", layout="centered")
    
    st.title("🏫 Consulta de Unidades, Cursos e Visão Computacional")
    st.write("Utilize o chat para tirar dúvidas sobre as unidades ou envie uma imagem para identificar objetos e pessoas.")

    faltando = validar_configuracao()
    if faltando:
        st.error(f"Erro de configuração: As seguintes variáveis não foram definidas no ambiente ou secrets: {', '.join(faltando)}")
        return

    # Executa o scraping e a leitura dos arquivos locais com cache
    dados_site = raspar_site_dados()
    dados_csv, dados_pdf = carregar_dados_locais()

    # Consolida todo o contexto do RAG
    dados_recuperados = f"""
[FONTE 1: DADOS DO CSV]
{dados_csv}

[FONTE 2: DADOS DO PDF]
{dados_pdf}

[FONTE 3: DADOS DO SITE]
{dados_site}
"""

    # --- SEÇÃO 1: RAG / CHAT ---
    with st.form(key="form_pergunta"):
        pergunta = st.text_input("Qual a sua dúvida ou unidade que deseja buscar?", placeholder="Ex: Qual o endereço da unidade mais próxima?")
        botao_enviar = st.form_submit_button("Buscar Unidade / Perguntar")

    if botao_enviar:
        if not pergunta.strip():
            st.warning("Por favor, digite uma pergunta válida.")
        elif not client:
            st.error("Cliente Azure OpenAI não inicializado corretamente.")
        else:
            prompt = f"""Você é um assistente educacional prestativo. Use apenas os dados extraídos das fontes abaixo para responder de forma clara, objetiva e educativa:

Dados disponíveis:
{dados_recuperados}
"""

            try:
                with st.spinner("Consultando informações..."):
                    response = client.chat.completions.create(
                        model=DEPLOYMENT_NAME,
                        messages=[
                            {
                                "role": "system",
                                "content": prompt,
                            },
                            {
                                "role": "user",
                                "content": pergunta
                            }
                        ]
                    )
                    resposta_texto = response.choices[0].message.content
                    
                    st.markdown("### Resposta:")
                    st.success(resposta_texto)

            except APIError as e:
                st.error(f"Erro na API do Azure: {e}")
            except Exception as e:
                st.error(f"Erro inesperado: {e}")

    st.markdown("---")

    # --- SEÇÃO 2: VISÃO COMPUTACIONAL (AZURE FOUNDRY) ---
    st.subheader("🖼️ Identificação de Objetos e Pessoas em Imagens")
    uploaded_file = st.file_uploader("Carregue uma imagem (JPG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # st.image(uploaded_file, caption="Imagem Carregada", use_column_width=True)
        st.image(uploaded_file, caption="Imagem Carregada", use_container_width=True)
        
        if st.button("Identificar Objetos e Pessoas na Imagem"):
            if not client:
                st.error("Cliente Azure OpenAI não inicializado corretamente.")
            else:
                with st.spinner("Analisando imagem com Visão Computacional do Azure..."):
                    try:
                        base64_image = image_to_base64(uploaded_file)
                        
                        response_visao = client.chat.completions.create(
                            model=DEPLOYMENT_NAME,
                            messages=[
                                {
                                    "role": "system",
                                    "content": "Você é um especialista em visão computacional. Sua tarefa é analisar detalhadamente a imagem fornecida, listando e descrevendo todos os objetos e pessoas detectados."
                                },
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": "Identifique e descreva detalhadamente todos os objetos e pessoas presentes nesta imagem."},
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/jpeg;base64,{base64_image}"
                                            }
                                        }
                                    ]
                                }
                            ],
                            max_completion_tokens=1000  # Alterado de max_tokens para max_completion_tokens
                        )
                        
                        resultado_visao = response_visao.choices[0].message.content
                        st.markdown("### Resultado da Análise Visual:")
                        st.success(resultado_visao)

 
                    except APIError as e:
                        st.error(f"Erro na API do Azure: {e}")
                    except Exception as e:
                        st.error(f"Erro inesperado ao processar a imagem: {e}")
                    
                    # inserindo no banco
                    ID  = st.number_input('ID')
                    cursor.execute('INSERT INTO img values(?,?,?)', (ID,base64_image,response_visao))    
                    banco.commit()
                    banco.commit()
                    # banco.close()
  

if __name__ == "__main__":
    main()
