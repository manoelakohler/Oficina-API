import io
import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File


app = FastAPI(docs_url='/', title="Oficina DM BIMaster - PUC-Rio")

# Carregar modelo
pipeline = joblib.load('breast_pipeline.pkl')


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """ Endpoint para previsão de malignidade em tumores de mama.

    Este endpoint recebe um arquivo CSV contendo dados de tumores como entrada e retorna previsões
    para malignidade usando um modelo de aprendizado de máquina pré-carregado.

    :param file: Um arquivo CSV contendo dados de tumores.


    :type file: UploadFile


    :return: Uma resposta JSON contendo previsões para malignidade:
            - M = Maligno
            - B = Benigno
            - S - Suspeito


    :rtype: dict

    Exemplo:
    {
        "Predictions": [B, M, B , S]
    }

    """

    # Lê o arquivo CSV enviado pelo usuário
    content = await file.read()
    # Cria um "arquivo" de texto em memória a partir do conteúdo decodificado do arquivo CSV
    file = io.StringIO(content.decode('utf-8'))
    # Carrega os dados em um dataframe
    data = pd.read_csv(file, index_col=0)

    # Realiza a previsão usando o modelo carregado
    predictions = pipeline.predict(data)

    return {"Predictions": predictions.tolist()}
