import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import requests
import base64
import pandas as pd
import plotly.express as px
from io import BytesIO

app_dash = dash.Dash(__name__)

app_dash.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Arraste e solte ou ',
            html.A('selecione um arquivo CSV')
        ]),
        multiple=False
    ),
    html.Div(id='output-data-upload'),
])


def parse_contents(contents):
    # Decodifica o conteúdo base64 diretamente em um DataFrame
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(BytesIO(decoded), encoding='utf-8')
    return df


@app_dash.callback(Output('output-data-upload', 'children'),
                   Input('upload-data', 'contents'))
def update_output(contents):
    if contents is None:
        return

    df = parse_contents(contents)

    # Chame a API FastAPI para obter previsões
    response = requests.post("http://localhost:5000/predict", files={"file": contents})
    response = requests.post("http://localhost:5000/#/default/predict_predict_post", files={"file": contents})

    predictions = response.json()["Predictions"]

    # Crie a tabela com os resultados
    table = html.Table(
        # Header
        [html.Tr([html.Th(col) for col in df.columns] + [html.Th("Prediction")])] +
        # Body
        [html.Tr([html.Td(df.iloc[i, col]) for col in range(len(df.columns))] + [html.Td(predictions[i])]) for i in
         range(len(df))]
    )

    # Crie o gráfico de pizza
    pie_chart = px.pie(names=["M", "B", "S"],
                       values=[predictions.count("M"), predictions.count("B"), predictions.count("S")],
                       title='Prediction Distribution')

    return [
        html.H5("Resultados da previsão"),
        table,
        dcc.Graph(figure=pie_chart)
    ]


if __name__ == '__main__':
    app_dash.run_server(debug=True)
