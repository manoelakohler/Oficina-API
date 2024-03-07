import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import requests
from io import StringIO
import base64

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Arraste e solte ou ', html.A('selecione um arquivo CSV')]),
        multiple=False
    ),
    html.Button('Fazer Previsões', id='btn-predict'),
    html.Div(id='output-predictions')
])


def call_api(file_contents):
    url = 'https://bimaster-c4c1bb7dcb68.herokuapp.com/predict'
    files = {'file': ('filename.csv', file_contents)}
    response = requests.post(url, files=files)
    return response.json()


@app.callback(
    Output('output-predictions', 'children'),
    [Input('btn-predict', 'n_clicks')],
    [dash.dependencies.State('upload-data', 'contents')]
)
def update_predictions(n_clicks, contents):
    if n_clicks is None:
        raise PreventUpdate

    if contents is None:
        return 'Nenhum arquivo carregado.'

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    file_contents = StringIO(decoded.decode('utf-8'))

    api_response = call_api(file_contents)

    predictions = api_response.get('Predictions', [])

    if not predictions:
        return 'Nenhuma previsão disponível.'

    return html.Div([
        html.Div('Previsões:'),
        html.Pre(', '.join(predictions))
    ])


if __name__ == '__main__':
    app.run_server(debug=True)
