
#%%

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go

app = dash.Dash(__name__)

history_data = pd.read_csv('solar_history.csv')
projected_data = pd.read_csv('energy_projection.csv')

app.layout = html.Div(
    style={'backgroundColor': 'yellow'},
    children=[
        html.H1('Projected Energy Supply', style={'textAlign': 'center'}),
        dcc.Graph(
            id='energy-supply-graph',
            figure={
                'data': [
                    go.Scatter(
                        x=history_data['Date'],
                        y=history_data['Supply'],
                        name='Solar Supply History',
                        mode='lines',
                        line={'color': 'darkgreen'}
                    ),
                    go.Scatter(
                        x=projected_data['Date'],
                        y=projected_data['Supply'],
                        name='Projected Energy Supply',
                        mode='lines',
                        line={'color': 'limegreen'}
                    )
                ],
                'layout': go.Layout(
                    xaxis={'title': 'Date'},
                    yaxis={'title': 'Energy Supply'},
                    plot_bgcolor='yellow',
                    paper_bgcolor='yellow'
                )
            }
        )
    ]
)

if __name__ == '__main__':
    app.run_server(debug=True)
# %%
