import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, html, dcc, Input, Output
import dash_daq as daq

def make_dashboard(df: pd.DataFrame, target_label: str) -> None:
    filter_controls = []

    # Boolean columns
    bool_cols = sorted(df.select_dtypes(include=[bool]).columns.tolist())
    row_size = 5

    for i in range(0, len(bool_cols), row_size):
        row_cols = bool_cols[i:i + row_size]
        row_controls = []

        for col in row_cols:
            default_val = True if True in df[col].unique() else False
            row_controls.append(html.Div([
                html.Label(col, style={"margin-right": "8px", "align-self": "center"}),
                daq.BooleanSwitch(
                    id=f"filter-{col}",
                    on=default_val,
                    labelPosition="bottom"
                )
            ], style={"display": "flex", "align-items": "center", "flex": "1", "margin-right": "10px"}))
        filter_controls.append(html.Div(row_controls, style={"display": "flex", "margin-bottom": "10px"}))

    float_cols = df.select_dtypes(include=[float]).columns.tolist()
    float_cols.remove(target_label)
    for i in range(0, len(float_cols), row_size):
            row_cols = float_cols[i:i + row_size]
            row_controls = []
            for col in row_cols:
                unique_vals = sorted(df[col].unique())
                row_controls.append(html.Div([
                    html.Label(f"{col}"),
                    dcc.RangeSlider(
                        id=f"filter-{col}",
                        min=unique_vals[0],
                        max=unique_vals[-1],
                        step=None,
                        marks={v: str(v) for v in unique_vals},
                        value=[unique_vals[0], unique_vals[-1]],
                        tooltip={"placement": "top", "always_visible": False}
                    )
                ], style={"flex": "1", "margin-right": "10px"}))
            filter_controls.append(html.Div(row_controls, style={"display": "flex", "margin-bottom": "15px"}))

    app = Dash(__name__)

    app.layout = html.Div([
        html.H2("Dynamic Distribution Explorer"),
        html.Div(filter_controls),
        dcc.Graph(id="dist-plot")
    ])

    nbins = 100
    min_reward = df[target_label].min()
    max_reward = df[target_label].max()

    # Build callback inputs dynamically
    bool_inputs = [Input(f"filter-{col}", "on") for col in bool_cols]
    float_inputs = [Input(f"filter-{col}", "value") for col in float_cols]
    all_inputs = bool_inputs + float_inputs

    @app.callback(
        Output("dist-plot", "figure"),
        all_inputs
    )
    def update_distribution(*filter_values):
        filtered = df.copy()
        bool_vals = filter_values[:len(bool_cols)]
        float_vals = filter_values[len(bool_cols):]

        # Apply boolean filters
        for col, val in zip(bool_cols, bool_vals):
            if not val:
                filtered = filtered[~filtered[col]]

        # Apply float range filters
        for col, val_range in zip(float_cols, float_vals):
            filtered = filtered[(filtered[col] >= val_range[0]) & (filtered[col] <= val_range[1])]

        # Build histogram
        fig = px.histogram(
            filtered,
            range_x=[min_reward, max_reward],
            x=target_label,
            nbins=nbins,
            barmode="overlay",
            opacity=0.7
        )
        return fig

    app.run(debug=True)
