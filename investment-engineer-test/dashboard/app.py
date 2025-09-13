# dashboard/app.py
import threading
import webbrowser
import requests
import pandas as pd
import numpy as np
import dash
from dash import Dash, html, dcc, dash_table, Input, Output, State, ALL, MATCH
import plotly.express as px
import plotly.graph_objects as go

API_BASE = "http://127.0.0.1:8000"

app = Dash(__name__, title="Portfolio Forecaster")
server = app.server

# Fetch assets
try:
    resp = requests.get(f"{API_BASE}/assets", timeout=10)
    resp.raise_for_status()
    assets_data = resp.json()
    all_assets = assets_data.get("assets", [])
    categories_df = pd.DataFrame(assets_data.get("categories", []))
except Exception as e:
    print(f"Error fetching assets: {e}")
    all_assets = []
    categories_df = pd.DataFrame()

# ----------------- Layout -----------------
app.layout = html.Div(style={
    "font-family": "Arial, sans-serif",
    "margin": "20px",
    "background": "linear-gradient(to right, #f7f9fc, #e3ebf6)",
    "padding": "20px"
}, children=[
    html.H1("Portfolio Forecaster", style={"textAlign": "center", "color": "#1F3B5F"}),

    dcc.ConfirmDialog(id="error-banner"),
    dcc.Store(id='portfolio-state', data=[]),
    dcc.Store(id='weights-store', data={}),

    html.Div(style={"background": "white", "padding": "20px", "border-radius": "10px",
                    "box-shadow": "0 4px 6px rgba(0,0,0,0.1)", "margin-bottom": "30px"}, children=[
        html.H3("Portfolio Allocation", style={"color": "#1F3B5F"}),

        html.Div(style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '10px'}, children=[
            dcc.Dropdown(
                id='asset-dropdown',
                options=[],
                placeholder="Select an asset to add...",
                style={'width': '300px', 'margin-right': '10px'}
            ),
            html.Button('Add Asset', id='add-asset-btn', n_clicks=0,
                        style={"background-color": "#2E86AB", "color": "white", "border": "none", "cursor": "pointer"}),
        ]),

        html.Div(id='allocation-inputs-container'),

        html.Div(id="normalization-message", style={"margin-top": "15px", "color": "#E74C3C", "font-weight": "bold"}),

        html.Br(),
        html.Div([
            html.Label("Starting Portfolio Value:", style={"fontWeight": "bold"}),
            dcc.Input(id="start-value", type="number", value=100000, style={"margin-left": "10px"}),
        ], style={"margin-top": "10px"}),
        html.Br(),
        html.Button("Recalculate Portfolio", id="recalc", n_clicks=0,
                    style={"background-color": "#2E86AB", "color": "white", "padding": "10px 20px",
                           "border": "none", "cursor": "pointer", "border-radius": "5px"}),
    ]),

    html.Div(style={"display": "flex", "justify-content": "space-between", "margin-bottom": "30px"}, children=[
        html.Div(style={"background": "white", "padding": "20px", "border-radius": "10px",
                        "box-shadow": "0 4px 6px rgba(0,0,0,0.1)", "width": "48%"}, children=[
            html.H3(" Allocation by Asset", style={"color": "#1F3B5F"}),
            dcc.Loading(type="circle", children=dcc.Graph(id="pie-chart", style={"height": "400px"}))
        ]),
        html.Div(style={"background": "white", "padding": "20px", "border-radius": "10px",
                        "box-shadow": "0 4px 6px rgba(0,0,0,0.1)", "width": "48%"}, children=[
            html.H3("Allocation by Category", style={"color": "#1F3B5F"}),
            dcc.Loading(type="circle", children=dcc.Graph(id="bar-chart", style={"height": "400px"}))
        ]),
    ]),

    html.Div(style={"background": "white", "padding": "20px", "border-radius": "10px",
                    "box-shadow": "0 4px 6px rgba(0,0,0,0.1)", "margin-bottom": "30px"}, children=[
        html.H3("Projected Portfolio Value", style={"color": "#1F3B5F"}),
        dcc.Loading(type="circle", children=dcc.Graph(id="projection-chart", style={"height": "450px"}))
    ]),

    html.Div(style={"background": "white", "padding": "20px", "border-radius": "10px",
                    "box-shadow": "0 4px 6px rgba(0,0,0,0.1)", "margin-bottom": "30px"}, children=[
        html.H3("Portfolio Metrics", style={"color": "#1F3B5F"}),
        html.P("Click a metric name for an explanation.", style={"font-style": "italic", "color": "#555"}),
        dcc.Loading(type="circle", children=dash_table.DataTable(
            id="metrics-table",
            columns=[
                {"name": "Metric", "id": "metric"},
                {"name": "Value", "id": "value"},
            ],
            style_cell={"textAlign": "center"},
            style_header={"backgroundColor": "#1F3B5F", "color": "white", "fontWeight": "bold"},
            style_data={"backgroundColor": "#F0F4F7"},
            data=[]
        )),
        html.Div(id="metric-explanation-output", style={"margin-top": "20px", "padding": "10px", "border-left": "3px solid #2E86AB"}),
        html.Div(id="sharpe-commentary", style={"margin-top": "15px", "font-style": "italic", "color": "#1F3B5F"})
    ]),
    
    html.Div(style={"background": "white", "padding": "20px", "border-radius": "10px",
                    "box-shadow": "0 4px 6px rgba(0,0,0,0.1)", "margin-bottom": "30px"}, children=[
        html.H3("Asset Metrics", style={"color": "#1F3B5F"}),
        dcc.Loading(type="circle", children=dash_table.DataTable(
            id="asset-metrics-table",
            columns=[
                {"name": "Asset", "id": "asset_name"},
                {"name": "Annualised Return", "id": "annualised_return"},
                {"name": "Annualised Volatility", "id": "annualised_volatility"},
            ],
            style_cell={"textAlign": "center"},
            style_header={"backgroundColor": "#1F3B5F", "color": "white", "fontWeight": "bold"},
            style_data={"backgroundColor": "#F0F4F7"},
            data=[]
        )),
        html.Br(),
        html.H4("Correlation Matrix", style={"color": "#1F3B5F"}),
        dcc.Loading(type="circle", children=dcc.Graph(id="correlation-heatmap", style={"height": "450px"}))
    ]),
])

# ----------------- Callbacks -----------------

# Callback to manage the list of assets in the portfolio
@app.callback(
    Output('portfolio-state', 'data'),
    Output('weights-store', 'data'),
    Input('add-asset-btn', 'n_clicks'),
    Input({'type': 'remove-btn', 'asset_name': ALL}, 'n_clicks'),
    State('asset-dropdown', 'value'),
    State('portfolio-state', 'data'),
    State('weights-store', 'data'),
    prevent_initial_call=True
)
def manage_portfolio_state(add_clicks, remove_clicks, selected_asset, current_state, current_weights, ctx = dash.callback_context):
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    if trigger_id == 'add-asset-btn' and selected_asset and selected_asset not in current_state:
        current_state.append(selected_asset)
        current_weights[selected_asset] = 0
    
    if trigger_id and 'remove-btn' in trigger_id:
        asset_to_remove = eval(trigger_id)['asset_name']
        if asset_to_remove in current_state:
            current_state.remove(asset_to_remove)
            if asset_to_remove in current_weights:
                del current_weights[asset_to_remove]
    
    return current_state, current_weights

# Callback to update the weights store in real-time
@app.callback(
    Output('weights-store', 'data', allow_duplicate=True),
    Input({'type': 'weight-input', 'asset_name': ALL}, 'value'),
    State({'type': 'weight-input', 'asset_name': ALL}, 'id'),
    State('weights-store', 'data'),
    prevent_initial_call=True
)
def update_weights_store(weights_values, weights_ids, weights_data):
    if not weights_values:
        return {}
    
    new_weights = {weights_ids[i]['asset_name']: v for i, v in enumerate(weights_values)}
    
    weights_data.update(new_weights)
    return weights_data

# Callback to generate the dynamic input fields
@app.callback(
    Output('allocation-inputs-container', 'children'),
    Input('portfolio-state', 'data'),
    State('weights-store', 'data')
)
def generate_allocation_ui(portfolio_assets, weights_data):
    children = []
    for asset in portfolio_assets:
        children.append(
            html.Div(style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '10px'}, children=[
                html.Label(f"{asset} Weight:", style={'width': '150px', 'fontWeight': 'bold'}),
                dcc.Input(
                    id={'type': 'weight-input', 'asset_name': asset},
                    type='number',
                    value=weights_data.get(asset, 0),
                    min=0,
                    style={'width': '100px', 'margin-right': '10px'}
                ),
                html.Button('Remove', id={'type': 'remove-btn', 'asset_name': asset}, n_clicks=0,
                            style={"background-color": "#C0392B", "color": "white", "border": "none", "cursor": "pointer"})
            ])
        )
    return children

# Callback to update the dropdown menu
@app.callback(
    Output('asset-dropdown', 'options'),
    Output('asset-dropdown', 'placeholder'),
    Output('add-asset-btn', 'disabled'),
    Input('portfolio-state', 'data')
)
def update_dropdown(portfolio_assets):
    available_assets = [a for a in all_assets if a not in portfolio_assets]
    if not available_assets:
        return [{'label': 'No more assets to add', 'value': ''}], 'No more assets to add', True
    options = [{'label': a, 'value': a} for a in available_assets]
    return options, 'Select an asset to add...', False

# Main callback for recalculating all graphs and metrics
@app.callback(
    Output("pie-chart", "figure"),
    Output("bar-chart", "figure"),
    Output("projection-chart", "figure"),
    Output("metrics-table", "data"),
    Output("asset-metrics-table", "data"),
    Output("correlation-heatmap", "figure"),
    Output("error-banner", "displayed"),
    Output("error-banner", "message"),
    Output("normalization-message", "children"),
    Output("sharpe-commentary", "children"),
    Input("recalc", "n_clicks"),
    State("weights-store", "data"),
    State("start-value", "value"),
    prevent_initial_call=True
)
def recalc(n, raw_weights, start_val):
    error_message = None
    normalization_message = ""
    sharpe_commentary = ""
    metrics = []

    filtered_weights = {k: v for k, v in raw_weights.items() if v is not None and v > 0}
    total_raw_weight = sum(filtered_weights.values())

    if total_raw_weight > 0:
        weights = {k: v / total_raw_weight for k, v in filtered_weights.items()}
        
        if abs(total_raw_weight - 1.0) > 0.001:
            normalization_message = "Note: Your allocations do not sum to 1. They have been automatically normalized."
    else:
        error_message = "Total portfolio weight is zero. Please allocate a positive weight to at least one asset."
        return go.Figure(), go.Figure(), go.Figure(), [], [], go.Figure(), True, error_message, normalization_message, sharpe_commentary
    
    pie_fig = px.pie(names=list(filtered_weights.keys()), values=list(filtered_weights.values()),
                     color_discrete_sequence=px.colors.qualitative.Set2)

    bar_fig = go.Figure()
    if not categories_df.empty:
        selected_assets_df = pd.DataFrame({"asset_name": list(filtered_weights.keys()), "weight": list(filtered_weights.values())})
        merged = selected_assets_df.merge(categories_df, on="asset_name", how="left")
        bar_grouped = merged.groupby("category", dropna=False)["weight"].sum().reset_index()
        bar_fig = px.bar(bar_grouped, x="category", y="weight", text="weight", color="category",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
    else:
        bar_fig = go.Figure().update_layout(
            title="Bar Chart Unavailable: No Category Data from API",
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                go.layout.Annotation(
                    text="Your API is not providing category data. Please check the `/assets` endpoint.",
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(size=14)
                )
            ]
        )

    fig_proj = go.Figure()
    try:
        proj_resp = requests.post(f"{API_BASE}/projection/percentiles",
                                 json={"weights": weights, "start_value": start_val, "normalise": True},
                                 timeout=10)
        proj_resp.raise_for_status()
        proj = proj_resp.json()
        if "percentiles" in proj and "years" in proj:
            years = proj["years"]
            pct = proj["percentiles"]
            if all(k in pct for k in ["0.05", "0.95", "0.25", "0.75", "0.5"]):
                fig_proj.add_trace(go.Scatter(x=years + years[::-1], y=pct["0.95"] + pct["0.05"][::-1],
                                              fill='toself', fillcolor='rgba(135,206,250,0.2)',
                                              line=dict(color='rgba(255,255,255,0)'), name='5-95th percentile'))
                fig_proj.add_trace(go.Scatter(x=years + years[::-1], y=pct["0.75"] + pct["0.25"][::-1],
                                              fill='toself', fillcolor='rgba(0,128,0,0.2)',
                                              line=dict(color='rgba(255,255,255,0)'), name='25-75th percentile'))
                fig_proj.add_trace(go.Scatter(x=years, y=pct["0.5"], mode="lines+markers",
                                              name="50th percentile", line=dict(color="#1F3B5F", width=3)))
    except Exception as e:
        error_message = f"Projection API error: {e}"

    endpoints = {
        "Annualised Volatility": "/forecast_statistic/annualised_volatility",
        "Downside Deviation": "/forecast_statistic/downside_deviation",
        "5% Value at Risk (VaR)": "/forecast_statistic/value_at_risk",
        "5% Conditional VaR (CVaR)": "/forecast_statistic/conditional_value_at_risk",
        "Maximum Drawdown": "/forecast_statistic/maximum_drawdown",
        "Sharpe Ratio": "/forecast_statistic/sharpe_ratio",
    }
    sharpe_ratio = None
    
    for name, url in endpoints.items():
        try:
            payload = {"weights": weights, "normalise": True}
            if "Sharpe" in name: 
                payload["risk_free"] = 0.02
            if "VaR" in name or "CVaR" in name: 
                payload["alpha"] = 0.05
            if "Downside" in name: 
                payload["mar"] = 0.0
            
            resp = requests.post(f"{API_BASE}{url}", json=payload, timeout=10)
            resp.raise_for_status()
            val = resp.json()
            
            if val:
                value = round(list(val.values())[-1], 4)
                metrics.append({"metric": name, "value": value})
                if name == "Sharpe Ratio":
                    sharpe_ratio = value
        except Exception as e:
            print(f"Metrics API error for {name}: {e}")
            error_message = "Error fetching some metrics. Please check the backend API."
            
    if sharpe_ratio is not None:
        if sharpe_ratio < 0.5:
            sharpe_commentary = "Your Sharpe Ratio is low. This suggests a low return for the amount of risk taken. Consider adjusting your asset allocation to potentially improve performance."
        elif sharpe_ratio < 1.0:
            sharpe_commentary = "Your Sharpe Ratio is in a moderate range. This indicates a solid but not exceptional risk-adjusted return."
        else:
            sharpe_commentary = "Your Sharpe Ratio is high! This indicates excellent risk-adjusted returns. Great job!"
    else:
        sharpe_commentary = ""

    asset_metrics, corr_fig = [], go.Figure()
    try:
        asset_resp = requests.get(f"{API_BASE}/assets/metrics", timeout=10)
        asset_resp.raise_for_status()
        asset_data = asset_resp.json()
        asset_metrics = asset_data.get("metrics", [])

        corr_resp = requests.get(f"{API_BASE}/assets/correlation", timeout=10)
        corr_resp.raise_for_status()
        corr_data = corr_resp.json()
        corr_dict = corr_data.get("correlation", {})
        if corr_dict:
            corr_df = pd.DataFrame(corr_dict)
            
            portfolio_assets = list(filtered_weights.keys())
            
            if all(asset in corr_df.columns for asset in portfolio_assets):
                corr_df = corr_df.loc[portfolio_assets, portfolio_assets]
            
            if not corr_df.empty:
                corr_fig = px.imshow(corr_df, text_auto=True, aspect="auto",
                                     color_continuous_scale=px.colors.sequential.Viridis,
                                     title="Asset Return Correlation")
    except Exception as e:
        error_message = f"Asset/correlation API error: {e}"

    return pie_fig, bar_fig, fig_proj, metrics, asset_metrics, corr_fig, bool(error_message), error_message or "", normalization_message, sharpe_commentary

# New callback to handle metric explanation on table click
@app.callback(
    Output("metric-explanation-output", "children"),
    Input("metrics-table", "active_cell"),
    State("metrics-table", "data"),
)
def display_metric_explanation(active_cell, table_data):
    if not active_cell or not table_data:
        return ""
    
    row_index = active_cell['row']
    
    if row_index >= len(table_data):
        return ""

    metric_name = table_data[row_index]["metric"]
    metric_value = table_data[row_index]["value"]

    explanations = {
        "Annualised Volatility": "Measures the fluctuation of portfolio returns. A higher value means the portfolio's value is more likely to swing up or down. This is a measure of risk.",
        "Downside Deviation": "Similar to volatility, but only measures the bad swings (returns below zero). A lower number is better.",
        "5% Value at Risk (VaR)": "Indicates the maximum you can expect to lose on the portfolio over a given period, with 95% confidence.",
        "5% Conditional VaR (CVaR)": "Measures the expected loss in the worst 5% of all outcomes. It's a more conservative measure of risk than VaR.",
        "Maximum Drawdown": "The largest peak-to-trough decline over a specific period. It shows the biggest loss a portfolio has endured from a peak to a subsequent trough.",
        "Sharpe Ratio": "Measures a portfolio's return for each unit of risk taken. A higher number is better, as it indicates a better return for the same amount of risk."
    }

    commentary = ""
    # Check if the metric value is a number before using it
    if isinstance(metric_value, (int, float)):
        if metric_name == "Annualised Volatility":
            if metric_value > 0.15:
                commentary = "This volatility is relatively high for a typical diversified portfolio."
            elif metric_value < 0.05:
                commentary = "This volatility is relatively low, suggesting a less risky portfolio."
            else:
                commentary = "This volatility is in a moderate range."
        elif metric_name == "Downside Deviation":
            if metric_value < 0.05:
                commentary = "This is a low downside deviation, indicating good performance relative to negative returns."
            else:
                commentary = "This downside deviation is high, suggesting significant losses in negative periods."
        elif metric_name == "5% Value at Risk (VaR)":
            if metric_value > 0.1:
                commentary = "Your VaR is high, indicating a higher potential for a significant loss over the period."
            else:
                commentary = "Your VaR is in a low-to-moderate range, indicating a lower potential for significant loss."
        elif metric_name == "5% Conditional VaR (CVaR)":
            if metric_value > 0.15:
                commentary = "Your CVaR is high, indicating that expected losses in the worst-case scenarios are significant."
            else:
                commentary = "Your CVaR is in a low-to-moderate range, indicating smaller expected losses in the worst-case scenarios."
        elif metric_name == "Maximum Drawdown":
            if metric_value < -0.2:
                commentary = "This is a significant maximum drawdown, indicating a large historical loss."
            elif metric_value >= -0.1:
                commentary = "This maximum drawdown is relatively low."
            else:
                commentary = "This drawdown is in a moderate range."

    explanation_text = explanations.get(metric_name, "No explanation available.")
    
    return html.Div([
        html.P([
            html.Span(f"{metric_name}: ", style={"font-weight": "bold"}), 
            explanation_text
        ]),
        html.P(commentary, style={"font-style": "italic"}) if commentary else ""
    ])


if __name__ == "__main__":
    port = 8050
    url = f"http://127.0.0.1:{port}"
    threading.Timer(1.25, lambda: webbrowser.open(url)).start()
    app.run(debug=True, port=port)