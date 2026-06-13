import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import dash
from dash import Dash, html, dcc, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import re

# --- DEPLOYMENT PATH CONFIGURATION ---
# All paths are now relative to this file's location (works on any machine / server)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOG_PATH  = os.path.join(BASE_DIR, "resource", "food_catalog_v5.csv")
MODEL_WEIGHTS = os.path.join(BASE_DIR, "resource", "abns_surrogate_weights.pth")
IMAGE_DIR     = os.path.join(BASE_DIR, "images")

# --- ARCHITECTURE (ResNet — identical to trainer) ---
class TabularResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super(TabularResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim, dim), nn.LayerNorm(dim)
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))

class BioPerceiverFusionNet(nn.Module):
    def __init__(self, input_bio_dim, input_clin_dim, num_foods):
        super(BioPerceiverFusionNet, self).__init__()
        self.bio_stem  = nn.Sequential(nn.Linear(input_bio_dim, 128), nn.LayerNorm(128), nn.GELU())
        self.bio_res   = TabularResBlock(128, dropout=0.1)
        self.bio_out   = nn.Linear(128, 64)
        self.clin_stem = nn.Sequential(nn.Linear(input_clin_dim, 64), nn.LayerNorm(64), nn.GELU())
        self.clin_res  = TabularResBlock(64, dropout=0.1)
        self.clin_out  = nn.Linear(64, 32)
        self.fusion_stem = nn.Sequential(nn.Linear(64 + 32, 256), nn.LayerNorm(256), nn.GELU())
        self.fusion_res  = TabularResBlock(256, dropout=0.1)
        self.projection  = nn.Sequential(nn.Linear(256, num_foods), nn.Sigmoid())

    def forward(self, x_bio, x_clin):
        h_bio  = self.bio_out(self.bio_res(self.bio_stem(x_bio)))
        h_clin = self.clin_out(self.clin_res(self.clin_stem(x_clin)))
        return self.projection(self.fusion_res(self.fusion_stem(torch.cat((h_bio, h_clin), dim=1))))

# --- SYSTEM INITIALIZATION ---
print("🚀 Initializing Bio-Perceiver OS...")
if not os.path.exists(CATALOG_PATH):
    raise FileNotFoundError(f"Missing: {CATALOG_PATH}")

foods_df         = pd.read_csv(CATALOG_PATH)
device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model            = BioPerceiverFusionNet(4, 5, len(foods_df)).to(device)

if os.path.exists(MODEL_WEIGHTS):
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device, weights_only=True))
    print(f"✅ Weights loaded from {MODEL_WEIGHTS}")
else:
    print(f"⚠️  Weights not found at {MODEL_WEIGHTS} — predictions will be random.")

model.eval()
unique_regions   = ["Any"] + sorted(foods_df['region'].dropna().unique().tolist())
total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ Model ready | {total_parameters:,} params | {len(foods_df)} foods | device: {device}")

# --- UX DESIGN TOKENS ---
FONT_FAMILY  = "'Inter', sans-serif"
TEXT_PRIMARY = '#0f172a'
TEXT_MUTED   = '#64748b'
ACCENT       = '#0284c7'
SUCCESS      = '#059669'

LUX_PANEL = {
    'background': 'rgba(255,255,255,0.75)', 'backdropFilter': 'blur(24px)',
    'WebkitBackdropFilter': 'blur(24px)',
    'border': '1px solid rgba(255,255,255,0.8)', 'borderRadius': '16px',
    'boxShadow': '0 10px 40px -10px rgba(0,0,0,0.05), inset 0 0 0 1px rgba(255,255,255,0.6)',
    'padding': '20px', 'display': 'flex', 'flexDirection': 'column'
}
SECTION_HEADER = {
    'fontFamily': FONT_FAMILY, 'fontWeight': '800', 'color': TEXT_PRIMARY,
    'fontSize': '14px', 'marginBottom': '16px',
    'borderBottom': '1px solid rgba(0,0,0,0.04)', 'paddingBottom': '10px',
    'letterSpacing': '0.05em'
}
LABEL_STYLE = {
    'fontFamily': FONT_FAMILY, 'fontSize': '10px', 'fontWeight': '700',
    'color': TEXT_MUTED, 'textTransform': 'uppercase', 'marginBottom': '4px',
    'letterSpacing': '0.02em'
}
INPUT_STYLE = {
    'background': 'rgba(255,255,255,0.9)', 'border': '1px solid rgba(0,0,0,0.06)',
    'color': TEXT_PRIMARY, 'borderRadius': '8px', 'padding': '6px 12px',
    'fontFamily': FONT_FAMILY, 'fontSize': '13px', 'width': '100%',
    'marginBottom': '10px', 'outline': 'none', 'transition': 'all 0.2s',
    'boxShadow': 'inset 0 2px 4px rgba(0,0,0,0.01)'
}

# --- DASH APP ---
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"
    ],
    assets_folder=IMAGE_DIR,
    assets_url_path='/images',
    title="Bio-Perceiver OS"
)

# Required for Render/Railway deployment
server = app.server  # expose Flask server

# --- CLIENTSIDE CAROUSEL ---
app.clientside_callback(
    """
    function(n_interval, left_clicks, right_clicks) {
        const ctx = dash_clientside.callback_context;
        if (!ctx.triggered.length) return "";
        const trigger = ctx.triggered[0].prop_id;
        if (trigger.includes('auto-stepper')) {
            document.querySelectorAll('.carousel-track-lux').forEach(track => {
                let n = track.children.length;
                let i = parseInt(track.getAttribute('data-index') || '0');
                i = (i + 1) % n;
                track.setAttribute('data-index', i);
                track.style.transform = `translateX(-${i * 100}%)`;
            });
        } else {
            try {
                const tid = JSON.parse(trigger.split('.')[0]);
                const win = document.getElementById('carousel-window-' + tid.index);
                if (!win) return "";
                const track = win.querySelector('.carousel-track-lux');
                if (!track) return "";
                let n = track.children.length;
                let i = parseInt(track.getAttribute('data-index') || '0');
                i = tid.type === 'btn-left' ? (i - 1 + n) % n : (i + 1) % n;
                track.setAttribute('data-index', i);
                track.style.transform = `translateX(-${i * 100}%)`;
            } catch(e) { console.error("Carousel error:", e); }
        }
        return "";
    }
    """,
    Output('clientside-dummy', 'children'),
    Input('auto-stepper', 'n_intervals'),
    Input({'type': 'btn-left',  'index': ALL}, 'n_clicks'),
    Input({'type': 'btn-right', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True
)

# --- LAYOUT ---
app.layout = html.Div(
    style={
        'minHeight': '100vh',
        'background': 'radial-gradient(circle at 5% 5%, rgba(253,246,211,0.85) 0%, rgba(248,250,252,0.9) 45%, #f8fafc 100%)',
        'padding': '24px', 'fontFamily': FONT_FAMILY
    },
    children=[
        dcc.Interval(id='auto-stepper', interval=5000, n_intervals=0),
        html.Div(id='clientside-dummy', style={'display': 'none'}),

        # Diagnostics Modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle(
                "🔬 TWO-STAGE INFERENCE DIAGNOSTICS",
                style={'fontWeight': '800', 'color': TEXT_PRIMARY, 'fontSize': '16px', 'letterSpacing': '0.05em'}
            )),
            dbc.ModalBody(id="modal-body-content", style={'fontFamily': FONT_FAMILY, 'color': TEXT_MUTED}),
            dbc.ModalFooter(dbc.Button(
                "Acknowledge", id="close-modal", className="ms-auto", n_clicks=0,
                style={'background': TEXT_PRIMARY, 'border': 'none', 'fontWeight': '700'}
            ))
        ], id="diagnostics-modal", is_open=False, centered=True, style={'backdropFilter': 'blur(10px)'}),

        # Telemetry Offcanvas
        dbc.Offcanvas(
            html.Div(id='telemetry-console', style={
                'fontFamily': 'monospace', 'color': '#10b981',
                'fontSize': '13px', 'whiteSpace': 'pre-wrap', 'lineHeight': '1.6'
            }),
            id="telemetry-offcanvas", title="ENGINE TELEMETRY & DIAGNOSTICS",
            is_open=False, placement="end",
            style={'backgroundColor': '#020617', 'color': '#f8fafc', 'borderLeft': '1px solid #334155'}
        ),

        dbc.Container([
            dbc.Row(dbc.Col(html.Div([
                html.H1("BIO-PERCEIVER OS", style={
                    'textAlign': 'center', 'fontWeight': '800', 'fontSize': '32px',
                    'letterSpacing': '0.1em', 'color': TEXT_PRIMARY, 'margin': '0 0 4px 0'
                }),
                html.P("Retrieval-Augmented Neural Ranking Architecture", style={
                    'textAlign': 'center', 'color': ACCENT, 'fontSize': '14px',
                    'fontWeight': '600', 'margin': '0 0 24px 0'
                })
            ]), width=12)),

            dbc.Row(className="align-items-start g-4", children=[

                # LEFT PANEL
                dbc.Col(html.Div(
                    style={'position': 'sticky', 'top': '24px', 'display': 'flex',
                           'flexDirection': 'column', 'gap': '16px', 'height': 'calc(100vh - 48px)'},
                    children=[
                        html.Div(
                            style={**LUX_PANEL, 'flex': '0 0 auto', 'position': 'relative',
                                   'zIndex': '50', 'overflow': 'visible'},
                            children=[
                                html.Div("CLINICAL PARAMETERS", style=SECTION_HEADER),
                                dbc.Row([
                                    dbc.Col([
                                        html.Div("Biological Variables", style={'color': ACCENT, 'fontSize': '12px', 'fontWeight': '700', 'marginBottom': '8px'}),
                                        html.Div("Firmicutes Ratio",     style=LABEL_STYLE),
                                        dcc.Input(id='in-firm', type='number', value=0.5,  step=0.01, min=0, max=1, style=INPUT_STYLE),
                                        html.Div("Bacteroidetes Ratio",  style=LABEL_STYLE),
                                        dcc.Input(id='in-bact', type='number', value=0.4,  step=0.01, min=0, max=1, style=INPUT_STYLE),
                                        html.Div("Proteobacteria Load",  style=LABEL_STYLE),
                                        dcc.Input(id='in-prot', type='number', value=0.1,  step=0.01, min=0, max=1, style=INPUT_STYLE),
                                        html.Div("Actinobacteria Ratio", style=LABEL_STYLE),
                                        dcc.Input(id='in-acti', type='number', value=0.6,  step=0.01, min=0, max=1, style=INPUT_STYLE),
                                    ], width=6),
                                    dbc.Col([
                                        html.Div("Metabolic Metadata", style={'color': ACCENT, 'fontSize': '12px', 'fontWeight': '700', 'marginBottom': '8px'}),
                                        html.Div("Age (Years)",          style=LABEL_STYLE),
                                        dcc.Input(id='in-age',  type='number', value=30,   step=1,   style=INPUT_STYLE),
                                        html.Div("BMI (kg/m²)",          style=LABEL_STYLE),
                                        dcc.Input(id='in-bmi',  type='number', value=24.5, step=0.1, style=INPUT_STYLE),
                                        html.Div("TDEE Limit",           style=LABEL_STYLE),
                                        dcc.Input(id='in-tdee', type='number', value=2000, step=50,  style=INPUT_STYLE),
                                        html.Div("IBS Severity (0-1)",   style=LABEL_STYLE),
                                        dcc.Input(id='in-ibs',  type='number', value=0.2,  step=0.1, min=0, max=1, style=INPUT_STYLE),
                                    ], width=6)
                                ], className="mb-2"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Div("Dietary Anchor", style=LABEL_STYLE),
                                        dcc.Dropdown(
                                            id='in-diet',
                                            options=[{'label': 'Vegetarian', 'value': '0'},
                                                     {'label': 'Non-Vegetarian', 'value': '1'}],
                                            value='0', clearable=False,
                                            style={'fontSize': '12px', 'marginBottom': '8px'}
                                        )
                                    ], width=6, style={'zIndex': 1000}),
                                    dbc.Col([
                                        html.Div("Cultural Anchor", style=LABEL_STYLE),
                                        dcc.Dropdown(
                                            id='in-region',
                                            options=[{'label': r, 'value': r} for r in unique_regions],
                                            value='Any', clearable=False,
                                            style={'fontSize': '12px', 'marginBottom': '8px'}
                                        )
                                    ], width=6, style={'zIndex': 1000})
                                ]),
                                html.Div(
                                    style={'display': 'flex', 'gap': '12px', 'marginTop': 'auto'},
                                    children=[
                                        html.Button(
                                            "COMPILE DIAGNOSTIC", id='run-btn',
                                            style={'flex': '1', 'background': TEXT_PRIMARY, 'color': '#ffffff',
                                                   'border': 'none', 'borderRadius': '8px', 'padding': '12px',
                                                   'fontWeight': '700', 'letterSpacing': '0.05em',
                                                   'cursor': 'pointer', 'transition': 'background 0.3s',
                                                   'boxShadow': '0 4px 14px rgba(15,23,42,0.15)'}
                                        ),
                                        html.Button(
                                            "⚙️", id='open-telemetry-btn', title="Engine Diagnostics",
                                            style={'background': '#e2e8f0', 'color': TEXT_PRIMARY,
                                                   'border': 'none', 'borderRadius': '8px',
                                                   'padding': '12px 16px', 'cursor': 'pointer',
                                                   'boxShadow': '0 4px 14px rgba(15,23,42,0.05)',
                                                   'transition': 'all 0.2s'}
                                        )
                                    ]
                                )
                            ]
                        ),

                        html.Div(
                            style={**LUX_PANEL, 'flex': '1 1 auto', 'minHeight': '0',
                                   'position': 'relative', 'zIndex': '1'},
                            children=[
                                html.Div([
                                    html.Span("EXPLAINABLE AI METRICS", style={
                                        'fontFamily': FONT_FAMILY, 'fontWeight': '800',
                                        'color': TEXT_PRIMARY, 'fontSize': '14px', 'letterSpacing': '0.05em'
                                    }),
                                    html.Span(id='health-score-display', style={
                                        'float': 'right', 'color': ACCENT,
                                        'fontWeight': '800', 'fontSize': '16px'
                                    })
                                ], style={'borderBottom': '1px solid rgba(0,0,0,0.04)',
                                          'paddingBottom': '10px', 'marginBottom': '8px'}),
                                dcc.Graph(
                                    id='xai-plot',
                                    style={'height': '100%', 'width': '100%', 'background': 'transparent'},
                                    config={'displayModeBar': False}
                                )
                            ]
                        )
                    ]
                ), width=4),

                # RIGHT PANEL
                dbc.Col(html.Div(
                    style={**LUX_PANEL, 'minHeight': 'calc(100vh - 48px)', 'padding': '0'},
                    children=[
                        html.Div([
                            html.Span("CLINICAL PRESCRIPTION TRAY", style={
                                'fontFamily': FONT_FAMILY, 'fontWeight': '800',
                                'color': TEXT_PRIMARY, 'fontSize': '16px', 'letterSpacing': '0.05em'
                            }),
                            html.Span(id='tray-capacity-display', style={
                                'float': 'right', 'color': SUCCESS,
                                'fontWeight': '800', 'fontSize': '16px'
                            })
                        ], style={
                            'padding': '20px 24px 12px 24px',
                            'borderBottom': '1px solid rgba(0,0,0,0.06)',
                            'background': 'transparent', 'position': 'sticky',
                            'top': '0', 'zIndex': '10', 'backdropFilter': 'blur(10px)'
                        }),
                        html.Div(id='meal-plan-output', style={
                            'display': 'flex', 'flexDirection': 'column',
                            'padding': '16px', 'gap': '16px'
                        })
                    ]
                ), width=8)
            ])
        ], fluid=True, style={'maxWidth': '1800px'})
    ]
)


# --- CALLBACKS ---

@app.callback(
    Output("telemetry-offcanvas", "is_open"),
    Input("open-telemetry-btn", "n_clicks"),
    State("telemetry-offcanvas", "is_open"),
)
def toggle_telemetry(n1, is_open):
    if n1: return not is_open
    return is_open


@app.callback(
    Output("diagnostics-modal",  "is_open"),
    Output("modal-body-content", "children"),
    Input({'type': 'diag-btn', 'index': ALL}, 'n_clicks'),
    Input("close-modal", "n_clicks"),
    State("diagnostics-modal", "is_open"),
    prevent_initial_call=True
)
def toggle_modal(btn_clicks, close_click, is_open):
    ctx = dash.callback_context
    if not ctx.triggered: return is_open, dash.no_update
    trigger_info = ctx.triggered[0]
    if trigger_info['value'] is None: return is_open, dash.no_update
    trigger_id_str = str(trigger_info['prop_id'])

    if "close-modal" in trigger_id_str: return False, dash.no_update
    if "diag-btn" in trigger_id_str:
        try:
            match = re.search(r'"index":"([^"]+)"', trigger_id_str)
            if not match:
                match = re.search(r"'index': '([^']+)'", trigger_id_str)
            if match:
                food_name, kcal, fodmap, ibs_sev, score, tdee_limit = match.group(1).split('|')
                ibs_logic     = "permissive" if float(ibs_sev) < 0.4 else "restrictive"
                fodmap_impact = ("optimal" if fodmap == "Low"
                                 else "acceptable given metabolic state" if fodmap == "Medium"
                                 else "high risk, requires careful monitoring")
                content = html.Div([
                    html.H4(food_name, style={'color': TEXT_PRIMARY, 'fontWeight': '800'}),
                    html.Hr(),
                    html.P([html.Strong("Stage 1 – Candidate Retrieval: "),
                            "Retrieved via boolean diet/region constraints. Region relaxed if needed."]),
                    html.P([html.Strong("Stage 2 – Neural Ranking: "),
                            f"ResNet scored this item at {float(score)*100:.1f}% affinity from biological and metabolic gradients."]),
                    html.P([html.Strong("Stage 3 – Knapsack Math: "),
                            f"TDEE {tdee_limit} kcal → tray budget {int(float(tdee_limit)*0.4)} kcal. "
                            f"This {kcal} kcal item was packed with highest bio-score under budget."])
                ])
                return True, content
        except Exception:
            pass
    return is_open, dash.no_update


@app.callback(
    Output('xai-plot',              'figure'),
    Output('meal-plan-output',      'children'),
    Output('health-score-display',  'children'),
    Output('tray-capacity-display', 'children'),
    Output('telemetry-console',     'children'),
    Input('run-btn', 'n_clicks'),
    State('in-firm',   'value'), State('in-bact', 'value'),
    State('in-prot',   'value'), State('in-acti', 'value'),
    State('in-age',    'value'), State('in-bmi',  'value'),
    State('in-tdee',   'value'), State('in-diet', 'value'),
    State('in-ibs',    'value'), State('in-region','value'),
)
def execute_pipeline(n_clicks, firm, bact, prot, acti, age, bmi, tdee, diet, ibs, region_pref):
    empty_fig = go.Figure(layout=dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'))
    if n_clicks is None:
        return (empty_fig,
                html.Div("Awaiting execution…", style={'color': TEXT_MUTED, 'textAlign': 'center', 'marginTop': '40px'}),
                "", "0 / 0 KCAL", "> SYSTEM STANDBY.")

    try:
        inputs = [firm, bact, prot, acti, age, bmi, tdee, int(diet or 0), ibs]
        if any(v is None for v in inputs): raise ValueError
    except ValueError:
        return dash.no_update, html.Div("❌ Invalid input.", style={'color': '#ef4444'}), "", "", "> ERROR: BAD TENSOR."

    bio_tensor  = torch.tensor([[firm, bact, prot, acti]], dtype=torch.float32).to(device)
    clin_tensor = torch.tensor(
        [[age / 80.0, bmi / 45.0, float(tdee) / 2500.0, int(diet), ibs]],
        dtype=torch.float32
    ).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        base_preds    = model(bio_tensor, clin_tensor).cpu().numpy().flatten()
        mean_affinity = np.mean(base_preds)
    latency_ms = (time.perf_counter() - t0) * 1000

    # XAI saliency (zero-out ablation)
    sal = []
    with torch.no_grad():
        for i in range(4):
            m = bio_tensor.clone(); m[0, i] = 0.0
            sal.append(np.mean(np.abs(model(m, clin_tensor).cpu().numpy().flatten() - base_preds)))
        for i in range(5):
            m = clin_tensor.clone(); m[0, i] = 0.0
            sal.append(np.mean(np.abs(model(bio_tensor, m).cpu().numpy().flatten() - base_preds)))
    s = sum(sal)
    sal_n = [x / s * 100 if s > 0 else 0 for x in sal]

    xai_fig = go.Figure(go.Bar(
        x=sal_n,
        y=["Firmicutes","Bacteroidetes","Proteobacteria","Actinobacteria",
           "Age","BMI","TDEE","Diet","IBS"],
        orientation='h', marker=dict(color=ACCENT),
        text=[f"{x:.1f}%" for x in sal_n], textposition='outside',
        textfont=dict(color=TEXT_PRIMARY, family=FONT_FAMILY, size=11)
    ))
    xai_fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_family=FONT_FAMILY, font_color=TEXT_MUTED,
        margin=dict(l=0, r=40, t=10, b=0),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, autorange="reversed",
                   tickfont=dict(size=11, color=TEXT_MUTED, weight='bold'))
    )

    # Retrieval + Knapsack
    results = foods_df.copy()
    results['raw_score'] = base_preds
    if int(diet) == 0:
        results = results[results['diet_code'] == 0]
    results['final_score'] = results['raw_score']

    meal_kcal_limit  = float(tdee) * 0.40
    current_kcal     = 0
    priority_slots   = ["Appetizers","Breads_Rice","Dals_Legumes","Main_Course","Desserts","Beverages"]
    meal_ui_elements = []

    for slot_idx, category in enumerate(priority_slots):
        base_cat = results[results['category'] == category]
        cat_items = base_cat if region_pref == "Any" else base_cat[base_cat['region'] == region_pref]
        region_relaxed = False
        if cat_items.empty and region_pref != "Any":
            cat_items = base_cat
            region_relaxed = True

        if int(diet) == 1 and category in ['Main_Course', 'Appetizers']:
            forced = cat_items[cat_items['diet_code'] == 1]
            if not forced.empty:
                cat_items = forced
            elif region_relaxed:
                forced2 = base_cat[base_cat['diet_code'] == 1]
                if not forced2.empty:
                    cat_items = forced2

        cat_items = cat_items.sort_values('final_score', ascending=False)

        if cat_items.empty:
            meal_ui_elements.append(html.Div(
                f"[{category.upper()}] RETRIEVAL FAILED — no matching items.",
                style={'background': '#fee2e2', 'border': '1px solid #fca5a5',
                       'borderRadius': '12px', 'padding': '16px',
                       'color': '#ef4444', 'fontSize': '13px', 'fontWeight': '600'}
            ))
            continue

        primary_kcal = int(cat_items.iloc[0]['calories_kcal'])
        if current_kcal + primary_kcal > meal_kcal_limit:
            meal_ui_elements.append(html.Div(
                f"[{category.upper()}] BUDGET EXHAUSTED — needs {primary_kcal} kcal, "
                f"remaining {int(meal_kcal_limit - current_kcal)} kcal.",
                style={'background': '#f1f5f9', 'border': '1px dashed #cbd5e1',
                       'borderRadius': '12px', 'padding': '16px',
                       'color': TEXT_MUTED, 'fontSize': '13px', 'fontWeight': '600'}
            ))
            continue

        current_kcal += primary_kcal
        cards = []

        for rank_idx, (_, item) in enumerate(cat_items.head(5).iterrows()):
            rank     = rank_idx + 1
            kcal_val = int(item['calories_kcal'])
            img_src  = f"/images/{item['food_id']}.jpg"
            diag_data = f"{item['food_name']}|{kcal_val}|{item['fodmap_level']}|{ibs}|{item['final_score']}|{tdee}"

            badges = []
            if item['diet_code'] == 1:
                badges.append(html.Span("NON-VEG", style={
                    'background': '#fee2e2', 'color': '#ef4444', 'padding': '4px 8px',
                    'borderRadius': '4px', 'fontSize': '10px', 'fontWeight': '800', 'marginLeft': '8px'
                }))
            if region_relaxed:
                badges.append(html.Span("REGION RELAXED", style={
                    'background': '#fef3c7', 'color': '#d97706', 'padding': '4px 8px',
                    'borderRadius': '4px', 'fontSize': '10px', 'fontWeight': '800', 'marginLeft': '8px'
                }))

            card = html.Div(
                style={'minWidth': '100%', 'maxWidth': '100%', 'display': 'flex',
                       'padding': '16px 20px', 'boxSizing': 'border-box'},
                children=[
                    html.Div(
                        style={'position': 'relative', 'height': '200px', 'width': '220px',
                               'minWidth': '220px', 'flexShrink': '0'},
                        children=[
                            html.Div(f"#{rank}", style={
                                'position': 'absolute', 'top': '8px', 'left': '8px',
                                'background': TEXT_PRIMARY, 'color': '#fff', 'fontWeight': '800',
                                'padding': '4px 10px', 'borderRadius': '6px', 'fontSize': '13px',
                                'zIndex': '10', 'boxShadow': '0 4px 10px rgba(0,0,0,0.2)'
                            }),
                            html.Img(src=img_src, style={
                                'width': '100%', 'height': '100%', 'objectFit': 'cover',
                                'borderRadius': '12px', 'boxShadow': '0 4px 15px rgba(0,0,0,0.08)',
                                'backgroundColor': '#e2e8f0'
                            })
                        ]
                    ),
                    html.Div(
                        style={'flex': '1', 'marginLeft': '28px', 'display': 'flex',
                               'flexDirection': 'column', 'justifyContent': 'center'},
                        children=[
                            html.Div(
                                style={'display': 'flex', 'justifyContent': 'space-between',
                                       'alignItems': 'flex-start'},
                                children=[
                                    html.Div([
                                        html.Div(
                                            style={'display': 'flex', 'alignItems': 'center',
                                                   'marginBottom': '6px'},
                                            children=[
                                                html.H4(item['food_name'], style={
                                                    'margin': '0', 'color': TEXT_PRIMARY,
                                                    'fontWeight': '800', 'fontSize': '22px'
                                                }),
                                                html.Div(badges, style={'display': 'flex'})
                                            ]
                                        ),
                                        html.Div(category.replace('_', ' '), style={
                                            'color': ACCENT, 'fontSize': '12px', 'fontWeight': '800',
                                            'textTransform': 'uppercase', 'letterSpacing': '0.05em'
                                        })
                                    ]),
                                    html.Div(f"Match: {item['final_score']*100:.1f}%", style={
                                        'background': '#d1fae5', 'color': SUCCESS,
                                        'padding': '8px 16px', 'borderRadius': '8px',
                                        'fontWeight': '800', 'fontSize': '15px',
                                        'boxShadow': '0 2px 8px rgba(16,185,129,0.2)'
                                    })
                                ]
                            ),
                            html.Div(
                                style={'marginTop': '16px', 'display': 'flex', 'gap': '12px'},
                                children=[
                                    html.Button(
                                        f"🔥 {kcal_val} KCAL  |  🧬 FODMAP: {item['fodmap_level']}",
                                        id={'type': 'diag-btn', 'index': diag_data},
                                        style={
                                            'background': '#f1f5f9', 'color': TEXT_MUTED,
                                            'padding': '8px 16px', 'borderRadius': '6px',
                                            'fontSize': '12px', 'fontWeight': '700',
                                            'border': '1px solid rgba(0,0,0,0.05)',
                                            'cursor': 'pointer', 'transition': 'all 0.2s',
                                            'boxShadow': '0 2px 4px rgba(0,0,0,0.02)'
                                        }
                                    )
                                ]
                            ),
                            html.P(
                                f"Clinical Assessment: {item['fodmap_level']} FODMAP profile. "
                                f"Prioritises systemic bioavailability within the {kcal_val} kcal threshold.",
                                style={'marginTop': '16px', 'color': TEXT_MUTED, 'fontSize': '14px',
                                       'lineHeight': '1.6', 'fontWeight': '500',
                                       'borderTop': '1px solid rgba(0,0,0,0.05)', 'paddingTop': '16px'}
                            )
                        ]
                    )
                ]
            )
            cards.append(card)

        meal_ui_elements.append(html.Div(
            id=f'carousel-window-{slot_idx}',
            style={
                'display': 'flex', 'alignItems': 'center',
                'background': 'rgba(255,255,255,0.6)', 'borderRadius': '16px',
                'position': 'relative', 'border': '1px solid rgba(0,0,0,0.04)',
                'overflow': 'hidden', 'boxShadow': 'inset 0 2px 10px rgba(0,0,0,0.02)',
                'height': '240px', 'margin': '8px 0'
            },
            children=[
                html.Button("◀", id={'type': 'btn-left',  'index': slot_idx}, style={
                    'position': 'absolute', 'left': '12px', 'zIndex': '20',
                    'background': '#fff', 'color': TEXT_PRIMARY,
                    'border': '1px solid rgba(0,0,0,0.1)', 'borderRadius': '50%',
                    'width': '36px', 'height': '36px', 'cursor': 'pointer',
                    'boxShadow': '0 4px 10px rgba(0,0,0,0.1)',
                    'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'
                }),
                html.Div(
                    className="carousel-track-lux", **{'data-index': '0'},
                    style={'display': 'flex', 'width': '100%', 'height': '100%',
                           'transition': 'transform 0.8s cubic-bezier(0.25,1,0.5,1)'},
                    children=cards
                ),
                html.Button("▶", id={'type': 'btn-right', 'index': slot_idx}, style={
                    'position': 'absolute', 'right': '12px', 'zIndex': '20',
                    'background': '#fff', 'color': TEXT_PRIMARY,
                    'border': '1px solid rgba(0,0,0,0.1)', 'borderRadius': '50%',
                    'width': '36px', 'height': '36px', 'cursor': 'pointer',
                    'boxShadow': '0 4px 10px rgba(0,0,0,0.1)',
                    'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'
                })
            ]
        ))

    telemetry = (
        f"> SYSTEM BOOT... OK\n"
        f"> HARDWARE: {str(device).upper()}\n"
        f"> ARCHITECTURE: BioPerceiverFusionNet (ResNet)\n"
        f"> PARAMETERS: {total_parameters:,}\n\n"
        f"[INFERENCE]\n"
        f"> LATENCY: {latency_ms:.3f} ms\n"
        f"> BIO INPUT:  [{firm:.3f}, {bact:.3f}, {prot:.3f}, {acti:.3f}]\n"
        f"> CLIN INPUT: [{age}, {bmi}, {tdee}, {diet}, {ibs}]\n"
        f"> OUTPUT DIM: {len(foods_df)}\n\n"
        f"[DISTRIBUTION]\n"
        f"> MEAN AFFINITY: {mean_affinity:.4f}\n"
        f"> PEAK SCORE:    {np.max(base_preds):.4f}\n"
        f"> MIN SCORE:     {np.min(base_preds):.4f}\n"
        f"> VARIANCE:      {np.var(base_preds):.6f}\n\n"
        f"> STATUS: NOMINAL."
    )

    return (
        xai_fig,
        meal_ui_elements,
        f"{mean_affinity*100:.1f}%",
        f"{current_kcal} / {meal_kcal_limit:.0f} KCAL",
        telemetry
    )


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
