import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import odeint
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# 1. Definición del sistema usando la TRANSFORMACIÓN DE LIÉNARD
def lienard_forced(z, t, mu, A, omega):
    x, y = z
    dxdt = y + mu * (x - (x**3) / 3.0)
    dydt = -x + A * np.sin(omega * t)
    return [dxdt, dydt]

z0 = [0.5, 0.0]              
t_end = 200                  
num_points = 10000
t = np.linspace(0, t_end, num_points)
dt = t[1] - t[0]             

# 2. Inicializar la aplicación Dash
app = dash.Dash(__name__)
app.title = "Oscilador Van der Pol"

# ¡ESTA ES LA LÍNEA VITAL PARA RENDER!
server = app.server 

# 3. Diseño de la interfaz
app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'padding': '20px'}, children=[
    html.H2("Análisis del Oscilador de Van der Pol (Plano de Liénard)"),
    
    html.Div(style={'display': 'flex', 'gap': '40px', 'marginBottom': '20px', 'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '10px'}, children=[
        html.Div(style={'flex': 1}, children=[
            html.Label("Parámetro no lineal (Mu):", style={'fontWeight': 'bold'}),
            dcc.Slider(min=0.1, max=5.0, step=0.1, value=1.5, id='mu-slider', marks={i: str(i) for i in range(1, 6)})
        ]),
        html.Div(style={'flex': 1}, children=[
            html.Label("Amplitud del Forzamiento (A):", style={'fontWeight': 'bold'}),
            dcc.Slider(min=0.0, max=15.0, step=0.1, value=1.2, id='A-slider', marks={i: str(i) for i in range(0, 16, 3)})
        ]),
        html.Div(style={'flex': 1}, children=[
            html.Label("Frecuencia (Omega):", style={'fontWeight': 'bold'}),
            dcc.Slider(min=0.5, max=5.0, step=0.1, value=1.7, id='omega-slider', marks={i: str(i) for i in range(1, 6)})
        ])
    ]),
    dcc.Graph(id='vdp-graph', style={'height': '70vh'})
])

# 4. Lógica de actualización (Callback)
@app.callback(
    Output('vdp-graph', 'figure'),
    [Input('mu-slider', 'value'),
     Input('A-slider', 'value'),
     Input('omega-slider', 'value')]
)
def update_graph(mu, A, omega):
    transient_fft_ratio = 0.1  
    transient_plot_ratio = 0.8 
    fft_idx = int(num_points * transient_fft_ratio)
    plot_idx = int(num_points * transient_plot_ratio)

    sol_unforced = odeint(lienard_forced, z0, t, args=(mu, 0, 0))
    x_unforced = sol_unforced[:, 0]
    
    t_steady = t[plot_idx:]
    x_unforced_steady = x_unforced[plot_idx:]
    
    peaks, _ = find_peaks(x_unforced_steady)
    if len(peaks) > 1:
        T_natural = np.mean(np.diff(t_steady[peaks]))
        omega_natural = 2 * np.pi / T_natural
    else:
        omega_natural = 1.0 

    ratio_omega = omega / omega_natural

    sol_forced = odeint(lienard_forced, z0, t, args=(mu, A, omega))
    x_forced = sol_forced[:, 0]
    
    x_unforced_fft = x_unforced[fft_idx:]
    x_forced_fft = x_forced[fft_idx:]
    N_fft = len(x_unforced_fft)
    
    yf_unforced = fft(x_unforced_fft)
    xf_unforced = fftfreq(N_fft, dt)[:N_fft//2]
    w_unforced = 2 * np.pi * xf_unforced 
    mag_unforced = 2.0/N_fft * np.abs(yf_unforced[0:N_fft//2])
    
    yf_forced = fft(x_forced_fft)
    xf_forced = fftfreq(N_fft, dt)[:N_fft//2]
    w_forced = 2 * np.pi * xf_forced 
    mag_forced = 2.0/N_fft * np.abs(yf_forced[0:N_fft//2])

    fig = make_subplots(
        rows=1, cols=3, 
        subplot_titles=(f'ω₀ ≈ {omega_natural:.2f} | ω_forz = {omega:.2f} | Ratio ω/ω₀ = {ratio_omega:.2f}', 'Plano de Liénard', 'Espectro de Frecuencias (FFT)'),
        horizontal_spacing=0.08
    )

    fig.add_trace(go.Scatter(x=t[plot_idx:], y=x_unforced[plot_idx:], mode='lines', name='Libre', line=dict(color='green', width=1.5, dash='dot'), opacity=0.5), row=1, col=1)
    fig.add_trace(go.Scatter(x=t[plot_idx:], y=x_forced[plot_idx:], mode='lines', name='Forzado', line=dict(color='#1f77b4', width=2)), row=1, col=1)

    x_vals = np.linspace(-3.5, 3.5, 400)
    y_nullcline_x = -mu * (x_vals - (x_vals**3) / 3.0) 
    
    fig.add_trace(go.Scatter(x=x_vals, y=y_nullcline_x, mode='lines', name='Nulclina dx/dt=0', line=dict(color='green', width=2, dash='dash')), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0, 0], y=[-10, 10], mode='lines', name='Nulclina dy/dt=0 (Libre)', line=dict(color='orange', width=2, dash='dash')), row=1, col=2)
    
    fig.add_trace(go.Scatter(x=sol_unforced[plot_idx:, 0], y=sol_unforced[plot_idx:, 1], mode='lines', name='Órbita Libre', line=dict(color='blue', width=1.5), opacity=0.6), row=1, col=2)
    fig.add_trace(go.Scatter(x=sol_forced[plot_idx:, 0], y=sol_forced[plot_idx:, 1], mode='lines', name='Órbita Forzada', line=dict(color='red', width=1)), row=1, col=2)

    fig.add_trace(go.Scatter(x=w_unforced, y=mag_unforced, mode='lines', name='FFT Libre', line=dict(color='green', width=2), opacity=0.7), row=1, col=3)
    fig.add_trace(go.Scatter(x=w_forced, y=mag_forced, mode='lines', name='FFT Forzado', line=dict(color='purple', width=2), opacity=0.7), row=1, col=3)
    
    fig.add_vline(x=omega_natural, line_width=1.5, line_dash="dash", line_color="green", row=1, col=3, name='Omega Natural')
    if A > 0:
        fig.add_vline(x=omega, line_width=1.5, line_dash="dash", line_color="purple", row=1, col=3, name='Omega Forzamiento')

    y_lim_phase = np.max(np.abs(sol_forced[:, 1])) * 1.2 if mu > 0.5 else 3

    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40), legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5), plot_bgcolor='white', paper_bgcolor='#f9f9f9')
    fig.update_xaxes(title_text="Tiempo", row=1, col=1, showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(title_text="Amplitud x", row=1, col=1, showgrid=True, gridcolor='lightgray')
    fig.update_xaxes(title_text="x", range=[-3.5, 3.5], row=1, col=2, showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(title_text="y (Variable Liénard)", range=[-y_lim_phase, y_lim_phase], row=1, col=2, showgrid=True, gridcolor='lightgray')
    fig.update_xaxes(title_text="Frecuencia Angular (ω) [rad/s]", range=[0, max(omega_natural, omega) * 3], row=1, col=3, showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(title_text="Magnitud", row=1, col=3, showgrid=True, gridcolor='lightgray')

    return fig

if __name__ == '__main__':
    app.run(debug=True)
