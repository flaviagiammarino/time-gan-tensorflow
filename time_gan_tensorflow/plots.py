import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot(actual, reconstructed, synthetic):
    '''
    Plot the actual, reconstructed and synthetic time series.
    '''
    
    fig = make_subplots(
        subplot_titles=['Actual', 'Reconstructed', 'Synthetic'],
        vertical_spacing=0.15,
        rows=3,
        cols=1
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=60, b=60, l=30, r=30),
        font=dict(
            color='#1b1f24',
            size=8,
        ),
        legend=dict(
            traceorder='normal',
            font=dict(
                color='#1b1f24',
                size=10,
            ),
            x=0,
            y=-0.1,
            orientation='h'
        ),
    )
    
    fig.update_annotations(
        font=dict(
            color='#1b1f24',
            size=12,
        )
    )
    
    # plot the actual time series
    for i in range(actual.shape[1]):
        fig.add_trace(
            go.Scatter(
                y=actual[:, i],
                showlegend=False,
                mode='lines',
                line=dict(
                    color='rgba(175,184,193,0.2)',
                    width=0.5
                )
            ),
            row=1,
            col=1
        )

    fig.add_trace(
        go.Scatter(
            y=np.mean(actual, axis=1),
            name='Actual Avg.',
            showlegend=True,
            mode='lines',
            line=dict(
                color='#0969da',
                width=1,
                shape='spline',
            )
        ),
        row=1,
        col=1
    )
    
    # plot the reconstructed time series
    for i in range(reconstructed.shape[1]):
        fig.add_trace(
            go.Scatter(
                y=reconstructed[:, i],
                showlegend=False,
                mode='lines',
                line=dict(
                    color='rgba(175,184,193,0.2)',
                    width=0.5
                )
            ),
            row=2,
            col=1
        )
    
    fig.add_trace(
        go.Scatter(
            y=np.mean(reconstructed, axis=1),
            name='Reconstructed Avg.',
            showlegend=True,
            mode='lines',
            line=dict(
                color='#0969da',
                width=1,
                shape='spline',
                dash='dash',
            )
        ),
        row=2,
        col=1
    )
    
    # plot the synthetic time series
    for i in range(synthetic.shape[1]):
        fig.add_trace(
            go.Scatter(
                y=synthetic[:, i],
                showlegend=False,
                mode='lines',
                line=dict(
                    color='rgba(175,184,193,0.2)',
                    width=0.5
                )
            ),
            row=3,
            col=1
        )
        
    fig.add_trace(
        go.Scatter(
            y=np.mean(synthetic, axis=1),
            name='Synthetic Avg.',
            showlegend=True,
            mode='lines',
            line=dict(
                color='#0969da',
                width=1,
                shape='spline',
                dash='dot',
            )
        ),
        row=3,
        col=1
    )
    
    for i in [1, 2, 3]:
        fig.update_xaxes(
            title='Time',
            color='#424a53',
            tickfont=dict(
                color='#6e7781',
                size=6,
            ),
            linecolor='#eaeef2',
            mirror=True,
            showgrid=False,
            row=i,
            col=1
        )
        
        fig.update_yaxes(
            range=[0.9 * np.min(actual), 1.1 * np.max(actual)],
            title='Value',
            color='#424a53',
            tickfont=dict(
                color='#6e7781',
                size=6,
            ),
            linecolor='#eaeef2',
            mirror=True,
            showgrid=False,
            zeroline=False,
            row=i,
            col=1
        )
    
    return fig
