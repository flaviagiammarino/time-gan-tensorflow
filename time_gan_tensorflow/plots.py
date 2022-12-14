import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot(actual, reconstructed, simulated):
    '''
    Plot the actual, reconstructed and simulated time series.
    '''
    
    fig = make_subplots(
        subplot_titles=['Actual', 'Reconstructed', 'Simulated'],
        vertical_spacing=0.15,
        rows=3,
        cols=1
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=40, b=10, l=10, r=10),
        font=dict(
            family='Arial',
            color='#1b1f24',
            size=8,
        ),
        legend=dict(
            traceorder='normal',
            font=dict(
                family='Arial',
                color='#1b1f24',
                size=10,
            ),
        ),
    )
    
    fig.update_annotations(
        font=dict(
            family='Arial',
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
            name='Average',
            showlegend=True,
            mode='lines',
            line=dict(
                color='#0969da',
                width=1,
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
            showlegend=False,
            mode='lines',
            line=dict(
                color='#0969da',
                width=1,
            )
        ),
        row=2,
        col=1
    )
    
    # plot the simulated time series
    for i in range(simulated.shape[1]):
        fig.add_trace(
            go.Scatter(
                y=simulated[:, i],
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
            y=np.mean(simulated, axis=1),
            showlegend=False,
            mode='lines',
            line=dict(
                color='#0969da',
                width=1,
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
                family='Arial',
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
            title='Value',
            color='#424a53',
            tickfont=dict(
                family='Arial',
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
