# -*- coding: utf-8 -*-


from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import plotly.graph_objects as go



def create_pie_four_subplots(df, title, labels):
  fig = make_subplots(
      rows=2,
      cols=2,
      specs=[[{'type': 'pie'}] * 2, [{'type': 'pie'}] * 2],
      subplot_titles=[df.columns[0], df.columns[1], df.columns[2], df.columns[3]],
  )

  labels = labels

  for i, col in enumerate(df.columns):
      counts = df[col].value_counts()
      trace = go.Pie(values=counts.values, labels=counts.index, name=col, textinfo='percent', hoverinfo='label+text')
      fig.add_trace(trace, row=i // 2 + 1, col=i % 2 + 1)

  fig.update_layout(
      title={
          'text': title,
          'y': 0.99,
          'x': 0.5,
          'xanchor': 'center',
          'yanchor': 'top',
          'font': dict(size=24)
      },
      margin=dict(l=20, r=20, t=50, b=20),
      legend=dict(
          traceorder='normal',
          font=dict(
              family='sans-serif',
              size=16,
          ),

          title=dict(
              text="Responses",
              font=dict(
                  family='Arial',
                  size=20,
                  color='Black'
              )
          )
      )
  )

  fig.show()

def create_bin_multi_bar_plot(data, cols, subplot_title,xtitle,ytitle):
    fig = go.Figure()
    for col in cols:
        graph_data = data[col].value_counts()
        fig.add_trace(go.Bar(x=[col + ' No', col+' Yes'], y=graph_data.values,
                             name=col, text=graph_data.values, textposition='auto'))
    
    fig.update_layout(
        title=subplot_title,
        xaxis_title=xtitle,
        yaxis_title=ytitle,
    )

    fig.show()



def find_optimal_clusters(data, maximumClusters):
    k_values = list(range(1, maximumClusters+1))
    wss = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data.values.reshape(-1, 1))
        wss.append(kmeans.inertia_)
    dx = np.gradient(wss)
    dx2 = np.gradient(dx)
    elbow = -1
    for i in range(len(dx2) - 1):
        if (dx2[i] > 0) and (dx2[i + 1] < 0):
            elbow = i
            break
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k_values, y=wss, mode='lines+markers', name='WSS'))
    fig.add_trace(go.Scatter(x=[k_values[elbow]], y=[wss[elbow]], mode='markers', name='Elbow',
                             marker=dict(size=10, color='red', line=dict(width=2, color='red'), symbol='circle-open')))
    fig.update_layout(title='Elbow method for determining the optimal number of clusters',
                      xaxis_title='Number of clusters (k)', yaxis_title='Within-cluster sum of squares',
                      hovermode='x')
    k = elbow + 1
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data.values.reshape(-1, 1))
    labels = kmeans.predict(data.values.reshape(-1, 1))
    bounds = []
    for i in range(k):
        clusters = data[labels == i]
        lower = clusters.min()
        upper = clusters.max()
        bounds.append((lower, upper))

   
    bounds = sorted(bounds, key=lambda x: x[0])
    clusterIdx = [i for i in range(k)]
    clusterIdx = sorted(clusterIdx, key=lambda x: bounds[x][0])
    labels = [clusterIdx[label] for label in labels]
    intervals = []
    for i in range(k - 1):
        low = bounds[i][0]
        upper = bounds[i+1][0]
        intervals.append((low,upper))
    last = (bounds[k - 1][0], bounds[k - 1][1])
    intervals.append(last)

    return fig,labels,bounds

def create_bar_plot(data, x_col, title, xtitle, ytitle, color=None):
  graph_data = data[x_col].value_counts()
  if color is None:
      colors = px.colors.sequential.Plasma_r[:len(graph_data)]
  else:
      colors = [color] * len(graph_data)
  color_scale = [[x, c] for x, c in zip(graph_data.values, colors)]
  colors_ = [c for x, c in sorted(color_scale)]

  fig = go.Figure(go.Bar(
      x=graph_data.index,
      y=graph_data.values,
      marker_color=colors_,
      text=graph_data.values,
      textposition='auto',
      name=''
  ))

  fig.update_layout(
      title=title,
      xaxis_title=xtitle,
      yaxis_title=ytitle
  )
  fig.show()

def create_bar_subplot(data, column1, column2, title1, title2, subplot_title):
    graph1_data = data[column1].value_counts()
    graph2_data = data[column2].value_counts()

    colors = px.colors.sequential.Plasma_r[:len(graph1_data)]
    color_scale = [[x, c] for x, c in zip(graph1_data.values, colors)]
    colors_1 = [c for x, c in sorted(color_scale)]

    colors = px.colors.sequential.Plasma_r[:len(graph2_data)]
    color_scale = [[x, c] for x, c in zip(graph2_data.values, colors)]
    colors_2 = [c for x, c in sorted(color_scale)]

    fig1 = go.Figure(go.Bar(
        x=graph1_data.index,
        y=graph1_data.values,
        marker_color=colors_1,
        text=graph1_data.values,
        textposition='auto',
        name=''
    ))

    fig1.update_layout(
        title=title1,
        xaxis_title=column1,
        yaxis_title='Number of responses'
    )
    fig2 = go.Figure(go.Bar(
        x=graph2_data.index,
        y=graph2_data.values,
        marker_color=colors_2,
        text=graph2_data.values,
        textposition='auto',
        name=''

    ))
    fig2.update_layout(
        title=title2,
        xaxis_title=column2,
        yaxis_title='Number of responses'
    )
    figs = make_subplots(rows=1, cols=2, shared_yaxes=True)
    figs.add_trace(fig1.data[0], row=1, col=1)
    figs.add_trace(fig2.data[0], row=1, col=2)
    figs.update_layout(height=700, width=2000)
    figs.update_layout(
        title={
            'text': subplot_title,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )

    figs.update_xaxes(title_text=title1, row=1, col=1)
    figs.update_xaxes(title_text=title2, row=1, col=2)
    figs.show()

def create_pie_table(data,col,categoryList_old,categoryList_new,layoutTitle,legendTitle,graphTitle):
  fig = make_subplots(rows=1, cols=2,specs=[[{"type": "pie"}, {"type": "table"}]], column_width=[0.5, 0.5])

  categories = pd.Categorical(data[col], categories=categoryList_old, ordered=True)
  if categoryList_new != None:
    categories = categories.rename_categories(categoryList_new)
  colors = px.colors.qualitative.Plotly

  counts = categories.value_counts()

  hover_text = [f"{p}: {c}" for p, c in zip(counts.index, counts.values)]

  pie = go.Figure(data=[go.Pie(labels=counts.index, values=counts.values,
                              hovertemplate='%{label}: %{value} (%{percent:.1%})<br>', 
                              customdata=hover_text, 
                              marker=dict(colors=colors[:len(counts)]),
                              name='')])
  pie.update_layout(title=layoutTitle, legend=dict(title=legendTitle), showlegend=True)
  fig.add_trace(pie.data[0], row=1, col=1)

  fig.add_trace(go.Table(
      header=dict(values=[legendTitle, 'Count']),
      cells=dict(values=[counts.index, counts.values])
  ), row=1, col=2)

  fig.update_layout(height=600, width=1000, title_text=graphTitle)
  fig.show()


def create_plotly_table(title, df, column):
    counts = df[column].value_counts()
    total_responses = counts.sum()

    percentages = counts / total_responses * 100

    countDF = pd.DataFrame({'Counts': counts, 'Percentages': percentages})

    fig = go.Figure(data=[go.Table(
        header=dict(values=['Response', 'Counts', 'Percentages'],
                    line_color='darkslategray',
                    fill_color='lightskyblue',
                    align='left'),
        cells=dict(values=[countDF.index, countDF['Counts'], countDF['Percentages']],
                   line_color='darkslategray',
                   fill_color='lightcyan',
                   align='left'))
    ])

    fig.update_layout(title_text=title)

    fig.show()