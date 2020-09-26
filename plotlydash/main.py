import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import os
from sklearn.cluster import KMeans
import math
from dash.dependencies import Input, Output, State
import datetime
import pyodbc
from flask import Flask
#remove type and unit
"""
con = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};'
                     'SERVER=dbtopagentserver.database.windows.net;'
                     'DATABASE=db_topagents;'
                     'UID=readonlylogin;'
                     'PWD=Topagent1;')
mapbox_access_token = "pk.eyJ1IjoibG5pa2UxIiwiYSI6ImNqOHEwb3dwajBuengycm8wcnR2bjM0NzIifQ.0TzrVKzKLDnaFxwzddgt6g"

# df_input = pd.read_sql(
#     "SELECT t1.OriginalPrice, t3.SqFt, t3.Bed, t2.Neighborhood, t1.SellersAgent, t1.ListedDate, t2.FormattedAddress, t3.Type, t1.DollarSqft, t1.ClosedDate, t1.ClosedPrice, t2.Lat, t2.Lng, t1.SellersAgent, t1.SellersCompany"
#    " FROM ta_Transaction t1 "
#    " LEFT JOIN ta_BuildingAddress t2 ON t1.BuildingAddressId=t2.Id"
#    " LEFT JOIN ta_Unit t3 ON t1.UnitId=t3.Id"
#    " WHERE t2.Neighborhood IS NOT NULL",con)
# con.commit()
#
# df_input.rename(columns={'ClosedDate':'ClosingDate','DollarSqft':'DollarSqFt','Bed':'Beds','FormattedAddress':'Address','ClosedPrice':'ClosingPrice'},inplace=True)
# df_input['DollarSqFt'] = df_input['ClosingPrice'].astype(float)/df_input['SqFt'].astype(float)
# df = df_input
# df=df[df["Lat"].notnull()]
# df['ClosingDate'] = pd.to_datetime(df['ClosingDate'], errors = 'coerce')
# df['ListedDate'] = pd.to_datetime(df['ListedDate'], errors = 'coerce')
# df['Type']=df['Type'].replace('CONDP','COOP' )
# transactions = ((df['ClosingDate']>'2008-12-31') & (df['ClosingDate']<str(datetime.datetime.today().strftime('%Y-%m-%d')))) & (df['ClosingDate'].notnull())
# df = df.loc[transactions]
# print(max(df['ClosingDate']))
################################### help functions ###########################################################
#get all neighborhood for the drop down menu
def get_neighborhood_list():
    print("get_neighborhood_list()")
    neighborhood=pd.read_sql("SELECT DISTINCT Neighborhood from ta_BuildingAddress;",con)
    con.commit()
    neighborhood=neighborhood["Neighborhood"].tolist()
    return neighborhood

def df_ftxt(txt, column):
    # for get_cluster_info
    print("df_ftxt()")
    # choose the column
    other = "DollarSqFt"
    if column == "DollarSqFt":
        other = "ClosingPrice"


    dict_list = []
    for t in txt.split('\n'):
        if t != '':
            tmp = t.split(':')
            dict_list.append({'Neighbor':tmp[0], column:tmp[1], other:tmp[2], 'Num':tmp[3]})
    dataframe = pd.DataFrame.from_records(dict_list)
    dataframe['DollarSqFt']=pd.to_numeric(dataframe['DollarSqFt']).astype(int)
    dataframe['ClosingPrice']=pd.to_numeric(dataframe['ClosingPrice']).astype(int)
    dataframe['SqFt'] = (dataframe['ClosingPrice'] / dataframe['DollarSqFt']).astype(int)

    dataframe = dataframe.sort_values(by=column, ascending=False)

    dataframe['DollarSqFt'] = dataframe['DollarSqFt'].apply(intWithCommas)
    dataframe['ClosingPrice']=dataframe['ClosingPrice'].apply(intWithCommas)
    dataframe['SqFt']=dataframe['SqFt'].apply(intWithCommas_n)
    dataframe['Num'] = pd.to_numeric(dataframe['Num']).astype(int)

    dataframe = dataframe[['Num','Neighbor','ClosingPrice','DollarSqFt','SqFt']]

    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(len(dataframe))]
    )


def comma(x):
    # for cluster_figure
    if len(x)!=0:
        return x[:-2]
    return x


def get_average(df_clusters, column, neighbor):
    #for cluster_figure
    print("get_average()")
    # choose the year
    # date_y = date[:4]
    # date_m = date[4:7]
    df1 = df_clusters.set_index('Neighborhood').dropna()

    # choose the column
    other = "DollarSqFt"
    if column == "DollarSqFt":
        other = "ClosingPrice"

    n_clusters = max(int(len(df1) / 10), 10)
    if int(len(df1)) < 10:
        n_clusters = int(len(df1))

    txt = dict([(i, []) for i in range(n_clusters)])
    val1 = dict([(i, []) for i in range(n_clusters)])
    val2 = dict([(i, []) for i in range(n_clusters)])
    val3 = dict([(i, []) for i in range(n_clusters)])

    s = KMeans(n_clusters=n_clusters, random_state=0).fit(df1[column].values.reshape(len(df1), 1))
    for i in range(len(df1)):
        txt[s.labels_[i]].append(str(df1.index[i]))
        val1[s.labels_[i]].append(df1[column].values[i])
        val2[s.labels_[i]].append(df1[other].values[i])
        val3[s.labels_[i]].append(df1['Num'].values[i])

    global df_view

    df_view = pd.DataFrame({'types': txt, 'nums': val1, 'other': val2, 'counter': val3})
    df_view['means'] = df_view['nums'].apply(np.mean)
    df_view['markers'] = df_view.apply(gen_marker, axis=1)
    df_view.sort_values(by='means', inplace=True)
    df_view.index = range(len(df_view))

    color_label = [0 * i for i in range(n_clusters)]
    text = ['' for i in range(n_clusters)]
    for nei in neighbor:
        color_label += df_view['types'].apply(lambda x: int(nei in x))
        text[find_nei(nei, df_view)] += nei + ', '
    colors = dict([(0, 'rgba(204,204,204,1)')] + [(i, 'rgba(222,45,38,0.8)') for i in range(1, len(neighbor)+2)])


    text = list(map(comma,text))
    data = [go.Bar(
        x=[str(x) for x in range(n_clusters)],
        y=df_view['means'],
        text=text,
        marker=dict(
            color=[colors[i] for i in color_label],
        ))]

    layout = go.Layout(

        xaxis=dict(
            title=''
        ),
        yaxis=dict(
            title='Ave' + column
        ),
        bargap=0,
        bargroupgap=0.1
    )

    return{
    'data':data ,

    'layout' : layout}


def gen_color(line, c_range, c_min):
    # for update_map
    c = (line-c_min)/c_range*510
    c2 = 255 - (c-255)*int(c>255)
    c1 = min(c,255)
    return 'rgba('+str(int(c1))+','+str(int(c2))+',20,1)'


def intWithCommas(x):
    # for printing format "$1,000,000"
    if x < 0:
        return '-' + intWithCommas(-x)
    result = ''
    while x >= 1000:
        x, r = divmod(x, 1000)
        result = ",%03d%s" % (r, result)
    return "$%d%s" % (x, result)


def intWithCommas_n(x):
    # for printing format"1,000,000"

    if x < 0:
        return '-' + intWithCommas(-x)
    result = ''
    while x >= 1000:
        x, r = divmod(x, 1000)
        result = ",%03d%s" % (r, result)
    return "%d%s" % (x, result)


def mult_selections(selections_list, column, dataframe):
    # filter dataframe by selected items
    print("mult_selections()")
    if type(selections_list)==str or type(selections_list)==int:
        selections_list = [selections_list]
    index_label = []
    for i in selections_list:
        index_label.append(list(dataframe[column]==i))
    index_array = np.array(index_label)
    return index_array.sum(axis=0)>0


def update_plot_beds(selected_sqft, selected_neighborhood, selected_cp, selected_beds, selected_tog, selected_ds, selected_types,dataframe):
    # for
    print("update_plot_beds()")

    closing_price_labels = (dataframe['ClosingPrice'] < selected_cp[1]) & (dataframe['ClosingPrice'] > selected_cp[0])
    sqft_labels = (dataframe['SqFt'] < selected_sqft[1]) & (dataframe['SqFt'] > selected_sqft[0])
    dollarsqft_label = (dataframe['DollarSqFt'] < selected_ds[1]) & (dataframe['DollarSqFt'] > selected_ds[0])
    types_labels = mult_selections(np.array(selected_types), 'Type', dataframe)

    # generate data for plot line
    traces = []
    for i in range(len(selected_beds)):
        for j in range(len(selected_neighborhood)):
            beds_labels = mult_selections(selected_beds[i], 'Beds', dataframe)
            neighborhood_labels = mult_selections(selected_neighborhood[j], 'Neighborhood', dataframe)
            labels = types_labels&beds_labels & neighborhood_labels & closing_price_labels & sqft_labels & dollarsqft_label

            df_output = dataframe.loc[labels][['DollarSqFt', 'ClosingPrice', 'ClosingDate']]
            df_output.index = df_output['ClosingDate']
            del df_output['ClosingDate']
            df_output = df_output.resample('Q').mean()

            print(selected_tog)

            if selected_tog == 1:
                y1 = df_output['DollarSqFt'].interpolate()
                y1axis = {'title': 'Avg$/SqFt'}
            else:
                y1 = df_output['ClosingPrice'].interpolate()
                y1axis = {'title': 'ClosingPrice'}

            strtdate = {'1': '-03-31', '2': '-06-30', '3': '-09-30', '4': '-12-31'}

            traces.append(go.Scatter(
                x=[str(i) for i in list(y1.index.year.astype(str) + [strtdate[j] for j in y1.index.quarter.astype(str)])],
                #                 customdata = [selected_beds[i] for j in range(len(y1))],
                y=y1.values,
                text=[k for k in list(y1.index.year.astype(str) + [strtdate[j] for j in y1.index.quarter.astype(str)])],
                mode='lines',
                opacity=0.7,
                marker={
                    'size': 15,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=str(selected_neighborhood[j])+'-'+str(selected_beds[i])+"-bed"
            ))
    layout = go.Layout(
        xaxis={'title': 'Time'},
        yaxis=y1axis,
        title = 'plot for beds'
    )

    return {'data':traces, 'layout':layout}


def gen_marker(line):
    s = ''
    for i in range(len(line.types)):
        s+= str(line.types[i])+':'+str(line.nums[i])+':'+str(line.other[i])+':'+str(line.counter[i])+'\n'
    return s


def find_nei(nei, df_view):
    for i in range(len(df_view['types'])):
        if nei in df_view['types'][i]:
            return i

def get_meter_chart(a,b,title):
    print("get_meter_chart()")
    if np.isinf(a):
        txt = "inf"
        string = get_shape(15)
    else:
        a = min(15,a)
        string = get_shape(a)
        txt = str(a)[:4]
    if np.isinf(b):
        txt2 = "inf"
        string2 = get_shape(15)
    else:
        b = min(15,b)
        string2 = get_shape(b)
        txt2 = str(b)[:4]
    base_chart = {
        "values": [36, 54, 54, 54, 54, 54, 54],
        "labels": ["-", "0-Sell", "3", "6", "9", "12", "15-Buy"],
        "domain": {"x": [0, 1]},
        "marker": {
            "colors": [
                'rgb(255, 255, 255)',
                'rgb(255, 255, 255)',
                'rgb(255, 255, 255)',
                'rgb(255, 255, 255)',
                'rgb(255, 255, 255)',
                'rgb(255, 255, 255)',
                'rgb(255, 255, 255)'
            ],
            "line": {
                "width": 1
            }
        },
        "name": "Gauge",
        "hole": .7,
        "type": "pie",
        "direction": "clockwise",
        "rotation": 197,
        "showlegend": False,
        "hoverinfo": "none",
        "textinfo": "label",
        "textposition": "outside"
    }
    meter_chart = {
        "values": [20, 12, 12, 12, 12, 12],
        "labels": [title, "level 1", "level 2", "level 3", "level 4", "level 5"],
        "marker": {
            'colors': [
                'rgb(255, 255, 255)',
                'rgb(232,226,202)',
                'rgb(226,210,172)',
                'rgb(223,189,139)',
                'rgb(223,162,103)',
                'rgb(226,126,64)'
            ]
        },
        "domain": {"x": [0,1]},
        "name": "Gauge",
        "hole": .8,
        "type": "pie",
        "direction": "clockwise",
        "rotation": 135,
        "showlegend": False,
        "textinfo": "label",
        "textposition": "inside",
        "hoverinfo": "none"
    }
    layout = {
        'xaxis': {
            'showticklabels': False,
            'autotick': False,
            'showgrid': False,
            'zeroline': False,
        },
        'yaxis': {
            'showticklabels': False,
            'autotick': False,
            'showgrid': False,
            'zeroline': False,
        },
        'shapes': [
            {
                'type': 'path',
                'path': string,
                'fillcolor': 'rgba(219, 220, 221, 0.5)',
                'line': {
                    'width': 0.9
                },
                'xref': 'paper',
                'yref': 'paper'
            },
            ############ new arrow
            {
                'type': 'path',
                'path': string2,
                'fillcolor': 'rgba(0, 0, 0, 1)',
                'line': {
                    'width': 0.9
                },
                'xref': 'paper',
                'yref': 'paper'
            }
        ],
        'annotations': [
            {
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.45,
                'text': "Current = "+txt2+" | Historic = "+txt,
                'showarrow': False
            }
        ]
    }

    # we don't want the boundary now
    base_chart['marker']['line']['width'] = 0

    fig = {"data": [base_chart, meter_chart],
           "layout": layout}
    return fig


def axis_change(x,y):
    x = x*41/50
    x = x+0.5
    y = y+0.5
    return [str(x),str(y)]


def polar_axis(theta, r):
    theta_pi = theta/360*2*np.pi
    x = np.cos(theta_pi)*r
    y = np.sin(theta_pi)*r
    return axis_change(x,y)


def get_shape(a):
    theta_1 = 225 - a/15*270
    theta_left = 315 - a/15*270
    theta_right = 135 - a/15*270
    points = polar_axis(theta_left,0.01)+polar_axis(theta_1, 0.3)+polar_axis(theta_right, 0.01)
    string = 'M '+' '.join(points[:2])+' L '+' '.join(points[2:4])+' L '+' '.join(points[4:6])+' Z'
    return string

######################################## page layout setting ###########################################################
server = Flask(__name__)
app = dash.Dash(__name__, server=server)
server.secret_key = os.environ.get('SECRET_KEY', 'my-secret-key')
app.scripts.config.serve_locally = True
app.config['suppress_callback_exceptions']=True
colors = {
    'background': '#ffffff',
    'text': '#7FDBFF'
}
###webpage layout####

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


index_page = html.Div(style={'backgroundColor': colors['background'],
                             'font-family': 'Arial',
                             'textAlign': 'left'},
                      children=[
    html.H1(
        children='Data Analysis',
        style={
            'textAlign': 'center',
            'borderBottom': 'thin lightgrey solid'
        }
    ),
    #dropdwon menu
    html.Label('Neighborhoods'),
    dcc.Dropdown(
        id = 'neighborhood-sel',
        options=[{'label': i, 'value': i} for i in get_neighborhood_list()],
        multi=True,
        value=None
    ),



    html.Div([
    html.P('Beds'),
    dcc.Checklist(
        id = 'beds-checklist',
        options=[
            {'label': '0', 'value': 0},
            {'label': '1', 'value': 1},
            {'label': '2', 'value': 2},
            {'label': '3', 'value': 3},
            {'label': '4', 'value': 4},
            {'label': '5+', 'value': 5}
        ],
        value=[1,2]#default check
    )]),

    html.Div([
    html.P('Types'),
    dcc.Checklist(
        id = 'types-checklist',
        options=[
            {'label': 'Condo', 'value': 'CONDO'},
            {'label': 'Coop', 'value': 'COOP'},
            {'label': 'SingleFamily', 'value':'SFAM'},
            {'label': 'MultiFamily', 'value':'MULTI'},
        ],
        value=['CONDO','COOP']
    )]),

    html.Hr(style={'opacity': 0}),
    html.Hr(style={'opacity': 0}),
    html.P('Year'),
    dcc.RangeSlider(
            id = 'year-slider',
            marks={i: '{}'.format(i) for i in range(2009, datetime.datetime.now().year+1,1)}, #label on the slider
            min=2008,
            max=2021,
            value=[datetime.datetime.now().year-1, datetime.datetime.now().year], #value of the slider
            # step=None,
            #style={'width': 100, 'height': 10}
    ),

    html.Hr(style={'opacity': 0}),
    html.Hr(style={'opacity': 0}),
    html.P('Closing Price'),
    dcc.RangeSlider(
            id = 'closingprice-slider',
            marks={i: '{}'.format(intWithCommas(i)) for i in range(0, 10000000, 1000000)},
            min=0,
            max= 10000000,
            value=[800000, 2000000],
            # step=None,
            #style={'width': 100, 'height': 10}
    ),



    html.Hr(style={'opacity': 0}),
    html.Hr(style={'opacity': 0}),
    html.P('Square Feet'),
    dcc.RangeSlider(
                id = 'sqft-slider',
                min=0,
                max=8000,
                value=[0, 3000],#max(df['SqFt'])
                marks={i: '{}SqFt'.format(intWithCommas_n(i)) for i in range(0, 8000, 500)},
                # step=None
                ),
    html.Hr(style={'opacity': 0}),
    html.Hr(style={'opacity': 0}),
    html.P('Dollar/SqFt'),
    dcc.RangeSlider(
                id = 'dollarsqft-slider',
                min=0,
                max=8000,
                value=[0, 3000],#max(df['SqFt'])
                marks={i: '{}'.format(intWithCommas(i)) for i in range(0, 8000, 500)},
                # step=None
                ),
    html.Hr(style={'opacity': 0}),
    html.Hr(style={'opacity': 0}),

    html.Button('Update', id='button-3'),

    html.Div([

    html.Table(id='table'),

    ]),


    dcc.Graph(
            id='avg-dollar-sqft',
            #animate = True
    ),

    html.Br(),
    dcc.Link('Chart_1', href='/page-1'),
    html.P('Output'),
    dcc.RadioItems(
        id = 'toggle',
        options=[
            {'label': 'AverageDollarSqFt', 'value': 1},
            {'label': 'ClosingPrice', 'value': 2}
        ],
        value=1
    ),
    html.Div([dcc.Graph(id='cluster-graph')],
             style={'width': '64%', 'display': 'inline-block','vertical-align':'top', 'padding': '0 20'}),
    html.Div([
        dcc.Markdown('"" # change (' to (" and ') to ")
                **Similar Neighborhoods**
            ""'.replace('   ', '')),
        html.Div(id='hover-data')],
        style={'display': 'inline-block','width': '34%','vertical-align':'top'}),

    html.Button('Click Me', id='button-1'),

    html.Div([dcc.Graph(id='map-graph')]),
    html.Div([dcc.Graph(id='meter-chart-1')],
             style={'width': '34%', 'display': 'inline-block',
                    'vertical-align':'top', 'padding': '0 20'})
    ])

layout_page_1 = html.Div( children=[
    html.Button('Click Me', id='button-2'),
    html.H1('Page-1'),

    dcc.Graph(id='chart-1',animate = True),
    html.Br(),
    dcc.Link('Home', href='/')]
)


###################################### call back functions for interacts ###################
# update the scatter chart
@app.callback(
    Output('avg-dollar-sqft', 'figure'),
    [Input('button-3','n_clicks')],
     [State('sqft-slider', 'value'),
     State('neighborhood-sel', 'value'),
     State('closingprice-slider', 'value'),
     State('beds-checklist', 'value'),
     State('toggle', 'value'),
     State('dollarsqft-slider','value'),
     State('types-checklist','value'),
     State('year-slider','value')
     ])
def update_figure(n_clicks,selected_sqft, selected_neighborhood, selected_cp, selected_beds, selected_tog, selected_ds,selected_types,selected_year):
    print("update_figure()")

    string=("SELECT t1.OriginalPrice, t3.SqFt, t3.Bed, t2.Neighborhood, t1.SellersAgent, t1.ListedDate, t2.FormattedAddress, t3.Type, t1.DollarSqft, t1.ClosedDate, t1.ClosedPrice, t2.Lat, t2.Lng, t1.SellersAgent, t1.SellersCompany"
       " FROM ta_Transaction t1 "
       " LEFT JOIN ta_BuildingAddress t2 ON t1.BuildingAddressId=t2.Id"
       " LEFT JOIN ta_Unit t3 ON t1.UnitId=t3.Id"
       " WHERE t2.Neighborhood IS NOT NULL")
    sql_str = string + " AND t2.Neighborhood in ("
    for neighborhood in selected_neighborhood:
        sql_str += "'%s'," % neighborhood
    sql_str = sql_str[:-1] + ')'

    df_input = pd.read_sql(sql_str,con)
    df_input.rename(
        columns={'ClosedDate': 'ClosingDate', 'DollarSqft': 'DollarSqFt', 'Bed': 'Beds', 'FormattedAddress': 'Address',
                 'ClosedPrice': 'ClosingPrice'}, inplace=True)
    df_input['DollarSqFt'] = df_input['ClosingPrice'].astype(float)/df_input['SqFt'].astype(float)
    #make df availavble for all functions
    global df
    df = df_input
    df=df[df["Lat"].notnull()]
    df['ClosingDate'] = pd.to_datetime(df['ClosingDate'], errors = 'coerce')
    df['ListedDate'] = pd.to_datetime(df['ListedDate'], errors = 'coerce')
    df['Type']=df['Type'].replace('CONDP','COOP' )
    transactions = ((df['ClosingDate']>=datetime.datetime.strptime("%s-01-01"%selected_year[0],'%Y-%m-%d')) & (df['ClosingDate']<=datetime.datetime.strptime("%s-12-31"%selected_year[1],'%Y-%m-%d'))) & (df['ClosingDate'].notnull())
    df = df.loc[transactions]
    print("data size",df.shape[0])
    if selected_cp[1] > 9500000:
        selected_cp[1] = df['ClosingPrice'].max()+1

    if selected_ds[1] > 7500:
        selected_ds[1] = df['DollarSqFt'].max()+1
    if selected_sqft[1] > 7500:
        selected_sqft[1] = df['SqFt'].max()

    ############### beds additional ################
    extra_beds = []
    for i in range(6, 101):
        extra_beds.append(i)
    if 5 in selected_beds:
        selected_beds = selected_beds+extra_beds+[np.NaN]
    ################################################

    global sbed,scp,ssqft,sdsqft,st,sn,stog,sy
    sbed= selected_beds
    scp = selected_cp
    ssqft = selected_sqft
    sdsqft = selected_ds
    st = selected_types
    sn = selected_neighborhood
    stog =selected_tog
    sy=selected_year


    print('beds: ',selected_beds)
    print('sqft: ',selected_sqft)
    print('neighbor: ',selected_neighborhood)
    print('cp: ',selected_cp)
    print('tog: ',selected_tog)
    print('ds: ',selected_ds)
    print('types: ',selected_types)
    print('sy:',selected_year)
    
    #filter the df by user selection
    beds_labels = mult_selections(selected_beds, 'Beds', df)
    closing_price_labels = (df['ClosingPrice'] < selected_cp[1]) & (df['ClosingPrice'] > selected_cp[0])
    sqft_labels = (df['SqFt'] < selected_sqft[1]) & (df['SqFt'] > selected_sqft[0])
    dollarsqft_label = (df['DollarSqFt'] < selected_ds[1]) & (df['DollarSqFt'] > selected_ds[0])
    types_labels = mult_selections(np.array(selected_types), 'Type', df)

    # generate the df_clusters, for bar plot
    # global  labels_1
    # labels_1 = beds_labels & closing_price_labels & sqft_labels & dollarsqft_label & types_labels
    # df_output = df.loc[labels_1][['DollarSqFt', 'ClosingPrice', 'ClosingDate', 'Neighborhood']]
    # df_output.index = df_output['ClosingDate']
    # del df_output['ClosingDate']
    #
    # df_list = []
    # for i in df_output['Neighborhood'].unique():
    #     df1 = df_output[df_output['Neighborhood'] == i]
    #     se_tmp = df1.resample('A').mean()
    #     se_tmp['Neighborhood'] = i
    #     #         se_tmp['Num'] = len(df1)
    #     l = []
    #     for j in se_tmp.index.year:
    #         if math.isnan(se_tmp[str(j)]['DollarSqFt'][0]) == False:
    #             l.append(len(df1[str(j)]))
    #         else:
    #             l.append(0)
    #     se_tmp['Num'] = l
    #     df_list.append(se_tmp)
    #
    # global df_clusters
    # df_clusters = pd.concat(df_list)

    # generate data for plot line
    traces = []
    for i in range(len(selected_neighborhood)):
        neighborhood_labels = mult_selections([selected_neighborhood[i]], 'Neighborhood', df)
        labels = beds_labels & neighborhood_labels & closing_price_labels & sqft_labels & dollarsqft_label & types_labels

        df_output = df.loc[labels][['DollarSqFt', 'ClosingPrice', 'ClosingDate']]
        df_output.index = df_output['ClosingDate']
        del df_output['ClosingDate']
        df_output = df_output.resample('Q').mean()

        if selected_tog == 1:
            y1 = df_output['DollarSqFt'].interpolate()
            y1axis = {'title': 'Avg$/SqFt'}
        else:
            y1 = df_output['ClosingPrice'].interpolate()
            y1axis = {'title': 'ClosingPrice'}

        strtdate = {'1': '-03-31', '2': '-06-30', '3': '-09-30', '4': '-12-31'}
        x = [str(i) for i in list(y1.index.year.astype(str) + [strtdate[j] for j in y1.index.quarter.astype(str)])]
        print(x)
        traces.append(go.Scatter(
            customdata=[selected_neighborhood[i] for j in range(len(y1))],
            x=[str(i) for i in list(y1.index.year.astype(str) + [strtdate[j] for j in y1.index.quarter.astype(str)])],
            y=y1.values,
            text=[k for k in list(y1.index.year.astype(str) + [strtdate[j] for j in y1.index.quarter.astype(str)])],
            mode='lines',
            opacity=0.7,
            marker={
                'size': 15,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name=selected_neighborhood[i]
        ))
    return {
        'data': traces,
        'layout': go.Layout(
            xaxis={'title': 'Time'},
            yaxis= y1axis)

    }

# meter-chart-1, calculate the historical absorption rate
@app.callback(
    Output('meter-chart-1', 'figure'),
    [Input('sqft-slider', 'value'),
     Input('neighborhood-sel', 'value'),
     Input('closingprice-slider', 'value'),
     Input('beds-checklist', 'value'),
     Input('dollarsqft-slider','value'),
     Input('types-checklist','value'),
     Input('avg-dollar-sqft', 'clickData')])
def absortion_ratio_ave(selected_sqft, selected_neighborhood,
                    selected_cp, selected_beds, selected_ds,
                    selected_types, hoverData):
    print("absortion_ratio_ave()")
    if selected_cp[1] > 9500000:
        selected_cp[1] = df['ClosingPrice'].max()+1

    if selected_ds[1] > 7500:
        selected_ds[1] = df['DollarSqFt'].max()+1
    if selected_sqft[1] > 7500:
        selected_sqft[1] = df['SqFt'].max()

    ############### beds additional ################
    extra_beds = []
    for i in range(6, 101):
        extra_beds.append(i)
    if 5 in selected_beds:
        selected_beds = selected_beds+extra_beds+[np.NaN]
    ################################################

    beds_labels = mult_selections(selected_beds, 'Beds', df)
    closing_price_labels = (df['ClosingPrice'] < selected_cp[1]) & (df['ClosingPrice'] > selected_cp[0]) | (df['ClosingPrice'].isnull())
    sqft_labels = (df['SqFt'] < selected_sqft[1]) & (df['SqFt'] > selected_sqft[0]) | (df['SqFt'].isnull())
    dollarsqft_label = (df['DollarSqFt'] < selected_ds[1]) & (df['DollarSqFt'] > selected_ds[0]) | (df['DollarSqFt'].isnull())
    types_labels = mult_selections(np.array(selected_types), 'Type', df)
    neighborhood_labels = mult_selections(selected_neighborhood, 'Neighborhood', df)




    labels_2 = beds_labels & closing_price_labels & sqft_labels \
               & dollarsqft_label & neighborhood_labels & types_labels

    df1 = df.loc[labels_2]

    print("all beds for absorption rate = " + str(selected_beds))
    print("max Beds = " + str(df1['Beds'].max()))

    date = hoverData['points'][0]['x']

    date_y = pd.to_datetime(date)
    a_sum = 0
    for i in range(3):
        date_y -= datetime.timedelta(365)
        start = date_y - pd.tseries.offsets.MonthEnd(3)
        end = date_y + pd.tseries.offsets.MonthEnd()

        sales_labels = df1['ClosingDate'].between(start, end)
        # inventories_labels = (df1['ClosingDate'] > end) & (df1['ListedDate'] <= end)
        inventories_labels = ((df1['ClosingDate'].isnull()) | (df1['ClosingDate'] > end)) & (df1['ListedDate'] <= end)
        a_sum += sum(inventories_labels) / sum(sales_labels)
    a = a_sum / 3
    dateCurrent = hoverData['points'][0]['x']
    dateCurrent_y = pd.to_datetime("2018-07-11")
    startCurrent = dateCurrent_y - pd.tseries.offsets.MonthEnd(3)
    endCurrent = dateCurrent_y + pd.tseries.offsets.MonthEnd()

    sales_labels = df1['ClosingDate'].between(startCurrent, endCurrent)
    inventories_labels = ((df1['ClosingDate'].isnull()) | (df1['ClosingDate'] > endCurrent)) & ( df1['ListedDate'] <= endCurrent)
    print("sales_labels for current : " + str(sum(sales_labels)))
    print("inventories_labels for current : " + str(sum(inventories_labels)))
    b = sum(inventories_labels) / sum(sales_labels)
    print("Current Absorption rate : " + str(b))

    return get_meter_chart(a=a, b=b, title='Absorption rate current vs Historic')
  
#function for the scatter heat map
@app.callback(
    Output('map-graph', 'figure'),
    [Input('button-1','n_clicks')],
    [State('sqft-slider', 'value'),
     State('avg-dollar-sqft', 'clickData'),
     State('closingprice-slider', 'value'),
     State('beds-checklist', 'value'),
     State('dollarsqft-slider','value'),
     State('types-checklist','value'),
      State('toggle', 'value')
    ])
def update_map(n_clicks,selected_sqft, hoverData, selected_cp,
               selected_beds, selected_ds, selected_types, toggle):
    print("update_map()")
    column = {2: 'ClosingPrice', 1: 'DollarSqFt'}[toggle]

    date = hoverData['points'][0]['x']
    date_y = pd.to_datetime(date)
    start = date_y - datetime.timedelta(365)


    year_labels = df['ClosingDate'].between(start, date_y)

    if selected_cp[1] > 9500000:
        selected_cp[1] = df['ClosingPrice'].max()+1
    if selected_ds[1] > 7500:
        selected_ds[1] = df['DollarSqFt'].max()+1
    if selected_sqft[1] > 7500:
        selected_sqft[1] = df['SqFt'].max()

    ############### beds additional ################
    extra_beds = []
    for i in range(6, 101):
        extra_beds.append(i)
    if 5 in selected_beds:
        selected_beds = selected_beds + extra_beds+[np.NaN]
    ################################################
    df['SqFt']= df['SqFt'].fillna(0)
    df['ClosingPrice'] = df['ClosingPrice'].fillna(0)
    df['DollarSqFt'] = df['DollarSqFt'].fillna(0)
    beds_labels = mult_selections(selected_beds, 'Beds', df)
    closing_price_labels = (df['ClosingPrice'] < selected_cp[1]) & (df['ClosingPrice'] > selected_cp[0])| (df['ClosingPrice']==0)
    sqft_labels = (df['SqFt'] < selected_sqft[1]) & (df['SqFt'] > selected_sqft[0])| (df['SqFt']==0)
    dollarsqft_label = (df['DollarSqFt'] < selected_ds[1]) & (df['DollarSqFt'] > selected_ds[0])| (df['DollarSqFt']==0)
    types_labels = mult_selections(np.array(selected_types), 'Type', df)


    labels_3 = beds_labels & closing_price_labels & sqft_labels & dollarsqft_label & types_labels &year_labels
    #print("labels_1 = " +str(labels_1))
    #print("column = " + str(column))
    df_output = df.loc[labels_3][['Lat','Lng','SellersAgent','Address',column]]

    c_min = df_output[column].min()
    print(column+"_min = "+str(c_min))
    c_range = df_output[column].max() - c_min
    print(column+"_range = "+str(c_range))

    df_output['color'] = df_output[column].apply(gen_color, args=[c_range,c_min])

    t1 = df_output[column].apply(intWithCommas).values
    t2 = df_output['Address'].values
    #
    # print(t1[0])
    # print(t2[0])

    txt = []
    for i in range(len(t1)):
        txt.append(t1[i]+' Address:'+t2[i])


    data = go.Data([
        go.Scattermapbox(
            lat=df_output['Lat'],
            lon=df_output['Lng'],
            mode='markers',
            marker=go.Marker(
                size=7,
                opacity=0.6,
                color=df_output['color']
            ),
            text=txt,
        )
    ])
    layout = go.Layout(
        autosize=False,
        width=1300,
        height=800,
        hovermode='closest',
        mapbox=dict(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=dict(
                lat=40.717086,
                lon=-73.991040
            ),
            pitch=0,
            zoom=10,
            style='dark'
        ),
    )


    return dict(data=data, layout=layout)



#update the scatter chart in new page, which shows different neighborhoods.


@app.callback(
    Output('chart-1','figure'),
    [Input('button-2','n_clicks')]
)
def update_chart_1(nclicks):
    print("update_chart_1()")
    return update_plot_beds(selected_sqft=ssqft,selected_neighborhood=sn,
                            selected_cp=scp,selected_beds=sbed,selected_tog=stog,
                            selected_ds=sdsqft,selected_types = st,dataframe=df)




# update the cluster bar chart
@app.callback(
    Output('cluster-graph', 'figure'),
    [Input('avg-dollar-sqft', 'clickData'),
     Input('toggle', 'value'),
     # Input('sqft-slider', 'value'),
     # Input('closingprice-slider', 'value'),
     # Input('beds-checklist', 'value'),
     # Input('dollarsqft-slider', 'value'),
     # Input('types-checklist', 'value')
])
def cluster_figure(hoverData, toggle):#,selected_sqft, selected_cp, selected_beds, selected_ds,selected_types):
    print("cluster_figure()")
    neighbor = [hoverData['points'][i]['customdata'] for i in range(len(hoverData['points']))]
    date = hoverData['points'][0]['x']
    column = {2:'ClosingPrice', 1:'DollarSqFt'}[toggle]
    start = datetime.datetime.strptime(date, '%Y-%m-%d')- datetime.timedelta(365)
    sql_str = (
            "SELECT t1.OriginalPrice, t3.SqFt, t3.Bed, t2.Neighborhood, t1.SellersAgent, t1.ListedDate, t2.FormattedAddress, t3.Type, t1.DollarSqft, t1.ClosedDate, t1.ClosedPrice, t2.Lat, t2.Lng, t1.SellersAgent, t1.SellersCompany"
            " FROM ta_Transaction t1 "
            " LEFT JOIN ta_BuildingAddress t2 ON t1.BuildingAddressId=t2.Id"
            " LEFT JOIN ta_Unit t3 ON t1.UnitId=t3.Id"
            " WHERE t2.Neighborhood IS NOT NULL AND t1.ClosedDate BETWEEN  '%s' AND '%s'" % (
            start, date)
    )#### select data from all negihborhood within specified date range ###

    df_all=pd.read_sql(sql_str,con)
    con.commit()
    df_all.rename(
        columns={'ClosedDate': 'ClosingDate', 'DollarSqft': 'DollarSqFt', 'Bed': 'Beds', 'FormattedAddress': 'Address',
                 'ClosedPrice': 'ClosingPrice'}, inplace=True)
    df_all['DollarSqFt'] = df_all['ClosingPrice'].astype(float)/df_all['SqFt'].astype(float)
    df_all=df_all[df_all["Lat"].notnull()]
    df_all['ClosingDate'] = pd.to_datetime(df_all['ClosingDate'], errors = 'coerce')
    df_all['ListedDate'] = pd.to_datetime(df_all['ListedDate'], errors = 'coerce')
    df_all['Type']=df_all['Type'].replace('CONDP','COOP' )

    beds_labels = mult_selections(sbed, 'Beds', df_all)
    closing_price_labels = (df_all['ClosingPrice'] < scp[1]) & (df_all['ClosingPrice'] > scp[0])
    sqft_labels = (df_all['SqFt'] < ssqft[1]) & (df_all['SqFt'] > ssqft[0])
    dollarsqft_label = (df_all['DollarSqFt'] < sdsqft[1]) & (df_all['DollarSqFt'] > sdsqft[0])
    types_labels = mult_selections(np.array(st), 'Type', df_all)

    # generate the df_clusters, for bar plot
    labels_1 = beds_labels & closing_price_labels & sqft_labels & dollarsqft_label & types_labels

    df_output = df_all.loc[labels_1][['DollarSqFt', 'ClosingPrice', 'ClosingDate', 'Neighborhood']]
    df_output.index = df_output['ClosingDate']
    del df_output['ClosingDate']

    df_list = []
    for i in df_output['Neighborhood'].unique():
        df1 = df_output[df_output['Neighborhood'] == i]
        se_tmp = df1.resample('A').mean()
        se_tmp['Neighborhood'] = i
        #         se_tmp['Num'] = len(df1)
        l = []
        for j in se_tmp.index.year:
            if math.isnan(se_tmp[str(j)]['DollarSqFt'][0]) == False:
                l.append(len(df1[str(j)]))
            else:
                l.append(0)
        se_tmp['Num'] = l
        df_list.append(se_tmp)
    global df_clusters
    df_clusters = pd.concat(df_list)
    return get_average(df_clusters, column, neighbor)


# update the list of neighborhood when hovering cluster.
@app.callback(
    Output('hover-data', 'children'),
    [Input('cluster-graph', 'hoverData'),
     Input('toggle', 'value')])
def get_cluster_info(hoverData, toggle):
    print("get_cluster_info()")
    i = hoverData['points'][0]['x']
    txt = df_view.loc[i]['markers']
    column = {2: 'ClosingPrice', 1: 'DollarSqFt'}[toggle]
    print(i)
    return df_ftxt(txt, column)




# Update the index
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return layout_page_1
    else:
        return index_page

from flask import request
"""
from flask import Flask, jsonify
from flask import request
app = Flask(__name__)
@app.route('/v1.0/caller', methods=['POST'])
def create_task():
    if not request.json or not 'title' in request.json:
        abort(400)
    task = {
        'data': 'Empty',
    }
    tasks.append(task)
    return jsonify({'task': task}), 201


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
