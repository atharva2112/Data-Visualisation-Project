#%%
#%%
import dash
import dash_html_components as html
import numpy as np
from dash import dcc, dash_table
from dash.dependencies import Input, Output,State
import plotly.express as px
import pandas as pd
import math
from scipy.fft import fft
from scipy.special import expit
from normal_test import shapiro_test,ks_test,da_k_squared_test
from datetime import date
from datetime import datetime
import base64
import io
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
#%%
df = pd.read_csv("/Users/atharvah/GWU/Sem 3 /Data Visualisation/Final Project/archive/fraudTrain.csv") # reading the train data
df = df.set_index(df.trans_date_trans_time)
df = df.drop(columns=["Unnamed: 0","trans_date_trans_time","cc_num","first","last","street","lat","long","dob","unix_time","merch_lat","merch_long"])
df2 = pd.read_csv("/Users/atharvah/GWU/Sem 3 /Data Visualisation/Final Project/archive/fraudTest.csv") # reading the test data
df2 = df2.set_index(df2.trans_date_trans_time)
df2 = df2.drop(columns=["Unnamed: 0","trans_date_trans_time","cc_num","first","last","street","lat","long","dob","unix_time","merch_lat","merch_long"])
fraud = df2[df2.is_fraud==1] # extracting all the 'fraud' transactions from the test dataset
fraud = fraud.append(df[df.is_fraud==1]) # combining all the fraud transactions from the test and the train dataset.
not_fraud = df[df.is_fraud==0] # Extracting all the "not fraud" data.
df_final = fraud
df_final = df_final.append(not_fraud)

#%%
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

my_app = dash.Dash('FTP', external_stylesheets=external_stylesheets,suppress_callback_exceptions=True)
my_app.layout = html.Div([html.H1('Final Project', style={'textAlign': 'center'}),
                          html.Br(),
                          dcc.Tabs(id='hw-questions',
                          children=[dcc.Tab(label="File Details",value="det"),
                              dcc.Tab(label="About the Data",value="ad"),
                                    dcc.Tab(label="Pre-processing",value="pp"),
                                    dcc.Tab(label="Transformation",value="transformed"),
                                    dcc.Tab(label="PCA",value="pca"),
                                    dcc.Tab(label='Normality test', value='nt'),
                                    dcc.Tab(label="Correlation Coefficient with Heatmap",value="hm"),
                                    dcc.Tab(label='Visualisations', value='viz')


                          ]),
                          html.Div(id='layout')])
#%%
file = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-data-upload'),
])

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        dash_table.DataTable(
            df.to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns]
        ),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])

@my_app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children
#%%
# about the data
fig1 = px.histogram(data_frame=df_final,
                   x ="amt",
                    title="Histogram for the transaction amounts")
fig2 = px.histogram(data_frame=df_final,
                   x ="city_pop",
                    title="Histogram for the city populations")
fig3 = px.pie(data_frame=df_final,names="is_fraud",title="Pie chart for fraud and not fraud")
dd = df_final.describe()
dd.insert(loc = 0,
          column = 'index',
          value = dd.index)
data = html.Div([html.H1("About the data"),
                 html.Br(),
                 html.Br(),
                 html.H2("Select an option"),
                 dcc.Dropdown(id = "info",
                              options=[{'label':"Data repetition","value":'rep'},
                                       {"label":"Missing Values",'value':'mis'}]),
                 html.Br(),
                 html.Br(),
                 html.Div(id = "data_out"),
                 html.H2("Description of the data"),
                 dash_table.DataTable(dd.to_dict('records'), [{"name": i, "id": i} for i in dd.columns]),
                 html.Br(),
                 html.Br(),
                 html.H2("First 10 columns of the data"),
                 dash_table.DataTable(df_final[:10].to_dict('records'), [{"name": i, "id": i} for i in df.columns]),
                 html.Br(),
                 html.Br(),
                 dcc.Graph(id = "amt", figure=fig1),
                 html.Br(),
                 html.Br(),
                 dcc.Graph(id = "city", figure=fig2),
                 html.Br(),
                 html.Br(),
                 dcc.Graph(id = "pie", figure=fig3)
                 ])

@my_app.callback([Output(component_id='data_out',component_property='children')],
                 Input(component_id="info",component_property="value"))

def data_update(a1):
    if a1 == "rep":
        return [html.Div([f"All the entries in the dataset are unique: {len(df_final) == len(set(df_final.trans_num))}"])]
    elif a1=="mis":
        return [html.Div([f"The number of missing values in the dataset: {df.isna().sum().sum()}"])]
#%%
#%%
# trans = html.Div([html.H1('Before and after z-transform'),
#                   html.Br(),
#                   html.Br(),
#                   dcc.Graph(id="trans1")
#                   ])
#
#
# def z_transform(df):
#     mean = np.mean(df)
#     std = np.std(df)
#     trans = (df - mean) / std
#     return trans
# @my_app.callback([Output("trans","figure")])
#
# def update():
#     df = pd.read_csv(
#         "/Users/atharvah/GWU/Sem 3 /Data Visualisation/Final Project/archive/fraudTrain.csv")  # reading the train data
#     df = df.set_index(df.trans_date_trans_time)
#     df = df.drop(
#         columns=["Unnamed: 0", "trans_date_trans_time", "cc_num", "first", "last", "street", "lat", "long", "dob",
#                  "unix_time", "merch_lat", "merch_long"])
#     df2 = pd.read_csv(
#         "/Users/atharvah/GWU/Sem 3 /Data Visualisation/Final Project/archive/fraudTest.csv")  # reading the test data
#     df2 = df2.set_index(df2.trans_date_trans_time)
#     df2 = df2.drop(
#         columns=["Unnamed: 0", "trans_date_trans_time", "cc_num", "first", "last", "street", "lat", "long", "dob",
#                  "unix_time", "merch_lat", "merch_long"])
#     fraud = df2[df2.is_fraud == 1]  # extracting all the 'fraud' transactions from the test dataset
#     fraud = fraud.append(
#         df[df.is_fraud == 1])  # combining all the fraud transactions from the test and the train dataset.
#     not_fraud = df[df.is_fraud == 0]  # Extracting all the "not fraud" data.
#     df_final = fraud
#     df_final = df_final.append(not_fraud)
#
#
#     transformed_data = z_transform(df_final[["amt", "city_pop"]])
#
#     figure1 = px.line(data_frame= transformed_data, x=["amt","city_pop"])
#     return [html.Div([figure1])]
#%%
tr = html.Div([html.H1("Before and after z-transform"),
               html.Br(),
               html.Br(),
               html.H2("Please select the following options"),
               dcc.RadioItems(id = "select",
                              options=[{"label":"Before Z-transform","value":"bf"},
                                       {"label": "After Z- transform","value":"af"}]
                              ),
               html.Br(),
               html.Br(),
               dcc.Graph("graph1")
               ])


@my_app.callback([Output(component_id="graph1",component_property="figure")],
                 [Input(component_id="select",component_property="value")])

def tr_update(a1):
    def z_transform(df):
        mean = np.mean(df)
        std = np.std(df)
        trans = (df - mean) / std
        return trans

    df = pd.read_csv(
        "/Users/atharvah/GWU/Sem 3 /Data Visualisation/Final Project/archive/fraudTrain.csv")  # reading the train data
    df = df.set_index(df.trans_date_trans_time)
    df = df.drop(
        columns=["Unnamed: 0", "trans_date_trans_time", "cc_num", "first", "last", "street", "lat", "long", "dob",
                 "unix_time", "merch_lat", "merch_long"])
    df2 = pd.read_csv(
        "/Users/atharvah/GWU/Sem 3 /Data Visualisation/Final Project/archive/fraudTest.csv")  # reading the test data
    df2 = df2.set_index(df2.trans_date_trans_time)
    df2 = df2.drop(
        columns=["Unnamed: 0", "trans_date_trans_time", "cc_num", "first", "last", "street", "lat", "long", "dob",
                 "unix_time", "merch_lat", "merch_long"])
    fraud = df2[df2.is_fraud == 1]  # extracting all the 'fraud' transactions from the test dataset
    fraud = fraud.append(
        df[df.is_fraud == 1])  # combining all the fraud transactions from the test and the train dataset.
    not_fraud = df[df.is_fraud == 0]  # Extracting all the "not fraud" data.
    df_final = fraud
    df_final = df_final.append(not_fraud)

    if a1=="bf":
        fig = px.line(data_frame=df_final,y=["amt","city_pop"],title="Transaction vs Population of the city")
        return [go.Figure(data = fig)]
    elif a1 == "af":
        transformed_data = z_transform(df_final[["amt", "city_pop"]])
        fig = px.line(data_frame=transformed_data,y=["amt","city_pop"],title="Transaction vs Population of the city after transformation")
        return [go.Figure(data = fig)]
#%%
# pre-processing
pre_process = html.Div([html.H1("Outlier Removal"),
                        html.H2("Select Boxplot before or after outlier removal"),
                        html.Br(),
                        html.Br(),
                        dcc.RadioItems(id = 'input-radio-button',
                                      options = [dict(label = 'Before Outlier Removal', value = 'bf'),
                                                 dict(label = 'After Outlier Removal', value = 'af')],
                                      value = 'bf'),
                       dcc.Graph(id = "box"),
                        html.Br(),
                        html.Br(),
                        dcc.Graph(id = "box2")])


@my_app.callback([Output('box', 'figure'),
                  Output("box2","figure")],
              [Input('input-radio-button', 'value')])
def update_graph(value):
    df = pd.read_csv(
        "/Users/atharvah/GWU/Sem 3 /Data Visualisation/Final Project/archive/fraudTrain.csv")  # reading the train data
    df = df.set_index(df.trans_date_trans_time)
    df = df.drop(
        columns=["Unnamed: 0", "trans_date_trans_time", "cc_num", "first", "last", "street", "lat", "long", "dob",
                 "unix_time", "merch_lat", "merch_long"])
    df2 = pd.read_csv(
        "/Users/atharvah/GWU/Sem 3 /Data Visualisation/Final Project/archive/fraudTest.csv")  # reading the test data
    df2 = df2.set_index(df2.trans_date_trans_time)
    df2 = df2.drop(
        columns=["Unnamed: 0", "trans_date_trans_time", "cc_num", "first", "last", "street", "lat", "long", "dob",
                 "unix_time", "merch_lat", "merch_long"])
    fraud = df2[df2.is_fraud == 1]  # extracting all the 'fraud' transactions from the test dataset
    fraud = fraud.append(
        df[df.is_fraud == 1])  # combining all the fraud transactions from the test and the train dataset.
    not_fraud = df[df.is_fraud == 0]  # Extracting all the "not fraud" data.
    df_final = fraud
    df_final = df_final.append(not_fraud)
    if value=="bf":
        fig1=px.box(data_frame=df_final,x = "amt",title="Boxplot for transaction amount before Outlier removal")
        fig2 = px.box(data_frame=df_final, x="city_pop", title="Boxplot for city population before Outlier removal")
        return [fig1,fig2]
    elif value=="af":
        df_final = df_final[(df_final.amt < 170) & (df_final.amt > -94.77)]
        df_final = df_final[(df_final.city_pop < 3900) & (df_final.city_pop > -94.77)]
        fig1=px.box(data_frame=df_final,x = "amt",title="Boxplot for transaction amount after Outlier removal")
        fig2 = px.box(data_frame=df_final, x="city_pop", title="Boxplot for city population after Outlier removal")
        return [fig1,fig2]

#%%
scaler = StandardScaler()
scaled = scaler.fit_transform(df_final[["amt","city_pop"]])

pca = PCA(n_components="mle")
ScaledComponents = pca.fit_transform(scaled)

pca = html.Div([html.H1("Principle component analysis"),
                html.Br(),
                html.Br(),
                dcc.Slider(id ="slide",min = 0,max = len(df_final)),
                dcc.Graph("scaled"),
                html.Br(),
                html.Br(),
                dcc.Graph("scaled2"),
                html.Br(),
                html.Br()
                ])
df_pca = pd.DataFrame(ScaledComponents, columns=['Principal col %i' % i for i in range(ScaledComponents.shape[1])],
                          index=df_final.index)
@my_app.callback([Output("scaled","figure"),
                  Output("scaled2","figure")],
                 [Input("slide","value")])

def pca_update(a1):
    df = pd.read_csv(
        "/Users/atharvah/GWU/Sem 3 /Data Visualisation/Final Project/archive/fraudTrain.csv")  # reading the train data
    df = df.set_index(df.trans_date_trans_time)
    df = df.drop(
        columns=["Unnamed: 0", "trans_date_trans_time", "cc_num", "first", "last", "street", "lat", "long", "dob",
                 "unix_time", "merch_lat", "merch_long"])
    df2 = pd.read_csv(
        "/Users/atharvah/GWU/Sem 3 /Data Visualisation/Final Project/archive/fraudTest.csv")  # reading the test data
    df2 = df2.set_index(df2.trans_date_trans_time)
    df2 = df2.drop(
        columns=["Unnamed: 0", "trans_date_trans_time", "cc_num", "first", "last", "street", "lat", "long", "dob",
                 "unix_time", "merch_lat", "merch_long"])
    fraud = df2[df2.is_fraud == 1]  # extracting all the 'fraud' transactions from the test dataset
    fraud = fraud.append(
        df[df.is_fraud == 1])  # combining all the fraud transactions from the test and the train dataset.
    not_fraud = df[df.is_fraud == 0]  # Extracting all the "not fraud" data.
    df_final = fraud
    df_final = df_final.append(not_fraud)
    # df_final1 = df_final[:a1]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_final[["amt", "city_pop"]])

    pca = PCA(n_components="mle")
    ScaledComponents = pca.fit_transform(scaled)
    df_pca = pd.DataFrame(ScaledComponents, columns=['Principal col %i' % i for i in range(ScaledComponents.shape[1])],
                          index=df_final.index)
    fig1 = px.line(data_frame=scaled[:a1],y=[0,1],title="Scaled Data")
    fig2 = px.line(data_frame=df_pca[:a1], y="Principal col 0", title="PCA Data")
    return [fig1,fig2]



#%%
# # Question
nt = html.Div([
    html.H2("Select the predictor"),
    dcc.Dropdown(id="predictor",
               options = [
                   {'label': f'{x}' ,'value':f"{x}"}for x in ["city_pop","amt"]],placeholder = True),
    html.Br(),
    html.Br(),
    html.H2("Select the test"),
    dcc.Dropdown(id = "legend",
                 options=[{'label':"shapiro_test", 'value':"shapiro_test"},
                    {'label':"ks_test", 'value':"ks_test"},
                   {'label':"da_k_squared_test", 'value':"da_k_squared_test"}],placeholder = True),
    html.Br(),
    html.Br(),
    html.H2("Normality Test"),
    html.Div(id = "output1"),
    html.Br(),
    html.Br(),
    dcc.Graph(id = "amt", figure=fig1),
    html.Br(),
    html.Br(),
    dcc.Graph(id = "city", figure=fig2)
])

@my_app.callback(
    Output(component_id='output1',component_property='children'),
    [Input(component_id='predictor',component_property='value'),
     Input(component_id="legend",component_property="value")]
)

def update(a1,a2):
    df = df_final[a1]
    if a2 == "shapiro_test":
        a, b = shapiro_test(df,title=a1)
        return f"pvalue:{round(a,2)} statistics:{b}"
    elif a2 == "ks_test":
        a, b = ks_test(df,title=a1)
        return f"pvalue:{round(a,2)} statistics:{b}"
    elif a2 == "da_k_squared_test":
        a,b = da_k_squared_test(df, title=a1)
        return f"pvalue:{round(a,2)} statistics:{b}"
#%%
heat = html.Div([html.H1("Correlation coefficients for the features with Heatmap"),
                 html.H2("Select the features to be considered"),
                 dcc.Checklist(id = "heatm",
                               options=[{'label': f'{x}' ,'value':f"{x}"}for x in ["amt","zip","city_pop","is_fraud"]],inline=True),
                 html.Br(),
                 html.Br(),
                 dcc.Graph(id = "heatmap")
                 ])
@my_app.callback(Output(component_id="heatmap",component_property="figure"),
                 Input(component_id="heatm",component_property="value"))
def update_heat(a):
    df = df_final[a]
    df = df.corr()
    fig = px.imshow(df,title=f"Heatmap for {a}",text_auto=True)
    return fig

#%%
viz = html.Div([html.H1("Data Visualizations"),
                html.Br(),
                html.Br(),
                dcc.Input(id='input-on-submit', type='number'),
                html.Button('Submit', id='submit-val', n_clicks=0),
                html.Div(id='container-button-basic',
                         children='Enter a value and press submit'),
                html.Br(),
                html.Br(),
                dcc.Graph(id = "line"),
                html.Br(),
                html.Br(),
                dcc.Graph(id = "line1"),
                html.Br(),
                html.Br(),
                dcc.Graph(id = "hist"),
                html.Br(),
                html.Br(),
                dcc.Graph(id = "hist2"),
                html.Br(),
                html.Br(),
                dcc.Graph(id = "pie")
                ])
@my_app.callback([Output(component_id="line",component_property="figure"),
                  Output(component_id="line1",component_property="figure"),
                  Output(component_id="hist", component_property="figure"),
                  Output(component_id="hist2", component_property="figure"),
                  Output(component_id="pie", component_property="figure")
                  ],
                 [Input(component_id="input-on-submit",component_property="value")])
def viz_update(a1):
    df = pd.read_csv(
        "/Users/atharvah/GWU/Sem 3 /Data Visualisation/Final Project/archive/fraudTrain.csv")  # reading the train data
    df = df.set_index(df.trans_date_trans_time)
    df = df.drop(
        columns=["Unnamed: 0", "trans_date_trans_time", "cc_num", "first", "last", "street", "lat", "long", "dob",
                 "unix_time", "merch_lat", "merch_long"])
    df2 = pd.read_csv(
        "/Users/atharvah/GWU/Sem 3 /Data Visualisation/Final Project/archive/fraudTest.csv")  # reading the test data
    df2 = df2.set_index(df2.trans_date_trans_time)
    df2 = df2.drop(
        columns=["Unnamed: 0", "trans_date_trans_time", "cc_num", "first", "last", "street", "lat", "long", "dob",
                 "unix_time", "merch_lat", "merch_long"])
    fraud = df2[df2.is_fraud == 1]  # extracting all the 'fraud' transactions from the test dataset
    fraud = fraud.append(
        df[df.is_fraud == 1])  # combining all the fraud transactions from the test and the train dataset.
    not_fraud = df[df.is_fraud == 0]  # Extracting all the "not fraud" data.
    df_final = fraud
    df_final = df_final.append(not_fraud)
    df_final1 = df_final[:a1]
    fig1 = px.line(data_frame=df_final1,y = "amt",title="Line plot for transaction amounts")
    fig2 = px.line(data_frame=df_final1,y="city_pop", title="Line plot for city population")
    fig3 = px.histogram(data_frame=df_final1, x="amt", title="Histogram for transaction amounts")
    fig4 = px.histogram(data_frame=df_final1, x="city_pop", title="Histogram for city population")
    fig5 = px.pie(data_frame=df_final,names="gender",title="Gender pie plot")
    return [fig1,fig2,fig3,fig4,fig5]
#%%
@my_app.callback(Output(component_id='layout', component_property='children'),
                 [Input(component_id='hw-questions', component_property='value')])

def update_layout(ques):
    if ques=="det":
        return file
    elif ques == 'ad':
        return data
    elif ques == "pp":
        return pre_process
    elif ques == "transformed":
        return tr
    elif ques == "pca":
        return pca
    elif ques == 'nt':
        return nt
    elif ques=="hm":
        return heat
    elif ques == "viz":
        return viz


my_app.run_server(
    port=8033,
    host='0.0.0.0'
)
#%%
