# python 37
# elsa@promasta.com

import pandas as pd
import openpyxl
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import math, os, re
from scipy.stats import pareto, lognorm, gamma, binom, poisson, nbinom

# Formatting objects
def strip_blanks_punctuation_umlaute_to_lower(strInput):
    pattern = '[^A-Za-z0-9]'
    replace = ''
    return (re.sub(pattern, replace, strInput)).lower()
allowed_columns = {
        'id':'id',
        'number':'id',
        'name':'id',
        'policy':'id',
        'year':'year', 
        'anne':'year',
        'jahr':'year',
        'claims':'claim',
        'claim':'claim',
        'loss': 'claim',
        'losses':'claim',
        'sinistre':'claim',
        'sinistres':'claim',
        'schaden':'claim',
        'ultimate':'claim',
        'ultimates':'claim',
        'line':'line',
        'lob':'line', 
        'area':'line', 
        'region':'line', 
        'country':'line'
}

# Load data
df = pd.read_excel('https://raw.githubusercontent.com/elsaburren/claims-dashboard/main/claims.xlsx', sheet_name='claims', skiprows = 0, engine='openpyxl')

# Remove unnamed
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Rename column name and check data validity
df.rename(lambda x: allowed_columns[strip_blanks_punctuation_umlaute_to_lower(x)] if strip_blanks_punctuation_umlaute_to_lower(x) in list(allowed_columns.keys()) else x, axis='columns', inplace=True)
for x in set(allowed_columns.values()):
    if list(df.columns).count(x) > 1:
        print('! ERROR: claims-dashboard identified more than one column as ' + str(x))
        exit()
    if (x=='claim' or x=='year') and list(df.columns).count(x) != 1:
        print('! ERROR: claims-dashboard did not find a column ' + str(x))
        exit()
    if (x=='line') and list(df.columns).count(x) != 1:
        print('! Warning: claims-dashbord did not find a column line')
        df['line'] = 'all'
    if (x=='id') and list(df.columns).count(x) != 1:
        df['id'] = [str(x) for x in range(1, df.shape[0]+1, 1)]

# Set types
df = df.astype({'id' : str, 'year' : int, 'claim' : float, 'line':str})
print(df.dtypes)

# Define a count variable
df['count'] = 1

# Create necessary variables for the dashboard interface
year_min, year_max = df['year'].min(), df['year'].max()
years = [x for x in range(year_min, year_max+1)]
str_years = [str(x) for x in years]
format_frq = pd.DataFrame.from_dict(dict({'year':years, 'count':[0 for x in years]}))
claim_min, claim_max = df['claim'].min(), df['claim'].max()
ndigits = max(0, len(str(int(claim_min))) - 1)
threshold_min = int(claim_min / (10 ** ndigits)) * 10 ** ndigits
threshold_max = int(claim_max / (10 ** ndigits)) * 10 ** ndigits
step_size = max(int((math.ceil(threshold_max)-math.ceil(threshold_min)) / 10), 1)
threshold_options = [x for x in range(math.floor(threshold_min), math.ceil(threshold_max), step_size)]
line_options = [x for x in set(df['line'])]

# Create functions for fitting
def create_survival(df,str_single_col, obj_dist):
    df_out = df[[str_single_col]].value_counts(normalize=True).to_frame('prob').sort_values(by=str_single_col,ascending=True)
    df_out['sf'] = 1.0 - df_out['prob'].cumsum(skipna=False)
    df_out['sf_type'] = 'observed'
    df_out = df_out.drop(columns='prob').reset_index(level=str_single_col)
    df_theoretical = pd.DataFrame()
    df_theoretical[str_single_col] = df_out.loc[:,str_single_col]
    df_theoretical['sf'] = df_theoretical[str_single_col].apply(lambda x:obj_dist.sf(x))
    df_theoretical['sf_type'] = obj_dist.type()
    df_out = pd.concat([df_out, df_theoretical])

    return df_out

# Classes
class Frequency:
    def __init__(self, **kwargs):
        self._mean = kwargs['mean'] if 'mean' in kwargs else 0.0
        self._variance = kwargs['variance'] if 'variance' in kwargs else kwargs['mean'] if 'mean' in kwargs else 0.0
        self._type = 'nbinom' if self._variance > self._mean else 'binom' if self._variance < self._mean else 'poisson'
        
    def mean(self, v=None):
        if v: self._mean = v
        try: return self._mean
        except AttributeError: return None

    def variance(self, v=None):
        if v: self._variance = v
        try: return self._variance
        except AttributeError: return None
    
    def sd(self):
        try: return math.sqrt(self._variance)
        except AttributeError: return None
    
    def type(self):
        try: return self._type
        except AttributeError: return None
        
    def sf(self, k=0):
        if self._type == 'poisson':
            return poisson.sf(k, self._mean, loc=0)
        elif self._type == 'nbinom':
            n = self._mean * self._mean  / (self._variance - self._mean)
            p = self._mean / self._variance
            return nbinom.sf(k, n, p)
        else:
            n = self._mean / (1.0 - (self._variance / self._mean))
            p = 1.0 - self._variance / self._mean
            return binom.sf(k, n, p)
        
class Severity:
    def __init__(self, **kwargs):
        self._shape = kwargs['shape'] if 'shape' in kwargs else 1.0
        self._scale = kwargs['scale'] if 'scale' in kwargs else 1.0
        self._loc   = kwargs['loc']   if 'loc' in kwargs else 0.0
        self._type  = kwargs['type']  if 'type'  in kwargs else 'undefined'
        
    def shape(self, v=None):
        if v: self._shape = v
        try: return self._shape
        except AttributeError: return None

    def scale(self, v=None):
        if v: self._scale = v
        try: return self._scale
        except AttributeError: return None
    
    def loc(self, v=None):
        if v: self._loc = v
        try: return self._loc
        except AttributeError: return None

    def type(self, v=None):
        allowed_types = ('pareto', 'gamma', 'lognorm')
        if v in allowed_types: self._type = v
        try: return self._type
        except AttributeError: return None
        
    def sf(self, x):
        if self._type == 'pareto':
            return pareto.sf(x, self._shape, 0.0, self._scale)
        elif self._type == 'gamma':
            return gamma.sf(x, self._shape, self._loc, self._scale)
        elif self._type == 'lognorm':
            return lognorm.sf(x, self._shape, self._loc, 1.0)
        else:
            return 'undefined'
    
# App Layout
app = dash.Dash(__name__, meta_tags=[{'name':'viewport', 'content':'width=device-width, initial-scale=1.0'}])

server = app.server

app.layout = html.Div([
    html.H1('Insurance Claims Dashboard', style={'text-align': 'left'}),
    
    html.Table([
        html.Div('Select Claims Threshold: ', style={'float':'left', 'padding': '5px 5px 5px 3px', 'border': '1px solid rgb(200,200,200)', 'width':'350px', 'height':'24px'}), dcc.Dropdown(
        id='slct_threshold', 
        options=[{'label' : '{:,}'.format(x), 'value' : x} for x in threshold_options],
        multi=False,
        value=threshold_min,
        style={'float':'left', 'width':'200px'}
    )], style={'height':'24px'}),
    
    html.Table([
        html.Div(id='out_claim', children=[]), 
        html.Div(id='out_claim_types'), 
        html.Div('Source of data: my imagination ;o)'), 
        html.Div(children=['Author: Elsa Burren, elsa@promasta.com, ', html.A('www.promasta.com', href='https://www.promasta.com', target='_blank'), '. Feel free to contact me!'], style={'margin-top':'10px'})
    ], style={'width':'95vw', 'margin-top':'5px'}),
    
    html.P([
        html.Hr(style={'width':'100%', 'margin-top':'1vh', 'float':'left'}),
        html.H2('Claims and Frequency', style={'text-align': 'left', 'width':'100%'}),
        html.Div([
            html.Div([
                dcc.Graph(id='fig_claim', figure={}, style={'min-width':'400px'})
                ], style={'float':'left', 'max-width':'98%', 'min-width':'48%', 'margin-left':'1%'}),
            html.Div([
                dcc.Graph(id='fig_frq', figure={}, style={'min-width':'400px'})
                ], style={'float':'left', 'max-width':'98%', 'min-width':'48%', 'margin-left':'1%'}),
        ], style={'width':'100%', 'float':'left'})
    ], style={'max-width':'1600px'}),

    html.P([
        html.Hr(style={'width':'100%', 'margin-top':'1vh', 'float':'left'}),
        html.H2('Fitting Distributions', style={'text-align': 'left', 'width':'100%'}),

        html.Table([
            html.Div('Select Line(s): ', style={'float':'left', 'padding': '5px 5px 5px 3px', 'border': '1px solid rgb(200,200,200)', 'width':'150px', 'height':'24px'}), dcc.Dropdown(
            id='slct_line', 
            options=[{'label' : x, 'value' : x} for x in line_options],
            multi=True,
            value=line_options,
            style={'float':'left', 'width':'480px'}
        )], style={'height':'24px'}),

        html.Div([
            html.Div([
                dcc.Graph(id='fig_sev_dist', figure={}, style={'min-width':'400px'}),
                html.Table([
                    html.Div('Select Severity Distribution: ', style={'float':'left', 'padding': '5px 5px 5px 3px', 'border': '1px solid rgb(200,200,200)', 'width':'350px', 'height':'24px'}),
                    dcc.Dropdown(id='slct_sev_dist', 
                                 options=[{'label' : 'Pareto',    'value' : 'pareto'},
                                          {'label' : 'Gamma',     'value' : 'gamma'},
                                          {'label' : 'Lognormal', 'value' : 'lognorm'}],
                                multi=False,
                                value='pareto',
                                style={'float':'left', 'width':'200px'})
                ], style={'height':'24px'}),
                html.P(id='out_sev_dist', children=[])
                ], style={'float':'left', 'max-width':'98%', 'min-width':'48%', 'margin-left':'1%'}),
            html.Div([
                dcc.Graph(id='fig_frq_dist', figure={}, style={'min-width':'400px'}),
                html.Table([
                    html.Div('Select Frequency Distribution: ', style={'float':'left', 'padding': '5px 5px 5px 3px', 'border': '1px solid rgb(200,200,200)', 'width':'350px', 'height':'24px'}),
                    dcc.Dropdown(
                        id='slct_frq_dist', 
                        options=[{'label' : 'Poisson',  'value' : 'poisson'},
                                 {'label' : 'Flexible', 'value' : 'flexible'}],
                        multi=False,
                        value='poisson',
                        style={'float':'left', 'width':'200px'})
                ], style={'height':'24px'}),
                html.P(id='out_frq_dist', children=[])
                ], style={'float':'left', 'max-width':'98%', 'min-width':'48%', 'margin-left':'1%'})
        ], style={'width':'100%', 'float':'left'}),
    ], style={'max-width':'1600px'}),

    html.Table([
        html.Div(children=['Source code: ', html.A('github/elsaburren', href='https://github.com/elsaburren/claims-dashboard', target='_blank')]) 
    ], style={'width':'95vw', 'margin-top':'5px'})
])

# Create and connect Plotly graphs with Dash Components

# Claims Threshold
@app.callback(
        [Output(component_id='out_claim', component_property='children'),
         Output(component_id='out_claim_types', component_property='children'),         
         Output(component_id='fig_claim', component_property='figure'),
         Output(component_id='fig_frq', component_property='figure'),
         Output(component_id='slct_frq_dist', component_property='value')],
        [Input(component_id='slct_threshold', component_property='value')]
)
def update_graphs(slct_threshold):
    # Filter data
    dff = df[df['claim'] >= slct_threshold]

    # Create claims-time scatter plot
    fig_claim = px.scatter(dff, x='year', y='claim', color='line', title='Claims >= {:,}'.format(slct_threshold), hover_name='id='+dff['id'])
    fig_claim.update_layout(legend=dict(title='', orientation='h', yanchor='bottom', y=-0.3))

    # Create frequency bar chart
    fig_freq = px.histogram(dff, color='line', x='year', range_x=[year_min-1, year_max+1], nbins=year_max-year_min+3, labels=dict(x=str_years), title='Nb of claims >= {:,}'.format(slct_threshold))
    fig_freq.update_layout(bargap=0.1)
    fig_freq.update_layout(legend=dict(title='', orientation='h', yanchor='bottom', y=-0.3))
    
    # Create text output and return
    out_claim = 'All claims >= {:,} are considered.'.format(slct_threshold)
    out_claim_types = 'Split of claims frequency by line of business: ' + ' '.join((dff['line'].value_counts(normalize=True).to_frame().astype(float).apply(lambda row : round(row*100.0,2), axis=1).
                                                                                               astype(str)+'%').to_string(index=True,header=False).replace('\n',', ').split())
    return out_claim, out_claim_types, fig_claim, fig_freq, ''

# Severity fitting
@app.callback(
    [Output(component_id='fig_sev_dist', component_property='figure'),
     Output(component_id='out_sev_dist', component_property='children')],
    [Input(component_id='slct_threshold', component_property='value'),
     Input(component_id='slct_line', component_property='value'),
     Input(component_id='slct_sev_dist', component_property='value')]
)
def update_sev_dist(slct_threshold, slct_line, slct_dist):
    # Filter data
    dff = df.loc[df['line'].isin(slct_line), ['claim']]
    dff = dff.loc[df['claim'] >= slct_threshold,:]
    # Fit
    if slct_dist == 'pareto':
        param = pareto.fit(dff['claim'], method='MLE', floc=0.0, fscale=slct_threshold)
        #param = pareto.fit(dff['claim'], method='MLE')
    if slct_dist == 'gamma':
        param = gamma.fit(dff['claim'], method='MLE')
    if slct_dist == 'lognorm':
        param = lognorm.fit(dff['claim'], method='MLE', fscale=1.0)
        #param = lognorm.fit(dff['claim'], method='MLE')
    fitted_dist = Severity(shape=param[0], loc=param[1], scale=param[2], type=slct_dist)
    format_dist_name = {'pareto':'Pareto', 'gamma':'Gamma', 'lognorm':'Lognormal'}
    
    # Compute survival probability
    dff = create_survival(dff, 'claim', fitted_dist)
    
    # Create graph output
    fig_dist = px.scatter(dff, x='claim', y='sf', color='sf_type', title='Survival Prob Severity >= {:,}'.format(slct_threshold))
    fig_dist.update_layout(legend=dict(title='', orientation='h', yanchor='bottom', y=-0.3), yaxis=dict(title='survival function (1-CDF)'))
    # Create text output and return
    out_dist = 'Fitted {0} parameters are: shape={1:.4f}, scale={2:,.2f}, location={3:,.2f}'.format(format_dist_name[fitted_dist.type()], fitted_dist.shape(), fitted_dist.scale(), fitted_dist.loc())
    
    return fig_dist, out_dist

# Frequency fitting
@app.callback(
    [Output(component_id='fig_frq_dist', component_property='figure'),
     Output(component_id='out_frq_dist', component_property='children')],
    [Input(component_id='slct_threshold', component_property='value'),
     Input(component_id='slct_line', component_property='value'),
     Input(component_id='slct_frq_dist', component_property='value')]
)
def update_frq_dist(slct_threshold, slct_line, slct_dist):
    # Filter data
    dff = df.loc[df['line'].isin(slct_line), ['year', 'count']]
    dff = dff.loc[df['claim'] >= slct_threshold,:]
    
    # Compute annual nb
    dff = pd.concat([dff, format_frq])
    dff = dff.groupby(['year'], as_index=False)[['count']].apply(lambda x: x.sum())
    
    # Fit
    fitted_dist = Frequency(mean=dff['count'].mean()) if slct_dist =='poisson' else Frequency(mean=dff['count'].mean(), variance=dff['count'].var())
    format_dist_name = {'poisson':'Poisson', 'binom':'Binomial', 'nbinom':'Negative Binomial'}
    
    # Compute survival probabilities
    dff = create_survival(dff, 'count', fitted_dist)

    # Create graph output
    fig_dist = px.scatter(dff, x='count', y='sf', color='sf_type', title='Survival Prob Frequency')
    fig_dist.update_layout(legend=dict(title='', orientation='h', yanchor='bottom', y=-0.3), yaxis=dict(title='survival function (1-CDF)'), xaxis=dict(title='number claims >= {:,}'.format(slct_threshold)))
    
    # Create output and return
    out_dist = 'Fitted {0} parameters are: mean={1:.4f}, standard deviation={2:.4f}'.format(format_dist_name[fitted_dist.type()], fitted_dist.mean(), fitted_dist.sd())
    
    return fig_dist, out_dist

# Run the app
if __name__=='__main__':
    app.run_server(debug=True)
