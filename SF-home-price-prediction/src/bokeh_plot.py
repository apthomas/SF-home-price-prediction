import bokeh.plotting as bp
from bokeh.models import Range1d
from bokeh.embed import components
import pandas as pd
import aggregate_predictions
from bokeh.models import HoverTool, DatetimeTickFormatter, NumeralTickFormatter
from bokeh.palettes import inferno
import numpy as np
from sklearn.preprocessing import MinMaxScaler




def load_graph_data(filename):
    df = aggregate_predictions.load_csv(filename).sort_values(by='Date Filed', ascending=False)
    df.rename(columns={'Offer Amount': 'Offer_Amount'}, inplace=True)
    df.rename(columns={'Company Name': 'Company_Name'}, inplace=True)
    df.rename(columns={'Date Filed': 'Date_Filed'}, inplace=True)
    df['Date_Filed_dt'] = pd.to_datetime(df['Date_Filed'])
    df['Date_Filed_year'] = df['Date_Filed_dt'].dt.strftime('%Y')
    df['scaled_offers'] = df['Offer_Amount']
    df['scaled_offers'] -= -df['scaled_offers'].min()
    df['scaled_offers'] /=df['scaled_offers'].max()
    df['scaled_offers'] *= 200
    df['scaled_offers'][df['scaled_offers'] < 5] = 5
    return df


def create_scatter(df, x, y):
    # Create Column Data Source that will be used by the plot

    year_list = df['Date_Filed_year'].unique().tolist()
    colors = inferno(len(year_list))
    # Create a map between factor and color.
    colormap = {}
    for i in range(0, len(year_list)):
        colormap[year_list[i]] = colors[i]
    # Create a list of colors for each value that we will be looking at.
    color = [colormap[x] for x in df['Date_Filed_year']]
    df['color'] = color
    source = bp.ColumnDataSource(df)

    TOOLTIPS_P = [
        ("Offer_Amount", "$@Offer_Amount{0,0}"),
        ("Company_Name", "@Company_Name"),
        ("Date_Filed", "@Date_Filed"),
        ("Symbol", "@Symbol"),
        ("Zipcode", "@Zipcode")
    ]
    p = bp.figure(plot_height=900, plot_width=1250, x_axis_type = 'datetime',  y_axis_type="log",  title="", toolbar_location=None, sizing_mode="scale_width", x_axis_label = 'Year', y_axis_label = 'Offer Amount')
    p.background_fill_alpha = 0
    p.border_fill_alpha = 0
    p.xaxis[0].formatter = DatetimeTickFormatter(months='%m/%Y')
    p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
    p.yaxis.formatter = NumeralTickFormatter(format="$ 0,0[.]00")
    p.circle(x="Date_Filed_dt",y ="Offer_Amount", source=source, size='scaled_offers', color='color', fill_alpha=0.8,line_color='#7c7e71',line_width=0.5, line_alpha=0.5)
    p.add_tools(HoverTool(tooltips=TOOLTIPS_P))



    grouped_by_year = df.groupby(['Date_Filed_year']).agg({
        'Offer_Amount':['sum','mean', 'count']
    })
    grouped_by_year.columns = ['_'.join(col).strip() for col in grouped_by_year.columns.values]
    grouped_by_year.reset_index(inplace=True)
    grouped_by_year['Date_Filed_simple'] = pd.to_datetime(grouped_by_year['Date_Filed_year'], format="%Y")
    color = [colormap[x] for x in grouped_by_year['Date_Filed_year']]
    grouped_by_year['color'] = color
    grouped_source = bp.ColumnDataSource(grouped_by_year)

    TOOLTIPS_H = [
        ("Sum_Offer_Amount", "$@Offer_Amount_sum{0,0}"),
        ("Year", "@Date_Filed_year")
    ]

    h = bp.figure(plot_height=900, plot_width=1250, x_axis_type = 'datetime',  title="", toolbar_location=None, sizing_mode="scale_width", x_axis_label = 'Year', y_axis_label = 'Sum of Offer Amounts')
    h.background_fill_alpha = 0
    h.border_fill_alpha = 0
    h.xaxis[0].formatter = DatetimeTickFormatter(months='%m/%Y')
    h.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
    h.yaxis.formatter = NumeralTickFormatter(format="$ 0,0[.]00")
    h.vbar(x='Date_Filed_simple', top='Offer_Amount_sum', width=100, color='color', source=grouped_source)
    h.add_tools(HoverTool(tooltips=TOOLTIPS_H, mode='vline'))

    # plots can be a single Bokeh Model, a list/tuple, or even a dictionary
    plots = {'scatter': p, 'bar': h}

    script, div = components(plots)

    print(script)
    print(div)


def main():
    df = load_graph_data("../data/processed/df_ipo.csv")
    create_scatter(df, 'Date Filed', 'Offer Amount')

if __name__ == "__main__":
    print("we are building plots")
    main()
