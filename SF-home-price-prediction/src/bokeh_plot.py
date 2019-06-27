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

    TOOLTIPS = [
        ("Offer_Amount", "$@Offer_Amount{0,0}"),
        ("Company_Name", "@Company_Name"),
        ("Date_Filed", "@Date_Filed"),
        ("Symbol", "@Symbol"),
        ("Zipcode", "@Zipcode")
    ]

    # select the tools we want

    # the red and blue graphs will share this data range
    #xr1 = Range1d(minx, maxx)
    #yr1 = Range1d(miny, maxy)

    # Get the number of colors we'll need for the plot.



    # build our figures
    #p1 = figure(x_range=xr1, y_range=yr1, tools=TOOLS, plot_width=900, plot_height=900)
    p = bp.figure(plot_height=900, plot_width=1200, x_axis_type = 'datetime',  y_axis_type="log",  title="", toolbar_location=None, sizing_mode="scale_width", x_axis_label = 'Year', y_axis_label = 'Offer Amount')
    p.background_fill_alpha = 0
    p.border_fill_alpha = 0
    p.xaxis[0].formatter = DatetimeTickFormatter(months='%m/%Y')

    p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks

    p.yaxis.formatter = NumeralTickFormatter(format="$ 0,0[.]00")


    #p1.scatter(x, y, size=12, color="red", alpha=0.5)
    p.circle(x="Date_Filed_dt",y ="Offer_Amount", source=source, size='scaled_offers', color='color', fill_alpha=0.8,line_color='#7c7e71',line_width=0.5, line_alpha=0.5)

    p.add_tools(HoverTool(tooltips=TOOLTIPS))
    # plots can be a single PlotObject, a list/tuple, or even a dictionary

    script, div = components(p)
    print(script)
    print(div)


def main():
    df = load_graph_data("../data/processed/df_ipo.csv")
    create_scatter(df, 'Date Filed', 'Offer Amount')


main()
