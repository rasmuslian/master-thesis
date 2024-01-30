import mplfinance as mpf

def generate_graph(data, output_path):
    # Create a copy of the DataFrame
    data = data.copy()

    # Adds ma20 column
    ma20dict = mpf.make_addplot(data['ma20'])

    # Create a candlestick graph with the custom style and volume, and save it as an image
    mpf.plot(data,
            type='candle',
            #  mav=2,
            addplot=ma20dict,
            style='yahoo',
            figratio=(1,1),
            volume=True,
            # tight_layout=True,
            axisoff=True,
            savefig=dict(fname=output_path,bbox_inches="tight"),
            update_width_config=dict(candle_linewidth=5)
            )