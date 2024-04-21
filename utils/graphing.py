import matplotlib.pyplot as plt
import streamlit as st


def simply_line_plot(data, resale_price, flat_model=None, town_type=None):
    if flat_model is not None and town_type is not None:
        tag = f"{town_type.title()} and {flat_model.title()}"
    else:
        tag = flat_model.title() if flat_model is not None else town_type.title()
    # Plotting
    fig, ax = plt.subplots()

    # Calculate the minimum and maximum resale prices
    min_price = data['resale_price'].min()
    max_price = data['resale_price'].max()

    # Calculate the range between min and max, and add one unit higher than resale_price
    y_min = min_price
    y_max = max(max_price, resale_price) + 50000

    ax.plot(data['date'], data['resale_price'])  # Plotting resale prices over time
    ax.set_title(f'Resale Price for {tag}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Resale Price')
    ax.set_ylim(y_min, y_max)

    # Add horizontal line
    ax.axhline(y=resale_price, color='r', linestyle='--', linewidth=2)

    # Label for the horizontal line
    label_position_x = data['date'][0]  # Positioning the label at the start of the data; adjust as needed
    label_position_y = resale_price + 6000  # Slightly above the line
    ax.text(label_position_x, label_position_y, f'Predicted Resale Price: ${resale_price:,.0f}', color='red')

    # Rotate date labels for better readability
    plt.xticks(rotation=45)

    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)
