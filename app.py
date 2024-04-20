import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pygwalker.api.streamlit import init_streamlit_comm
import joblib
from utils.model_func import prediction_price
from utils.query import query_table_from_bq, load_model_from_gcs

from utils.tranformation import lr_prediction, format_input_to_dict

st.set_page_config(layout="wide")


def simply_line_plot(data, flat_model=None, town_type=None):
    if flat_model is not None and town_type is not None:
        tag = f"{town_type} and {flat_model}"
    else:
        tag = flat_model if flat_model is not None else town_type
    # Plotting
    fig, ax = plt.subplots()
    ax.plot(data['date'], data['resale_price'])  # Plotting resale prices over time
    ax.set_title(f'Resale Price for {tag}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Resale Price')

    # Add horizontal line
    ax.axhline(y=resale_price, color='r', linestyle='--', linewidth=2)

    # Label for the horizontal line
    label_position_x = data['date'][0]  # Positioning the label at the start of the data; adjust as needed
    label_position_y = resale_price + 6000  # Slightly above the line
    ax.text(label_position_x, label_position_y, f'Predicted Resale Price: {resale_price:,.0f}', color='red')

    # Rotate date labels for better readability
    plt.xticks(rotation=45)

    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)


data = pd.read_csv('testing/resale_flat_prices_2017-2024_new.csv')

st.title('IS3107 Project')

tab1, tab2= st.tabs(["Main Predicted Resale Price", "Methodology"])

init_streamlit_comm()

with st.sidebar:
    st.header("Input Parameters")
    # Numeric inputs
    floor_area_sqm = st.number_input('Floor Area (sqm)', min_value=0, format="%d")
    age_of_flat = st.number_input('Age of Flat', min_value=0, format="%d")
    avg_storey_range = st.number_input('Storey', min_value=0, format="%d")
    vacancy = st.number_input('Vacancy units left', min_value=0, format="%d")
    total_dwelling_units = st.number_input('Total Dwelling Units', min_value=0, format="%d")

    # Boolean input
    commercial = st.checkbox('Commercial Unit?')
    mrt = st.checkbox('Near MRT Interchange?')

    # Dropdown inputs
    town_type = st.selectbox('Town Type', ['jurong west', 'other',
                                           'punggol', 'sengkang', 'tampines', 'woodlands',
                                           'yishun'])
    flat_model = st.selectbox('Flat Model', [
        'model a', 'new generation', 'other', 'premium apartment',
    ])

    # Submit button
    submit_button = st.button("Submit")

with (tab1):
    col1, col2 = st.columns([2, 3])
    with col1:
        if submit_button:
            user_input = {'floor_area_sqm': floor_area_sqm,
                          'avg_storey_range': avg_storey_range,
                          'total_dwelling_units': total_dwelling_units,
                          'vacancy ': vacancy,
                          'commercial': 1 if commercial else 0,
                          'mrt_interchange': 1 if mrt else 0,
                          'age_of_flat': age_of_flat,
                          'flat_model': flat_model,
                          'town': town_type}

            st.write(user_input)
            model = load_model_from_gcs()
            formatted_input_dict = format_input_to_dict(user_input)
            resale_price = lr_prediction(model, formatted_input_dict)
            st.header(f"Predicted Resale Price {resale_price:,.0f}")
        else:
            st.write("Please input the required parameters")
            st.write("Then Press Submit")
    with col2:
        if submit_button:
            data = query_table_from_bq(town_type)
            data_filtered1 = data[(data["flat_model"] == flat_model)].groupby("date")[
                "resale_price"].mean().reset_index()
            if not data_filtered1.empty:
                st.title('Resale Price vs. Date')
                simply_line_plot(data_filtered1, flat_model, town_type)
            else:
                data_filtered2 = data.groupby("date")["resale_price"].mean().reset_index()

                if not data_filtered2.empty:
                    st.title('Resale Price vs. Date')
                    simply_line_plot(data_filtered2, town_type)
                else:
                    st.write("No Data Available")

with tab2:
    st.write("Testing:")

