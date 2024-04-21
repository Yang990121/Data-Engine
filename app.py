import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pygwalker.api.streamlit import init_streamlit_comm
import joblib
from utils.model_func import prediction_price
from utils.query import query_table_from_bq_filtered, load_model_from_gcs

from utils.tranformation import model_prediction, format_input_to_dict

st.set_page_config(layout="wide")


def simply_line_plot(data, flat_model=None, town_type=None):
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
    ax.text(label_position_x, label_position_y, f'Predicted Resale Price: {resale_price:,.0f}', color='red')

    # Rotate date labels for better readability
    plt.xticks(rotation=45)

    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)


data = pd.read_csv('testing/resale_flat_prices_2017-2024_new.csv')

st.title('Singapore Housing Resale Price Prediction')

init_streamlit_comm()

if 'submitted' not in st.session_state:
    st.session_state['submitted'] = False

with st.sidebar:
    st.header("Input Parameters")
    # Numeric inputs

    floor_area_sqm = st.number_input('Floor Area (sqm)', min_value=0, value=100, format="%d")
    age_of_flat = st.number_input('Age of Flat', min_value=0, value=10, format="%d")
    avg_storey_range = st.number_input('Storey', min_value=0, value=10, format="%d")
    # vacancy = st.number_input('Vacancy units left', min_value=0, format="%d")
    total_dwelling_units = st.number_input('Total Dwelling Units', min_value=0, value=120, format="%d")
    
    # Boolean input
    commercial = st.checkbox('Commercial Unit?')
    # mrt = st.checkbox('Near MRT Interchange?')

    # Dropdown inputs
    town_type = st.selectbox('Town Type', ['jurong west', 'other', 'punggol', 'sengkang', 'tampines', 'woodlands', 'yishun'])
    flat_model = st.selectbox('Flat Model', ['model a', 'new generation', 'other', 'premium apartment'])

    # Submit button
    if st.button("Submit"):
        st.session_state['submitted'] = True
    else:
        st.session_state['submitted'] = False

with st.container():
    tab1, tab2 = st.tabs(["Main Predicted Resale Price", "Methodology"])
    with (tab1):
        col1, col2 = st.columns([1.2, 0.8])
        with col1:
            model_lasso = load_model_from_gcs("linear_model.pkl")
            model_lr = load_model_from_gcs("linear_model_all.pkl")
            model_knn = load_model_from_gcs("knn_full.pkl")
            if st.session_state['submitted']:
                user_input = {'floor_area_sqm': floor_area_sqm,
                              'avg_storey_range': avg_storey_range,
                              'total_dwelling_units': total_dwelling_units,
                              'vacancy ': 20,
                              'commercial': 1 if commercial else 0,
                              'mrt_interchange': 1,
                              'age_of_flat': age_of_flat,
                              'flat_model': flat_model,
                              'town': town_type}
                # st.write("Your inputs")
                # st.write(user_input)
                formatted_input_dict = format_input_to_dict(user_input)
                # st.write("Test")
                # st.write(formatted_input_dict)
                resale_price =0.5*model_prediction(model_lr, formatted_input_dict, False) + \
                    0.5*model_prediction(model_knn, formatted_input_dict, False)
                        
                        
                st.title(f"Predicted Resale Price: ${resale_price:,.0f}")
                
            else:
                st.write("Please input the required parameters")
                st.write("Then Press Submit")
        with col1:
            if st.session_state['submitted']:
                if town_type == "other":
                    town_filter = ['ang mo kio',
                                   'bedok',
                                   'bishan',
                                   'bukit batok',
                                   'bukit merah',
                                   'bukit panjang',
                                   'bukit timah',
                                   'central area',
                                   'choa chu kang',
                                   'clementi',
                                   'geylang',
                                   'hougang',
                                   'jurong east',
                                   'kallang/whampoa',
                                   'marine parade',
                                   'pasir ris',
                                   'queenstown',
                                   'serangoon',
                                   'toa payoh',
                                   'sembawang']
                else:
                    town_filter = [town_type]
                if flat_model == "other":
                    flat_filter = ['improved',
                                   'adjoined flat',
                                   'standard',
                                   'apartment',
                                   'maisonette',
                                   'model a-maisonette',
                                   'simplified',
                                   'multi generation',
                                   '2-room',
                                   'terrace',
                                   'improved-maisonette',
                                   'premium maisonette',
                                   'model a2',
                                   'dbss',
                                   'type s1',
                                   'type s2',
                                   'premium apartment loft']
                else:
                    flat_filter = [flat_model]
                data = query_table_from_bq_filtered(town_filter)
                data_filtered1 = data[(data["flat_model"].isin(flat_filter))].groupby("date")[
                    "resale_price"].mean().reset_index()
                if not data_filtered1.empty:
                    st.header('Historical Resale Price')
                    simply_line_plot(data_filtered1, flat_model, town_type)
                else:
                    data_filtered2 = data.groupby("date")["resale_price"].mean().reset_index()

                    if not data_filtered2.empty:
                        st.header('Historical Resale Price')
                        simply_line_plot(data_filtered2, town_type)
                    else:
                        st.write("No Data Available")

    with tab2:
        col1, col2, col3 = st.columns([5, 0.5,  5])
        with col1:
            st.header("Parameters Available")
            st.write("**Floor Area (sqm):** Total area of the flat in square meters")
            st.write("**Age of Flat:** Number of years since the flat was built")
            st.write("**Story:** Number of stories for the flat")
            # st.write("**Vacancy units left:** Number of unsold or unoccupied units in the building")
            st.write("**Total Dwelling Units:** Number of residential units in the building")
            st.write("**Commercial Unit:** Check for commercial development")
            # st.write("**Near MRT interchange:** Proximity to an MRT interchange")
            st.write("**Town Type:** General area the flat is located in")
            st.write("**Flat Model:** The design and layout type of the flat")
        with col3:
            st.header("Prediction Model Used")
            st.write("This calculator helps you make informed decisions by predicting outcomes based on your inputs. "
                     "It uses a combination of advanced techniques that analyze historical data patterns.")
            st.write("Think of it as having two expert advisors: one that uses trends and averages (like Linear "
                     "Regression) and another that compares your situation to similar past examples (like K-Nearest "
                     "Neighbors).")
            st.write("We combine their advice to give you the most accurate prediction possible.")
            st.write("Simply enter the required information, and let our tool do the rest to provide you with a "
                     "reliable forecast!")
            st.write("We will also provide historical resale price for comparison between the predicted and "
                     "historical price")


