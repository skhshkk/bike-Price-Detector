import streamlit as st
import pandas as pd
import pickle

# Load the saved model (which includes the preprocessing pipeline)
try:
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model: {str(e)}")

# Title and description
st.title("🏍️ Bike Price Prediction App")
st.markdown("Enter the bike details to get an estimated selling price.")

# List of bike names for auto-suggestions
bike_names = [
    "Royal Enfield Classic 350",
    "Honda Dio",
    "Royal Enfield Classic Gunmetal Grey",
    "Yamaha Fazer FI V 2.0 [2016-2018]",
    "Yamaha SZ [2013-2014]",
    "Honda CB Twister",
    "Honda CB Hornet 160R",
    "Royal Enfield Bullet 350 [2007-2011]",
    "Hero Honda CBZ extreme",
    "Bajaj Discover 125",
    "Yamaha FZ16",
    "Honda Navi",
    "Bajaj Avenger Street 220",
    "Yamaha YZF R3",
    "Jawa 42",
    "Suzuki Access 125 [2007-2016]",
    "Hero Honda Glamour",
    "Yamaha YZF R15 S",
    "Yamaha FZ25",
    "Hero Passion Pro 110",
    "Jawa Standard",
    "Royal Enfield Thunderbird 350",
    "Honda Dream Yuga",
    "TVS Apache RTR 160 4V",
    "Yamaha Fazer [2009-2016]",
    "Hero Honda Splendor NXG",
    "Hero Glamour 125",
    "Yamaha FZ S [2012-2016]",
    "Hero Xtreme Sports",
    "Honda X-Blade",
    "Honda CB Shine SP",
    "Honda Activa 5G",
    "Hero Glamour FI",
    "Bajaj Dominar 400",
    "KTM 390 Duke",
    "Hero Passion XPro",
    "Yamaha FZ S V 2.0",
    "Hero Achiever 150",
    "Yamaha Saluto",
    "Bajaj Discover 100",
    "Honda CB Trigger",
    "Royal Enfield Electra 5 S",
    "Hero Splendor PRO",
    "Hero Honda Passion Plus",
    "Bajaj Pulsar 150",
    "Bajaj Pulsar 150 [2001-2011]",
    "Honda Activa 3G",
    "Hero Hunk",
    "Suzuki Let''s",
    "Royal Enfield Electra 4 S",
    "TVS Scooty Pep Plus",
    "Mahindra Mojo XT300",
    "TVS Apache RTR 160",
    "Bajaj Pulsar AS200",
    "Royal Enfield Thunderbird 350X",
    "Suzuki Intruder 150",
    "Hero Honda Karizma ZMR [2010]",
    "Bajaj Xcd",
    "Hero Splendor Plus",
    "Honda CB Unicorn 150",
    "Honda Activa i [2016-2017]",
    "TVS Scooty Zest 110",
    "Hero CD Deluxe",
    "Suzuki GS150R",
    "Bajaj Pulsar 220S",
    "Honda Activa 4G",
    "Bajaj Pulsar NS160",
    "Royal Enfield Classic Desert Storm",
    "Suzuki Gixxer SF",
    "TVS Apache RTR 200 4V",
    "Bajaj V15",
    "TVS XL 100 Heavy Duty",
    "Aprilia SR 125",
    "Hero HF Deluxe",
    "Honda Aviator",
    "Vespa SXL 149",
    "Hero Xtreme [2013-2014]",
    "UM Renegade Commando",
]

# Initialize the session state for bike_name if not already set
if "bike_name" not in st.session_state:
    st.session_state.bike_name = ""

# Input for bike name with suggestions
bike_name_input = st.text_input(
    "Bike Name", value=st.session_state.bike_name, placeholder="Type the bike name..."
)

# Show filtered suggestions as the user types
matching_bikes = [
    name for name in bike_names if bike_name_input.lower() in name.lower()
]

# Display suggestions as buttons with unique keys
if matching_bikes and bike_name_input != "":
    st.write("Suggestions:")
    for i, suggestion in enumerate(matching_bikes):
        if st.button(suggestion, key=f"button_{i}"):  # Add unique key here
            st.session_state.bike_name = (
                suggestion  # Update session state with selected suggestion
            )

# Other inputs
year = st.number_input("Year", min_value=1900, max_value=2023, value=2020, step=1)
seller_type = st.selectbox("Seller Type", ["Individual", "Dealer"])
owner = st.selectbox(
    "Owner Type", ["1st owner", "2nd owner", "3rd owner", "4th owner and above"]
)
km_driven = st.number_input("Kilometers Driven", min_value=0, step=1)
ex_showroom_price = st.number_input("Ex-showroom Price (in INR)", min_value=0, step=500)

# Custom CSS for the predict button color
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #bd2052;
        color: white;
        font-weight: bold;
        padding: 0.5rem 2rem;
        border: none;
        border-radius: 5px;
    }
    div.stButton > button:hover {
        background-color: #a01b45;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Predict button
if st.button("Predict Price"):
    try:
        # Create input DataFrame with the same structure as training data
        input_data = pd.DataFrame(
            {
                "name": [st.session_state.bike_name],
                "year": [year],
                "seller_type": [seller_type],
                "owner": [owner],
                "km_driven": [km_driven],
                "ex_showroom_price": [ex_showroom_price],
            }
        )

        # Make prediction using the pipeline
        prediction = model.predict(input_data)

        # Display prediction
        st.markdown(
            f"""
            <div style="background-color: #bd2052; padding: 20px; border-radius: 10px; color: white; font-size: 22px; text-align: center; font-weight: bold; margin-top: 20px;">
                Predicted Selling Price: ₹{prediction[0]:,.2f}
            </div>
            """,
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
