import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the saved model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Extract unique bike names from the dataset
unique_bike_names = [
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
    "Hero Glamour",
    "Royal Enfield Bullet 500",
    "Hero Honda Splendor Plus",
    "Yamaha FZ S",
    "TVS Apache RTR 160",
    "Suzuki Gixxer SF",
    "KTM RC200",
    "Honda CBR 150R",
    "Royal Enfield Thunderbird 350",
    "Honda CB Unicorn",
    "Bajaj Pulsar 150",
    "Royal Enfield Himalayan",
    "Bajaj Avenger Street 150",
    "Honda CB Unicorn 160",
    "TVS Jupiter",
    "Yamaha FZ",
    "Honda Activa",
    "Hero Passion Pro",
    "Suzuki Access",
    "KTM Duke 200",
    "Hero Honda CD 100",
    "Yamaha YZF R15",
    "Bajaj Pulsar NS200",
    "Honda Dream Neo",
    "TVS Apache RTR 180",
    "Hero Splendor iSmart",
    "Hero Honda Hunk",
    "Hero HF Deluxe",
    "Honda Activa i",
    "Hero Honda Super Splendor",
    # Add all other unique names from your dataset here
]

# Manually define the classes for LabelEncoder to avoid loading preprocessed_data.pkl
le_bike_name = LabelEncoder()
le_bike_name.classes_ = np.array(unique_bike_names)
le_seller_type = LabelEncoder()
le_seller_type.classes_ = np.array(["Individual", "Dealer"])
le_owner = LabelEncoder()
le_owner.classes_ = np.array(
    ["1st owner", "2nd owner", "3rd owner", "4th owner and above"]
)

# Title and description
st.title("🏍️ Bike Price Prediction App")
st.markdown(
    """
    Welcome to the **Bike Price Prediction App**! 
    Enter the details of the bike you're interested in to get an estimated selling price. This model leverages historical bike sales data.
"""
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
        .main-title { 
            font-size: 32px; 
            color: #2E86C1;
            font-weight: bold;
            text-align: center;
            margin-bottom: 25px;
        }
        .custom-button {
            background-color: #54196b;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }
        .custom-button:hover {
            background-color: #B695C0;
        }
        .result-box {
            background-color: #333333;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            color: white;
            font-size: 22px;
            text-align: center;
            font-weight: bold;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# Get user input
bike_name = st.selectbox(
    "Bike Name",
    [""] + list(le_bike_name.classes_),
    index=0,
    help="Start typing to select the bike name",
)
year = st.number_input(
    "Year",
    min_value=1900,
    max_value=2023,
    value=2020,
    step=1,
    help="Enter the manufacturing year",
)
seller_type = st.selectbox(
    "Seller Type",
    le_seller_type.classes_,
    index=0,
    help="Is the seller an individual or a dealer?",
)
owner = st.selectbox(
    "Owner Type",
    le_owner.classes_,
    index=0,
    help="Select the ownership type of the bike",
)
km_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    step=1,
    help="Enter the total kilometers the bike has been driven",
)
ex_showroom_price = st.number_input(
    "Ex-showroom Price (in INR)",
    min_value=0,
    step=500,
    help="Enter the ex-showroom price of the bike",
)

# Prepare the user input as a dataframe for prediction
user_input = pd.DataFrame(
    {
        "name": [bike_name] if bike_name else [np.nan],
        "year": [year],
        "seller_type": [seller_type],
        "owner": [owner],
        "km_driven": [km_driven],
        "ex_showroom_price": [ex_showroom_price],
    }
)

# Ensure input is valid for the model
if bike_name:
    user_input["name"] = le_bike_name.transform(user_input["name"].astype(str))
user_input["seller_type"] = le_seller_type.transform(
    user_input["seller_type"].astype(str)
)
user_input["owner"] = le_owner.transform(user_input["owner"].astype(str))

# Use custom HTML button for prediction
predict_button = st.markdown(
    '<button class="custom-button" onclick="window.predict()">Predict Price</button>',
    unsafe_allow_html=True,
)

# JavaScript function to trigger prediction
st.markdown(
    """
    <script>
        function predict() {
            const event = new Event("click");
            document.querySelector("button[aria-label='Predict Price']").dispatchEvent(event);
        }
    </script>
    """,
    unsafe_allow_html=True,
)

# Prediction logic
if predict_button and bike_name:
    prediction = model.predict(user_input)
    st.markdown(
        f"""
        <div class="result-box">
            Predicted Selling Price: ₹{prediction[0]:,.2f}
        </div>
    """,
        unsafe_allow_html=True,
    )

# Footer information
st.write(
    "This app provides an estimate and should be used as a guideline only. Prices may vary."
)
