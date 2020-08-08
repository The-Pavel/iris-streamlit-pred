import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
  ## Let's play **guess that Iris** :tulip:

  """)

flowers = datasets.load_iris()

data = pd.DataFrame(flowers.data)

def user_input():
    sepal_length = st.sidebar.slider('Sepal Length', 4.3, 7.9)
    sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4)
    petal_length = st.sidebar.slider('Petal Length', 1.0, 6.9)
    petal_width = st.sidebar.slider('Sepal Length', 0.1, 2.5)
    features = {
    'Sepal Length': sepal_length,
    'Sepal Width': sepal_width,
    'Petal Length': petal_width,
    'Petal Width': petal_length
    }
    return pd.DataFrame(features, index=[0])


feature_df = user_input()
st.write("Tune these features in the **widgets!** 🤩")
st.write(feature_df)

st.write("We will be predicting one of these Iris flower categories :rose:")
labels = flowers.target_names
labels

rfc = RandomForestClassifier()

data = flowers.data
target = flowers.target

rfc.fit(data, target)

prediction = rfc.predict(feature_df)
prediction_proba = rfc.predict_proba(feature_df)

st.write("The robots are doing their thinking... 🤖")
st.write(prediction_proba)

st.write("We concur - it's a **" + labels[prediction][0] + "!**")


# YAHOO FINANCE CHART DRAWER #
# import yfinance as yf

# st.write("""
#   # Welcome to Streamlit :rocket:

#   _We will rock it._

#   """)

# ticker = "GOOGL"
# tickerData = yf.Ticker(ticker)




# start = st.date_input('Select start date')
# end = st.date_input('Select end date')
# df = tickerData.history(interval='1d', start=start, end=end)
# options = st.multiselect("What to draw", df.columns)



# trimmed_df = df[options]
# st.line_chart(trimmed_df)
# st.line_chart(df['Volume'])
