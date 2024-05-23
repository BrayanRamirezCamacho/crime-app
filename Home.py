import streamlit as st


## Page config
st.set_page_config(page_title="Crime MX")

## Introductory text

st.markdown("# Welcome!")
st.markdown("In this app, I try to visualize criminality data for different mexican states. The variables available in the dataset are the crime commited, gender, age, state, month, hour and place.")

st.markdown("The data used for this project has been retrieved from Mexico's National Institute of Statistics and Geography, or INEGI.")

col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image("https://upload.wikimedia.org/wikipedia/commons/8/89/INEGI.png?20200220225518")

with col3:
    st.write(' ')

st.markdown("This dashboard uses 2018 data provided by INEGI. You can find tons of interesting data to explore on their website, which you can access via [this link](https://www.inegi.org.mx/)!")

st.sidebar.header("Select a section above to begin!")