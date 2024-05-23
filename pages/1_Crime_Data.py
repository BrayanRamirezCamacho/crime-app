import pandas as pd
import streamlit as st
import altair as alt
import numpy as np

st.set_page_config(page_title="Data comparison")

## LOADING DATA
@st.cache_data
def get_data():
    datos = pd.read_csv('./data/TMod_Vic.csv')
    
    # Extracting columns, but converting Series to lists for the new DataFrame
    dict = {
        'DELITO': datos['BPCOD,C,5'].tolist(),
        'SEXO': datos['SEXO,C,4'].tolist(),
        'EDAD': datos['EDAD,C,4'].tolist(),
        'MES': datos['BP1_1,C,5'].tolist(),
        'ESTADO': datos['BP1_2C,C,6'].tolist(),
        'HORA': datos['BP1_4,C,5'].tolist(),
        'LUGAR': datos['BP1_5,C,5'].tolist(),
    }

    df = pd.DataFrame(dict)
    df.set_index('DELITO', inplace=True)  # Set index directly on the DataFrame
    return df

def ml_pipeline(df):
    #Se elimina también la columa 'DELITO' por tratarse de la variable objetivo o target. Así, datos1 representa la matriz de diseño
    x = df.drop(columns=['DELITO'])
    y = df[['DELITO']]   
    # Instancia
    # Codificar etiquetas con valores entre 0 y {n_clases - 1}.
    le = preprocessing.LabelEncoder()
    # Ajuste y transformación
    # usar df.apply() para aplicar le.fit_transform a todas las columnas
    x2 = x.apply(le.fit_transform)
    # Instancia
    enc = preprocessing.OneHotEncoder()
    # Ajuste
    enc.fit(x2)
    # Transformación
    onehotlabels = enc.transform(x2).toarray()
    #División de los datos en 'entrenamiento' y 'prueba', destinando el 70% y 30%, respectivamente
    train_x, test_x, train_y, test_y = train_test_split(onehotlabels, y, train_size=0.7)
    # Modelo
    # mult_log_reg = LogisticRegression(multi_class='multinomial', solver='newton-cg', penalty='l2', tol=1e-4, C=C, max_iter=N, warm_start=False)
    mult_log_reg = LogisticRegression(penalty='l2', C=1e-1)
    # Entrenamiento
    mult_lr = mult_log_reg.fit(train_x, train_y)    

    return mult_lr

def data_plot(df):
    # ... (rest of your code to get 'df' and select 'states' remains the same)

    data = df.loc[states]
    st.write("### Criminality data per crime type", data.sort_index())

    column = st.selectbox("Choose a variable to plot", list(df.columns))

    # Pivot the DataFrame to create the 'State' column
    data = data.reset_index().pivot(index='DELITO', columns='index', values=column)

    # Melt the DataFrame to create 'Variable' and 'Value' columns
    data = pd.melt(data, ignore_index=False).rename(
        columns={"index": "State", "value": column}  # Rename to match your desired field names
    ).reset_index()

    chart = (
        alt.Chart(data,title=column)
        .mark_bar(opacity=0.6)
        .encode(
            x=alt.X("State:N", sort="-y"),  # 'N' for nominal data (categories)
            y=f"{column}:Q",             # 'Q' for quantitative data (numbers)
            tooltip=['State', column] # Display values when hovering.
        )
        .properties(height=500)
    )
    st.altair_chart(chart, use_container_width=True)


st.markdown("Criminality data")
st.markdown("Choose as many crime types as you'd like,")
st.markdown("Where the numbers mean: ")
st.markdown("CRIME: 01- Total theft of vehicle 02- Theft of vehicle accessories, spare parts or tools 03- Painting of fence in your house, vandalism 04- Robbery of house/apartment 05- Robbery or assault on the street or public transport 06- Robbery different from the previous one 07- Bank fraud, cloning of bank accounts, fraud. 08- Consumer fraud 09- Telephone extortion 10- Verbal threats 11- Violent attacks 12- Kidnapping 13- Sexual harassment 14- Sexual rape 15- Crimes other than the above")
st.markdown("GENDER: 1- Male 2- Female")
st.markdown("MONTH: 01- January 02- February 03- March 04- April 05- May 06- June 07- July 08- August 09- September 10- October 11- November 12- December 99- Does not know/does not respond")
st.markdown("STATE: 01- Aguascalientes 02- Baja California 03- Baja California Sur 04- Campeche 05- Coahuila 06- Colima 07- Chiapas 08- Chihuahua 09- CDMX 10- Durango 11- Guanajuato 12- Guerrero 13- Hidalgo 14- Jalisco 15- State of Mexico 16- Michoacán de Ocampo 17- Morelos 18- Nayarit 19- Nuevo León 20- Oaxaca 21- Puebla de Zaragoza 22- Querétaro 23- Quintana Roo 24- San Luis Potosi 25- Sinaloa 26- Sonora 27- Tabasco 28- Tamaulipas 29 - Tlaxcala 30- Veracruz Key 31- Yucatán 32- Zacatecas 99- Not specified")
st.markdown("TIME: 1- In the morning (from 06:01 to 12:00 hrs.) 2- In the afternoon (from 12:01 to 18:00 hrs.) 3- At night (from 18:01 to 24:00 hrs.) 4- In the early morning (from 00:01 to 06:00 hrs.) 9- Does not know/does not respond")
st.markdown("PLACE: 1- On the street 2- At home 3- At work 4- In a business or establishment 5- In a public place 6- On public transportation 7- On a highway 8- Other 9- Don't know/no reply")

st.sidebar.header("Comparing state-level data")
st.sidebar.caption("In this section, you can compare criminality data for different crime types!")
st.sidebar.caption("You can select as many crime type as you want to add to the dataframe, and you can select which variable to compare in the bar chart below.")

data_plot(df)

#Se utiliza la funcion corr() para hallar correlaciones entre las variables
df_corr = df.corr(method='pearson', min_periods=1)
#### COMO PONER ESTO BIEN?       
#Se muestra la matriz de correlación
sb.heatmap( df_corr )