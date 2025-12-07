import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import seaborn as sns
from xgboost import XGBRegressor
import optuna
from sklearn import metrics
import altair as alt
from streamlit_option_menu import option_menu
from catboost import CatBoostRegressor
import base64
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Dashboard ML Project", layout="wide")
st.markdown(
    """
    <h1 style="color:#004269; font-size:70px; text-align:center;">
        Overview Of Our NBA Model   
    </h1>
    """, 
    unsafe_allow_html=True
)
with st.sidebar:

    selected = option_menu (
            menu_title="Navigation",
            options=["Dataset Presentation", "Model Presentation"],
            icons=["file-earmark-text", "file-earmark-text" ],
            menu_icon="cast",
            orientation="vertical",
            default_index=0
            
    )
    
    df = pd.read_csv('final_data.csv')
    pd.set_option('display.max_columns', None)
    columns_to_drop = [
    'GAME_ID', 'TEAM_ID', 'TEAM_NAME', 'TEAM_ABBREVIATION', 'TEAM_CITY',
    'HOME_TEAM', 'MIN', 'RESULT', 'PLUS_MINUS', 'FGM', 'FG3M', 'FTM'
    ]

    df_cleaned = df.drop(columns=columns_to_drop)
    X = df_cleaned.drop(columns=['PTS'])
    y = df_cleaned['PTS']

    df_cleaned = df_cleaned.drop_duplicates()

    target = "PTS"
    X = df_cleaned.drop(columns=[target])
    y = df_cleaned[target]

    label_encoders = {}
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)   

def page_1():
   

    st.write(df_cleaned.head())
    col1, col2=st.columns(2)
    with col1:
        st.markdown(
        """
        <h4 style="color:#004269; font-size:40px; text-align:start;">
            Target Variable   
        </h4>
        """, 
        unsafe_allow_html=True
        )
        st.write(df_cleaned['PTS'].describe())
    with col2:

        image = Image.open("Lebron-James-Basketball-Player-PNG-Pic.png")
        st.image(image, use_container_width=True)


    X_sub, _, y_sub, _ = train_test_split(
        X_train_scaled, y_train, 
        test_size=0.5,  
        random_state=42,
        shuffle=True
    )


    Z = TSNE().fit_transform(X_sub)

    df_tsne = pd.DataFrame({
        "TSNE1": Z[:, 0],
        "TSNE2": Z[:, 1],
        "PTS": y_sub.reset_index(drop=True)
    })
    tab1, tab2= st.tabs(["TSNE", "Corolary"])
    with tab1:
        chart = (
            alt.Chart(df_tsne)
            .mark_circle(size=60)
            .encode(
                x='TSNE1',
                y='TSNE2',
                color='PTS',
                tooltip=['TSNE1', 'TSNE2', 'PTS']
            )
            .interactive()
        )
        st.markdown(
        """
        <h4 style="color:#004269; font-size:40px; text-align:start;">
            TSNE Visualization   
        </h4>
        """, 
        unsafe_allow_html=True
        )
        st.altair_chart(chart, use_container_width=True)
    with tab2:
        df_numeric=df_cleaned.select_dtypes(include=[np.number])
        corr_matrix=df_numeric.corr()
        plt.figure(figsize=(20, 16))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, cbar_kws={"shrink":0.8}, annot_kws={"size":9})
        plt.title('correlation matrix')
        plt.tight_layout()
        plt.show()
        st.pyplot(plt) 

    

def page_2():

    cat_model = CatBoostRegressor(
        depth=4,
        learning_rate=0.24796976097472606,
        iterations=716,
        random_seed=42,
        verbose=100
    )
    cat_model.fit(X_train, y_train)  

    col1, col2 =st.columns(2)
    with col1:
        importances = pd.Series(cat_model.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False).head(10)

        plt.figure(figsize=(10, 8))
        importances.plot(kind="barh")
        plt.title("Variable Importance (CatBoost)")
        st.pyplot(plt)
    with col2:
        image = Image.open("nba-action-shot-png-76-vvc5nzd7t3flcx9x.png")
        st.image(image, use_container_width=True)

    variables = {
        "Offensive Rebounds": df_cleaned['OREB'].unique(),
        "Throw Percentage": df_cleaned['FT_PCT'].unique(),
        "Three-Point Field Goal Percentage": df_cleaned['FG3_PCT'].unique(),
        "Field Goal Percentage": df_cleaned['FG_PCT'].unique(),
        "COVID_FLAG": df_cleaned['COVID_FLAG'].unique(),
        "Free Throw Attempts": df_cleaned['FTA'].unique(),
        "Effective Field Goal Percentage": df_cleaned['EFG_PCT'].unique(),
    }

    st.markdown(
        """
        <h4 style="color:#004269; font-size:40px; text-align:start;">
            Value Menu  
        </h4>
        """, 
        unsafe_allow_html=True
    )
    selections = {}
    for var_name, values in variables.items():
        selections[var_name] = st.selectbox(f"Select {var_name} :", sorted(values))

    
    col1, col2=st.columns(2)
    with col1:
        st.json(selections)
        if st.button("Number of Points"):

            input_df = pd.DataFrame(columns=X.columns)

            input_df.loc[0] = 0
            for k, v in selections.items():
                input_df.loc[0, k] = v

            prediction = cat_model.predict(input_df)[0]

            st.success(f"Prediction of the number of points : {prediction:.2f}")
    with col2:
        image=Image.open('s-l1200.png')
        st.image(image, use_container_width=True)

    
if selected == "Dataset Presentation":
    page_1()
elif selected == "Model Presentation":
    page_2()