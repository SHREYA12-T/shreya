import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from collections import Counter
import plotly.express as px
from ydata_profiling import ProfileReport
import base64
import openpyxl
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Set page config (must be the first Streamlit command)
st.set_page_config(page_title="EDA", page_icon=":bar_chart:", layout="wide", initial_sidebar_state="collapsed")

 # Function to read and encode the image file
@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to set the background image
def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    [data-testid="stAppViewContainer"] {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    }
    </style>
    ''' % bin_str
        
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set the background image
set_png_as_page_bg('img.png')

# Define the Data_Base_Modelling class
class Data_Base_Modelling():

    def __init__(self):
        print("Data_Base_Modelling object created")

    def Label_Encoding(self, x):
        category_col = [var for var in x.columns if x[var].dtypes == "object"]
        labelEncoder = preprocessing.LabelEncoder()
        mapping_dict = {}
        for col in category_col:
            x[col] = labelEncoder.fit_transform(x[col])
            le_name_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
            mapping_dict[col] = le_name_mapping
        return mapping_dict

    def IMpupter(self, x):
        imp_mean = IterativeImputer(random_state=0)
        x = imp_mean.fit_transform(x)
        x = pd.DataFrame(x)
        return x

    def Logistic_Regression(self, x_train, y_train, x_test, y_test):
        pipeline_dt = Pipeline([('dt_classifier', LogisticRegression())])
        pipeline_dt.fit(x_train, y_train)
        return classification_report(y_test, pipeline_dt.predict(x_test))

    def Decision_Tree(self, x_train, y_train, x_test, y_test):
        pipeline_dt = Pipeline([('dt_classifier', DecisionTreeClassifier())])
        pipeline_dt.fit(x_train, y_train)
        return classification_report(y_test, pipeline_dt.predict(x_test))

    def RandomForest(self, x_train, y_train, x_test, y_test):
        pipeline_dt = Pipeline([('dt_classifier', RandomForestClassifier())])
        pipeline_dt.fit(x_train, y_train)
        return classification_report(y_test, pipeline_dt.predict(x_test))

    def naive_bayes(self, x_train, y_train, x_test, y_test):
        pipeline_dt = Pipeline([('dt_classifier', GaussianNB())])
        pipeline_dt.fit(x_train, y_train)
        return classification_report(y_test, pipeline_dt.predict(x_test))

    def XGb_classifier(self, x_train, y_train, x_test, y_test):
        pipeline_dt = Pipeline([('dt_classifier', XGBClassifier())])
        pipeline_dt.fit(x_train, y_train)
        return classification_report(y_test, pipeline_dt.predict(x_test))


# Add custom CSS
def custom_css():
    custom_css = """
    <style>
    body {
        background-color: #f5f5f5;
        font-family: Arial, sans-serif;
        margin: 0;
        padding: 0;
    }

    .container {
        max-width: 800px;
        margin: 0 auto;
        text-align: center;
        padding: 40px;
    }

    .header {
        font-size: 48px;
        font-weight: bold;
        color: #333;
        margin-bottom: 16px;
    }

    .tagline {
        font-size: 24px;
        color: #666;
        margin-bottom: 32px;
    }

    .features {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        margin-bottom: 40px;
    }

    .feature {
        flex: 1;
        text-align: center;
        padding: 20px;
        background-color: #fff;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin: 8px;
        transition: transform 0.3s ease-in-out;
    }

    .feature:hover {
        transform: scale(1.05);
    }

    .feature-icon {
        font-size: 36px;
        color: #4CAF50;
    }

    .feature-title {
        font-size: 18px;
        font-weight: bold;
        margin-top: 16px;
    }

    .action-button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 16px 32px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        transition: background-color 0.3s;
    }

    .action-button:hover {
        background-color: #45a049;
    }

    .target-audience {display: flex; justify-content: space-between; flex-wrap: wrap;}
    .audience {flex: 0 1 calc(50% - 10px); background-color: #f6f6f6; border-radius: 10px; margin: 5px; padding: 10px; text-align: center;}
    .audience-icon {font-size: 2em;}
    .start-button {display: inline-block; margin-top: 20px; background-color: #1E90FF; color: #FFF; padding: 10px 20px; text-align: center; border-radius: 5px; text-decoration: none;}
    .thank-you {font-size: 1.5em; margin-top: 20px; text-align: center; color: #555;}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

custom_css()


# Hide default Streamlit menu and footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Define a function to generate download link for a dataframe
def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    filename = uploaded_file.name.split(".")[0]
    if filename.startswith("cleaned"):
        filename = filename.split("cleaned_")[1]
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download ="cleaned_{filename}.csv">Download csv file</a>'
    return href

# Define a function to load data
def load_data(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, na_values=" ", keep_default_na=True)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        elif uploaded_file.name.endswith('.pkl'):
            df = pd.read_pickle(uploaded_file)
        return df

# Define a function to get categorical columns
def categorical_column(df):
    return [col for col in df.columns if df[col].dtype == 'object']

# Home page
def show_home_page():
    st.title("Welcome to Automated Data Preprocessing and Exploratory Data Analysis for Enhanced Data Insights ")
    st.text("A data analytics tool to make data processing & EDA simpler than ever!")

    # Key Features
    st.subheader("Key Features")
    st.write("📊 **Interactive Exploration:** Explore your datasets with interactive visualizations.")
    st.write("📈 **Stunning Charts:** Visualize data with beautiful and informative charts.")
    st.write("🛠️ **Effortless Preprocessing:** Streamline data preprocessing and preparation.")

    # Get Started Section
    st.subheader("Get Start with venture")
    st.write("This is your gateway to data analysis and preprocessing. We've simplified the process to help you make the most of your data.")

    # Target Audience
    st.write('<div class="target-audience">'
                '<div class="audience">'
                '<div class="audience-icon">📊</div>'
                '<div class="audience-title">Data Analysts</div>'
                '</div>'
                '<div class="audience">'
                '<div class="audience-icon">🔎</div>'
                '<div class="audience-title">Data Scientists</div>'
                '</div>'
                '<div class="audience">'
                '<div class="audience-icon">🧐</div>'
                '<div class="audience-title">Business Professionals</div>'
                '</div>'
                '<div class="audience">'
                '<div class="audience-icon">📈</div>'
                '<div class="audience-title">Students and Educators</div>'
                '</div>'
                '</div>', unsafe_allow_html=True)

    # Example Dataset
    st.subheader("Try it Out!")
    st.write("Get started by uploading your own dataset or use the example dataset included in sidebar. Select it and let AutoEDA do the rest!")

    # Final Message
    st.write('<div class="thank-you">Start your journey towards data-driven decision-making with AutoEDA!</div>', unsafe_allow_html=True)

# Sidebar for file upload and task selection
with st.sidebar:
    st.header("Begin the EDA-venture")
    task = st.selectbox("Choose your task", ["Home", "Data Exploration", "Data Preprocessing", "Data Cleaning", "Data Visualization", "Data Profiling","Model Building"])
    uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx"])
    st.markdown("**Note:** Only .csv, .xlsx  files are supported.")
    if uploaded_file is not None:
        df = load_data(uploaded_file)


if uploaded_file is None and task != "Home":
    st.info("Please upload a file to get started.")
else:
    if task == "Home":
        show_home_page()
    elif task == "Data Exploration":
        st.subheader("Data Exploration")
        with st.expander("Show Data"):
            st.dataframe(df)
        st.subheader("Visualise a column:")
        cols = ['None']
        cols.extend(df.columns)
        plot_col = st.selectbox("Select a column", cols)
        if plot_col != 'None':
            st.markdown(f"**Plotting the distribution of: {plot_col}**")
            st.altair_chart(alt.Chart(df).mark_bar().encode(
                x=alt.X(plot_col, bin=alt.Bin(maxbins=20)),
                y='count()'
            ))
        else:
            st.markdown("**No column selected.**")

    elif task == "Data Preprocessing":
        st.subheader("Data Preprocessing")
        all_columns = df.columns.tolist()
        if st.checkbox("Show Dataset"):
            number = st.number_input("Numbers of rows to view", 5)
            st.dataframe(df.head(number))

        if st.checkbox("Show Shape"):
            st.write(df.shape)
            data_dim = st.radio("Show Dimension by", ("Rows", "Columns"))
            if data_dim == "Columns":
                st.text("Number of Columns")
                st.write(df.shape[1])
            elif data_dim == "Rows":
                st.text("Number of Rows")
                st.write(df.shape[0])

        if st.checkbox("Show Columns"):
            st.write(df.columns)

        if st.checkbox("Summary"):
            st.write(df.describe())

        if st.checkbox("Show Selected Columns"):
            selected_columns = st.multiselect("Select Columns", all_columns)
            new_df = df[selected_columns]
            st.dataframe(new_df)

        if st.checkbox("Show Data Types"):
            st.write(df.dtypes)

        
                
    elif task == "Data Cleaning":
        st.subheader("Data Cleaning")
        all_columns = df.columns.tolist()
        st.write("Remove columns with missing values:")
        clean_columns = st.multiselect("Select columns", all_columns)
        cleaned_df = df.drop(clean_columns, axis=1)
        st.dataframe(cleaned_df)
        st.markdown(get_table_download_link(cleaned_df), unsafe_allow_html=True)
        
        st.write("Fill missing values:")
        if st.checkbox("Fill Missing Values"):
    # Separate numerical and categorical columns
            num_cols = df.select_dtypes(include='number').columns.tolist()
            cat_cols = df.select_dtypes(include='object').columns.tolist()
    
            # Fill missing values in numerical columns with the mean
            for col in num_cols:
                df[col].fillna(df[col].mean(), inplace=True)
    
            # Fill missing values in categorical columns with the mode
            for col in cat_cols:
                df[col].fillna(df[col].mode()[0], inplace=True)
    
            st.dataframe(df)
            st.markdown(get_table_download_link(df), unsafe_allow_html=True)

            
    
    elif task == "Data Visualization":
        st.subheader("Data Visualization")
        st.text("_______________________________________________________________________________________________")
    
        # Existing Streamlit code
        type_of_plot = st.selectbox("Select Type of Plot", ["area", "bar", "line", "hist", "box", "kde", "pie"])

        if type_of_plot in ["area", "hist", "kde"]:
            selected_columns_names = st.selectbox("Select Numerical Column for Plot", df.select_dtypes(include=["float", "int"]).columns)
        elif type_of_plot in ["bar", "line", "box", "pie"]:
            col1 = st.selectbox("Select 1st Column for Plot", df.columns)
            col2 = st.selectbox("Select 2nd Column for Plot", df.columns)

        if st.button("Generate Plot"):
            st.success(f"Generating Customizable Plot of {type_of_plot} for {selected_columns_names if type_of_plot in ['area', 'hist', 'kde'] else [col1, col2]}")

            if type_of_plot == 'area':
                if pd.api.types.is_numeric_dtype(df[selected_columns_names]):
                    st.area_chart(df[selected_columns_names])
                else:
                    st.error("For an area chart, the selected column must be numerical.")

            elif type_of_plot == 'bar':
                if df[col1].dtype == 'object' and pd.api.types.is_numeric_dtype(df[col2]):
                    fig = px.bar(df, x=col1, y=col2, title=f"Bar Chart of {col2} by {col1}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("For a bar chart, the 1st column must be categorical and the 2nd column must be numerical.")

            elif type_of_plot == 'line':
                if pd.api.types.is_numeric_dtype(df[col2]):
                    fig = px.line(df, x=col1, y=col2, title=f"Line Chart of {col2} over {col1}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("For a line chart, the 2nd column must be numerical.")

            elif type_of_plot == 'hist':
                if pd.api.types.is_numeric_dtype(df[selected_columns_names]):
                    fig = px.histogram(df, x=selected_columns_names, title=f"Histogram of {selected_columns_names}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("For a histogram, the selected column must be numerical.")

            elif type_of_plot == 'box':
                if df[col1].dtype == 'object' and pd.api.types.is_numeric_dtype(df[col2]):
                    fig = px.box(df, x=col1, y=col2, title=f"Box Plot of {col2} by {col1}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("For a box plot, the 1st column must be categorical and the 2nd column must be numerical.")

            elif type_of_plot == 'kde':
                if pd.api.types.is_numeric_dtype(df[selected_columns_names]):
                    fig = px.density_contour(df, x=selected_columns_names, title=f"KDE Plot of {selected_columns_names}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("For a KDE plot, the selected column must be numerical.")

            elif type_of_plot == 'pie':
                if df[col1].dtype == 'object' and pd.api.types.is_numeric_dtype(df[col2]):
                    fig = px.pie(df, names=col1, values=col2, title=f"Pie Chart of {col2} by {col1}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("For a pie chart, the 1st column must be categorical and the 2nd column must be numerical.")

    elif task == "Data Profiling":
        st.subheader("Data Profiling")
        st.info("Data profiling is a process used to examine and analyze data from existing databases to understand its structure, content, and interrelationships.\n- Overview\n- Variables\n- Interactions\n- Correlations\n- Missing values\n- A sample of your data.\n")
        if st.button("Generate report"):
            with st.spinner("Creating Profile. May take a while..."):
                profile = ProfileReport(df, title="Data Profile")
                profile.config.html.minify_html = False
                profile.to_file(output_file="data_profile.html")
                st.success("Profile Report Generated")
                st.balloons()  # Show balloons animation
                st.markdown("View your data profile from the link below:")
                st.markdown(get_table_download_link(df), unsafe_allow_html=True)
                
                with open("data_profile.html", "rb") as file:
                    btn = st.download_button(
                        label="Download data profile",
                        data=file,
                        file_name="data_profile.html",
                        mime="text/html"
                    )
        
    elif task == "Model Building":
            st.subheader("Model Building")
            model_type = st.selectbox("Select a Model", ["Logistic Regression", "Decision Tree", "Random Forest", "Naive Bayes", "XGBoost"])
            if st.button("Train and Evaluate Model"):
                data_model = Data_Base_Modelling()
                X = df.drop("target", axis=1)
                y = df["target"]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
                if model_type == "Logistic Regression":
                    st.text(data_model.Logistic_Regression(X_train, y_train, X_test, y_test))
                elif model_type == "Decision Tree":
                    st.text(data_model.Decision_Tree(X_train, y_train, X_test, y_test))
                elif model_type == "Random Forest":
                    st.text(data_model.RandomForest(X_train, y_train, X_test, y_test))
                elif model_type == "Naive Bayes":
                    st.text(data_model.naive_bayes(X_train, y_train, X_test, y_test))
                elif model_type == "XGBoost":
                    st.text(data_model.XGb_classifier(X_train, y_train, X_test, y_test))

