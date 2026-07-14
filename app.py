import streamlit as st
import pandas as pd
import pickle
import base64
import requests


st.set_page_config(
    page_title="Music Mental Health NLP",
    page_icon="🎵",
    layout="wide"
)


# -----------------------------
# Background Image
# -----------------------------

def add_bg(image_url):

    try:

        response = requests.get(
            image_url,
            timeout=10
        )

        image_bytes = response.content

        encoded = base64.b64encode(
            image_bytes
        ).decode()


        return f"""
        <style>

        .stApp {{
            background-image:
            url("data:image/png;base64,{encoded}");

            background-size:cover;
            background-position:center;
            background-attachment:fixed;
        }}


        .block-container {{
            background-color:
            rgba(255,255,255,0.85);

            padding:2rem;
            border-radius:20px;
        }}

        </style>
        """

    except Exception:

        return ""



st.markdown(
    add_bg(
        "https://static.vecteezy.com/system/resources/previews/024/029/748/original/music-band-clipart-transparent-background-free-png.png"
    ),
    unsafe_allow_html=True
)



# -----------------------------
# CSS
# -----------------------------

st.markdown(
"""
<style>

h1,h2,h3,h4,h5,h6,p,label,span,div{
    color:black !important;
}


[data-testid="stSidebar"]{
    background:black !important;
}


[data-testid="stSidebar"] *{
    color:white !important;
}


.stButton button{

    background:white !important;
    color:black !important;
    border:2px solid black;
    border-radius:12px;
    font-weight:bold;

}


.stTextArea textarea{

    background:white !important;
    color:black !important;

}

</style>
""",
unsafe_allow_html=True
)



# -----------------------------
# Load Model
# -----------------------------

@st.cache_resource
def load_model():

    try:

        df = pickle.load(
            open("music_data.pkl","rb")
        )

        tfidf = pickle.load(
            open("tfidf.pkl","rb")
        )

        model = pickle.load(
            open("nlp_model.pkl","rb")
        )


        return df,tfidf,model


    except Exception as e:

        st.error(
            f"Model Loading Error: {e}"
        )

        st.stop()



df, tfidf, model = load_model()



# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title(
    "🎵 Music NLP Analyzer"
)


st.sidebar.markdown(
"""
## Project

Music Impact on Mental Health


## Techniques Used

✔ Data Processing

✔ NLP Cleaning

✔ TF-IDF

✔ Machine Learning

"""
)



# -----------------------------
# Main Page
# -----------------------------

st.title(
    "🎵 Music Mental Health Analysis"
)


st.write(
"NLP Based Mental Health Prediction System"
)


st.divider()


user_text = st.text_area(
    "Enter your feeling about music",
    placeholder="Example: Music helps me relax and reduce stress"
)



if st.button("✨ Analyze"):


    if user_text.strip()=="":


        st.warning(
            "Please enter text"
        )


    else:


        vector = tfidf.transform(
            [user_text]
        )


        prediction = model.predict(
            vector
        )


        result = prediction[0]


        st.success(
            "Analysis Completed Successfully 🎉"
        )


        st.subheader(
            "Mental Health Status"
        )


        if result=="High":

            st.error(result)


        elif result=="Medium":

            st.warning(result)


        else:

            st.success(result)



st.divider()


st.subheader(
    "Dataset Preview"
)


st.dataframe(
    df.head(10),
    use_container_width=True
)



st.markdown(
"<p style='color:black'>🎼 Powered by NLP + TF-IDF + Machine Learning</p>",
unsafe_allow_html=True
)
