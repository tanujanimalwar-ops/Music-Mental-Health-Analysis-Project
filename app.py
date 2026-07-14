import streamlit as st
import pandas as pd
import pickle
import base64
import numpy
import sys


# numpy pickle compatibility fix
sys.modules['numpy._core.numeric'] = numpy.core.numeric


st.set_page_config(
    page_title="Music Mental Health NLP",
    page_icon="🎵",
    layout="wide"
)



# -----------------------------
# Background Image
# -----------------------------

def add_bg(https://www.bing.com/videos/riverview/relatedvideo?q=music+images+png&&mid=D1B2E7D16B6A36CBCF1AD1B2E7D16B6A36CBCF1A&churl=https%3a%2f%2fwww.youtube.com%2fchannel%2fUCsKSB0ApOWdeX2WkwF24MyA&FORM=VCGVRP):

    try:

        with open(image_path, "rb") as img:

            encoded = base64.b64encode(
                img.read()
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
    add_bg("image/music.png"),
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
            open(
                "music_data.pkl",
                "rb"
            )
        )


        tfidf = pickle.load(
            open(
                "tfidf.pkl",
                "rb"
            )
        )


        model = pickle.load(
            open(
                "nlp_model.pkl",
                "rb"
            )
        )


        return df,tfidf,model


    except Exception as e:

        st.error(
            f"Model Loading Error: {e}"
        )

        st.stop()



df,tfidf,model = load_model()



# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title(
    "🎵 Music NLP Analyzer"
)


st.sidebar.markdown(
"""
## 🎵 Project

Music Impact on Mental Health


## 🛠 Technologies Used


✔ Python

✔ Streamlit

✔ Pandas

✔ NumPy

✔ NLP

✔ TF-IDF

✔ Scikit-Learn

✔ Machine Learning

✔ Pickle Model


## Features

✔ Mental Health Prediction

✔ Song Recommendation

✔ Dataset Preview

✔ Interactive UI

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
    placeholder=
    "Example: Music helps me relax and reduce stress"
)



# -----------------------------
# Prediction
# -----------------------------

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


        # Balloons
        st.balloons()



        st.subheader(
            "🧠 Mental Health Status"
        )



        if result=="High":

            st.error(result)


        elif result=="Medium":

            st.warning(result)


        else:

            st.success(result)



        # -----------------------------
        # Song Recommendation
        # -----------------------------

        songs = {


        "High":[

        "🎵 Weightless - Marconi Union",
        "🎵 Fix You - Coldplay",
        "🎵 Let It Be - The Beatles",
        "🎵 Someone Like You - Adele"

        ],


        "Medium":[

        "🎵 Perfect - Ed Sheeran",
        "🎵 Photograph - Ed Sheeran",
        "🎵 Counting Stars - OneRepublic",
        "🎵 A Thousand Years - Christina Perri"

        ],


        "Low":[

        "🎵 Happy - Pharrell Williams",
        "🎵 Good Life - OneRepublic",
        "🎵 Don't Stop Me Now - Queen",
        "🎵 On Top Of The World - Imagine Dragons"

        ]

        }



        st.subheader(
            "🎧 Recommended Songs"
        )


        for song in songs[result]:

            st.write(song)



# -----------------------------
# Dataset
# -----------------------------

st.divider()


st.subheader(
    "📊 Dataset Preview"
)


st.dataframe(
    df.head(10),
    use_container_width=True
)



# -----------------------------
# Footer
# -----------------------------

st.markdown(
"""
<p style='color:black;font-size:14px;'>
🎼 Powered by NLP + TF-IDF + Machine Learning
</p>
""",
unsafe_allow_html=True
)
