from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Fiverr Market Gap",
    page_icon="📊",
    layout="wide",
)


ROOT = Path(__file__).resolve().parents[1]

CLEAN_DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "fiverr_gigs_cleaned.csv"
)

CLUSTERED_DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "fiverr_gigs_with_clusters.csv"
)

MARKET_GAP_PATH = (
    ROOT
    / "data"
    / "processed"
    / "market_gap_summary.csv"
)

MODEL_PATH = (
    ROOT
    / "models"
    / "best_demand_model.joblib"
)

VECTORIZER_PATH = (
    ROOT
    / "models"
    / "cluster_vectorizer.joblib"
)

KMEANS_PATH = (
    ROOT
    / "models"
    / "kmeans_model.joblib"
)

METRICS_PATH = (
    ROOT
    / "reports"
    / "model_metrics_test.csv"
)

TEST_PREDICTION_PATH = (
    ROOT
    / "reports"
    / "test_predictions.csv"
)


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


@st.cache_resource
def load_artifacts():
    demand_model = joblib.load(
        MODEL_PATH
    )

    cluster_vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    kmeans_model = joblib.load(
        KMEANS_PATH
    )

    return (
        demand_model,
        cluster_vectorizer,
        kmeans_model,
    )


def clean_title(title):
    title = str(title).lower()

    title = re.sub(
        r"[^a-z0-9\s]",
        " ",
        title,
    )

    title = re.sub(
        r"\s+",
        " ",
        title,
    )

    return title.strip()


df = load_csv(
    CLEAN_DATA_PATH
)

clustered_df = load_csv(
    CLUSTERED_DATA_PATH
)

gap_summary = load_csv(
    MARKET_GAP_PATH
)

metrics_df = load_csv(
    METRICS_PATH
)

test_predictions = load_csv(
    TEST_PREDICTION_PATH
)

(
    demand_model,
    cluster_vectorizer,
    kmeans_model,
) = load_artifacts()


st.title(
    "📊 Fiverr Market Gap Analysis"
)

st.caption(
    "Analisis niche, kompetisi, harga, dan prediksi "
    "potensi permintaan gig Fiverr."
)


tabs = st.tabs(
    [
        "Overview",
        "EDA",
        "Market Gap",
        "Demand Predictor",
        "Model Evaluation",
        "Documentation",
    ]
)


with tabs[0]:
    st.header("Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Jumlah Gig",
        f"{len(df):,}",
    )

    col2.metric(
        "Jumlah Niche",
        f"{gap_summary['cluster'].nunique()}",
    )

    col3.metric(
        "Median Harga",
        f"PKR {df['price_pkr'].median():,.0f}",
    )

    col4.metric(
        "Gig dengan Review",
        f"{df['review_count'].notna().sum():,}",
    )

    st.markdown(
        """
        Aplikasi ini menggunakan judul gig, harga, seller level,
        dan karakteristik teks untuk memprediksi potensi jumlah
        review. Judul gig juga dikelompokkan menjadi niche layanan
        menggunakan TF-IDF dan K-Means.

        Jumlah review digunakan sebagai proxy permintaan, bukan
        jumlah order sebenarnya.
        """
    )

    st.subheader("Top Market Gap")

    st.dataframe(
        gap_summary[
            [
                "market_gap_rank",
                "niche_name",
                "competition_count",
                "predicted_demand_median",
                "market_gap_score",
            ]
        ].head(10),
        use_container_width=True,
    )


with tabs[1]:
    st.header("Exploratory Data Analysis")

    chart_1 = px.histogram(
        df,
        x="log_price_pkr",
        nbins=30,
        title="Distribusi Log Harga",
    )

    st.plotly_chart(
        chart_1,
        use_container_width=True,
    )

    seller_count = (
        df["seller_level"]
        .value_counts()
        .reset_index()
    )

    seller_count.columns = [
        "seller_level",
        "jumlah_gig",
    ]

    chart_2 = px.bar(
        seller_count,
        x="seller_level",
        y="jumlah_gig",
        title="Jumlah Gig Berdasarkan Seller Level",
    )

    st.plotly_chart(
        chart_2,
        use_container_width=True,
    )

    review_data = df.dropna(
        subset=["review_count"]
    )

    chart_3 = px.scatter(
        review_data,
        x="log_price_pkr",
        y="log_review_count",
        color="seller_level",
        hover_data=["gig_title"],
        title="Harga dan Jumlah Review",
    )

    st.plotly_chart(
        chart_3,
        use_container_width=True,
    )


with tabs[2]:
    st.header("Market Gap Analysis")

    st.dataframe(
        gap_summary,
        use_container_width=True,
    )

    chart_gap = px.bar(
        gap_summary.sort_values(
            "market_gap_score",
            ascending=True,
        ),
        x="market_gap_score",
        y="niche_name",
        orientation="h",
        title="Ranking Market Gap",
    )

    st.plotly_chart(
        chart_gap,
        use_container_width=True,
    )

    chart_map = px.scatter(
        gap_summary,
        x="competition_count",
        y="predicted_demand_median",
        size="market_gap_score",
        color="market_gap_score",
        hover_name="niche_name",
        title="Demand vs Competition",
    )

    st.plotly_chart(
        chart_map,
        use_container_width=True,
    )


with tabs[3]:
    st.header("Gig Demand Predictor")

    title_input = st.text_input(
        "Judul Gig",
        value="I will create a machine learning model",
    )

    price_input = st.number_input(
        "Harga dalam PKR",
        min_value=1,
        value=5000,
        step=500,
    )

    seller_input = st.selectbox(
        "Seller Level",
        [
            "Unknown",
            "Level 1 Seller",
            "Level 2 Seller",
            "Top Rated Seller",
        ],
    )

    if st.button(
        "Prediksi Potensi Gig",
        type="primary",
    ):
        cleaned_title = clean_title(
            title_input
        )

        input_df = pd.DataFrame(
            [
                {
                    "title_clean": cleaned_title,
                    "price_pkr": price_input,
                    "seller_level": seller_input,
                    "title_word_count": len(
                        cleaned_title.split()
                    ),
                }
            ]
        )

        prediction_log = (
            demand_model.predict(
                input_df
            )[0]
        )

        predicted_review = max(
            0,
            np.expm1(
                prediction_log
            ),
        )

        title_vector = (
            cluster_vectorizer
            .transform(
                [cleaned_title]
            )
        )

        cluster_id = int(
            kmeans_model.predict(
                title_vector
            )[0]
        )

        niche_row = gap_summary[
            gap_summary["cluster"]
            == cluster_id
        ]

        st.success(
            f"Prediksi potensi review: "
            f"{predicted_review:,.0f}"
        )

        if not niche_row.empty:
            niche = niche_row.iloc[0]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Niche",
                niche["niche_name"],
            )

            col2.metric(
                "Market Gap Score",
                f"{niche['market_gap_score']:.2f}",
            )

            col3.metric(
                "Competition Count",
                int(
                    niche[
                        "competition_count"
                    ]
                ),
            )

        st.warning(
            "Prediksi merupakan estimasi berdasarkan data historis "
            "dan bukan jaminan gig akan memperoleh jumlah review tersebut."
        )


with tabs[4]:
    st.header("Model Evaluation")

    st.dataframe(
        metrics_df,
        use_container_width=True,
    )

    prediction_chart = px.scatter(
        test_predictions,
        x="actual_review",
        y="predicted_review",
        title="Actual vs Predicted Review",
    )

    st.plotly_chart(
        prediction_chart,
        use_container_width=True,
    )

    residual_chart = px.scatter(
        test_predictions,
        x="predicted_review",
        y="residual",
        title="Residual Plot",
    )

    residual_chart.add_hline(
        y=0
    )

    st.plotly_chart(
        residual_chart,
        use_container_width=True,
    )


with tabs[5]:
    st.header("Documentation")

    st.markdown(
        """
        ## Metodologi

        1. Data cleaning dan exploratory data analysis.
        2. TF-IDF untuk mengubah judul gig menjadi fitur numerik.
        3. Ridge Regression dan XGBoost Regressor.
        4. Hyperparameter tuning menggunakan Grid Search dan Random Search.
        5. K-Means untuk mengelompokkan gig menjadi niche.
        6. Market Gap Score berdasarkan demand dan competition.

        ## Keterbatasan

        - Review bukan jumlah order sebenarnya.
        - Dataset tidak mempunyai umur gig.
        - Dataset tidak mempunyai impression atau click-through rate.
        - Competition dihitung hanya dari sampel dataset.
        - Notasi review seperti `1k+` diperlakukan sebagai 1.000.
        - Market Gap Score adalah indikator analitis, bukan metrik resmi Fiverr.
        """
    )