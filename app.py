import streamlit as st
from models.risk_engine import calculate_risk
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os

from xai.shap_explainer import (
    explain_prediction
)

st.set_page_config(
    page_title="Mental Health AI",
    page_icon="🧠",
    layout="wide"
)

# =========================
# Custom styling
# =========================

st.markdown("""
<style>

.main{
    padding-top:1rem;
}

.stMetric{
    background-color:#1E1E1E;
    padding:10px;
    border-radius:10px;
    text-align:center;
}

</style>
""",unsafe_allow_html=True)


# =========================
# Header
# =========================

st.title(
    "🧠 Mental Health Risk Intelligence System"
)

st.markdown(
"""
AI-powered early warning system for emotional risk analysis
"""
)


# =========================
# Sidebar
# =========================

st.sidebar.header(
    "Settings"
)

show_emotions=st.sidebar.checkbox(
    "Show emotions",
    True
)

show_alerts=st.sidebar.checkbox(
    "Show critical alerts",
    True
)

show_xai=st.sidebar.checkbox(
    "Show Explainable AI",
    True
)

show_trend=st.sidebar.checkbox(
    "Show trend graph",
    True
)


st.sidebar.markdown("---")

st.sidebar.info(
"""
This system estimates emotional
risk indicators and is not a
medical diagnosis tool.
"""
)


# =========================
# User Input
# =========================

text=st.text_area(
    "Enter text",
    height=150,
    placeholder="Type text here..."
)


# =========================
# Analyze Button
# =========================

if st.button(
    "Analyze",
    use_container_width=True
):

    if text:

        result=calculate_risk(text)

        risk=result["risk_score"]

        # =====================
        # Save history
        # =====================

        new_record=pd.DataFrame({

            "text":[text],

            "risk_score":[risk]

        })

        history_path="history/history.csv"

        if os.path.exists(
            history_path
        ):

            history=pd.read_csv(
                history_path
            )

            history=pd.concat(

                [
                    history,
                    new_record
                ],

                ignore_index=True
            )

        else:

            history=new_record


        history.to_csv(

            history_path,

            index=False
        )



        st.divider()


        # =====================
        # Metrics
        # =====================

        col1,col2,col3=st.columns(3)

        with col1:

            st.metric(
                "Risk Score",
                f"{risk}%"
            )

        with col2:

            st.metric(
                "Risk Level",
                result[
                    "risk_level"
                ]
            )

        with col3:

            st.metric(
                "Critical Indicators",
                len(
                    result[
                        "critical_matches"
                    ]
                )
            )


        # =====================
        # Gauge Chart
        # =====================

        gauge=go.Figure(

            go.Indicator(

                mode="gauge+number",

                value=risk,

                title={
                    "text":"Risk Score"
                },

                gauge={

                    "axis":{
                        "range":[0,100]
                    },

                    "steps":[

                        {
                            "range":[0,30],
                            "color":"green"
                        },

                        {
                            "range":[31,70],
                            "color":"orange"
                        },

                        {
                            "range":[71,100],
                            "color":"red"
                        }
                    ]
                }

            )
        )

        st.plotly_chart(
            gauge,
            use_container_width=True,
            key="risk_gauge"
        )


        # =====================
        # Critical Alert
        # =====================

        if (

            show_alerts
            and
            result[
                "critical_matches"
            ]

        ):

            st.error(

                "🚨 Potential severe emotional distress detected"

            )

            st.markdown(
                "**Critical indicators detected:**"
            )

            for item in result[
                "critical_matches"
            ]:

                st.write(
                    f"• {item}"
                )


        # =====================
        # Emotion Distribution
        # =====================

        if show_emotions:

            st.subheader(
                "Emotion Distribution"
            )

            emotions=result[
                "emotions"
            ]

            emotion_df=pd.DataFrame({

                "Emotion":
                emotions.keys(),

                "Score":
                emotions.values()

            })

            fig=px.bar(

                emotion_df,

                x="Emotion",

                y="Score",

                text="Score"

            )

            fig.update_layout(

                xaxis_title="",

                yaxis_title=
                "Confidence Score"

            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="emotion_chart"
            )


        # =====================
        # AI Explanation
        # =====================

        st.subheader(
            "🧠 AI Explanation"
        )

        st.info(
            f"""

            **Sentiment detected:**  
            {result['sentiment']['label']}

            **Detected emotional indicators:**  
            {", ".join(result['emotions'].keys())}

            **Critical phrases:**  
            {", ".join(result['critical_matches'])}

            """
        )

        # =====================
        # SHAP Explainability
        # =====================

        if show_xai:
        
            st.subheader(
                "🔍 Prediction Drivers"
            )

            explanation=explain_prediction(
                text
            )

            explanation_df = pd.DataFrame(
                explanation
            )

            st.dataframe(
                explanation_df
            )

            chart=px.bar(
            
                explanation_df,

                x="Impact",

                y="Feature",

                orientation="h",

                title=
                "Detected Risk Indicators",

                text="Impact"

            )

            chart.update_layout(         
                xaxis=dict(
                    range=[0,1]
                ),

                height=350,

                yaxis_title="",
                xaxis_title="Contribution"
            )

            chart.update_traces(          
                textposition="outside"          
            )

            st.plotly_chart(
            
                chart,

                use_container_width=True,

                key="shap_chart"

            )

        # =====================
        # Trend Graph
        # =====================

        if show_trend:

            st.subheader(
                "📈 Risk Trend Over Time"
            )

            history=pd.read_csv(
                history_path
            )

            recent_history=history.tail(10)

            trend=go.Figure()

            trend.add_trace(
            
                go.Scatter(
                
                    x=recent_history.index,

                    y=recent_history["risk_score"],

                    mode="lines+markers",

                    name="Risk Score"

                )

            )

            # low risk zone

            trend.add_hrect(
            
                y0=0,
                y1=30,

                fillcolor="green",

                opacity=0.15,

                line_width=0
            )

            # moderate risk zone 

            trend.add_hrect(
            
                y0=31,
                y1=70,

                fillcolor="orange",

                opacity=0.15,

                line_width=0
            )

            #high risk zone
            
            trend.add_hrect(
            
                y0=71,
                y1=100,

                fillcolor="red",

                opacity=0.15,

                line_width=0
            )

            trend.update_layout(
            
                title="Risk Trend Over Recent Analyses",

                xaxis_title="Analysis Number",

                yaxis_title="Risk Score",

                yaxis_range=[0,100]

            )

            st.plotly_chart(
                trend,
                use_container_width=True,
                key="trend_chart"
            )