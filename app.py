import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(page_title="ClarifAI Dashboard", layout="wide")

# -----------------------------
# 1. LOAD DATA + MODEL
# -----------------------------
df = pd.read_csv("dashboard_scored.csv")
model = joblib.load("careflow_selected_model.joblib")
model_features = list(joblib.load("careflow_model_features.joblib"))

# -----------------------------
# 2. SIDEBAR FILTERS
# -----------------------------
st.title("ClarifAI")
st.subheader("High-Cost Risk Overview Dashboard")

st.sidebar.header("Filters")

region_options = ["All"] + sorted(df["region"].dropna().unique().tolist())
site_options = ["All"] + sorted(df["site_id"].dropna().unique().tolist())

selected_region = st.sidebar.selectbox("Region", region_options)
selected_site = st.sidebar.selectbox("Site", site_options)

# -----------------------------
# 3. FILTER DATA
# -----------------------------
filtered_df = df.copy()

if selected_region != "All":
    filtered_df = filtered_df[filtered_df["region"] == selected_region]

if selected_site != "All":
    filtered_df = filtered_df[filtered_df["site_id"] == selected_site]

# -----------------------------
# 4. KPI CARDS
# -----------------------------
total_pts = len(filtered_df)
high_cost_pct = (filtered_df["high_cost_flag"].mean() * 100) if total_pts > 0 else 0

col1, col2 = st.columns(2)
with col1:
    st.metric("Total Patients", total_pts)
with col2:
    st.metric("Percent High-Cost", f"{high_cost_pct:.1f}%")

st.caption(
    f"Current View: Region = {selected_region}, Site = {selected_site}, "
    f"N = {total_pts} Patients"
)

# -----------------------------
# 5. HELPER FUNCTION: CONTEXT CHARTS
# -----------------------------
def context_chart(data, col_name, label, color_scale):
    temp = data.copy()
    temp = temp[temp[col_name].notna()]

    stats = (
        temp.groupby(col_name)["high_cost_flag"]
        .agg(["mean", "count", "sum"])
        .reset_index()
        .rename(columns={
            "mean": "proportion_high_cost",
            "count": "sample_size",
            "sum": "high_cost_count"
        })
    )

    if col_name == "income_band":
        display_map = {
            "low": "Low",
            "medium": "Medium",
            "high": "High"
        }
        stats["display_label"] = (
            stats[col_name]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(display_map)
            .fillna(stats[col_name])
        )
        order = ["Low", "Medium", "High"]
        stats["display_label"] = pd.Categorical(
            stats["display_label"],
            categories=order,
            ordered=True
        )
        stats = stats.sort_values("display_label")
    else:
        stats["display_label"] = stats[col_name].astype(str).str.strip().str.title()
        stats = stats.sort_values("proportion_high_cost", ascending=True)

    x_max = max(0.35, stats["proportion_high_cost"].max() * 1.15) if len(stats) > 0 else 0.35

    fig = px.bar(
        stats,
        x="proportion_high_cost",
        y="display_label",
        orientation="h",
        labels={
            "display_label": label,
            "proportion_high_cost": "Proportion Of High-Cost Patients"
        },
        title=label,
        color="proportion_high_cost",
        color_continuous_scale=color_scale,
        custom_data=["sample_size", "high_cost_count"]
    )

    fig.update_traces(
        hovertemplate=
        f"{label}: %{{y}}<br>" +
        "Proportion High-Cost: %{x:.1%}<br>" +
        "Sample Size (N): %{customdata[0]}<br>" +
        "High-Cost Patients: %{customdata[1]}<extra></extra>"
    )

    fig.update_xaxes(
        tickformat=".0%",
        title="Proportion Of High-Cost Patients",
        range=[0, x_max]
    )
    fig.update_yaxes(title=label)
    fig.update_layout(
        template="plotly_white",
        coloraxis_showscale=False,
        title_font=dict(size=18),
        font=dict(size=13),
        height=420
    )
    return fig

# -----------------------------
# 6. HELPER FUNCTION: PREDICTOR CHARTS
# -----------------------------
def driver_chart(data, col_name, label, max_val, color_scale):
    temp = data.copy()
    temp[col_name] = temp[col_name].apply(
        lambda x: f"{int(max_val)}+" if x >= max_val else str(int(x))
    )

    stats = (
        temp.groupby(col_name)["high_cost_flag"]
        .agg(["mean", "count", "sum"])
        .reset_index()
        .rename(columns={
            "mean": "proportion_high_cost",
            "count": "sample_size",
            "sum": "high_cost_count"
        })
    )

    ordered_labels = [str(i) for i in range(max_val)] + [f"{max_val}+"]
    stats[col_name] = pd.Categorical(
        stats[col_name],
        categories=ordered_labels,
        ordered=True
    )
    stats = stats.sort_values(col_name)

    fig = px.bar(
        stats,
        x=col_name,
        y="proportion_high_cost",
        labels={
            col_name: label,
            "proportion_high_cost": "Proportion Of High-Cost Patients"
        },
        title=label,
        color="proportion_high_cost",
        color_continuous_scale=color_scale,
        custom_data=["sample_size", "high_cost_count"]
    )

    fig.update_traces(
        hovertemplate=
        f"{label}: %{{x}}<br>" +
        "Proportion High-Cost: %{y:.1%}<br>" +
        "Sample Size (N): %{customdata[0]}<br>" +
        "High-Cost Patients: %{customdata[1]}<extra></extra>"
    )

    fig.update_yaxes(tickformat=".0%", title="Proportion Of High-Cost Patients")
    fig.update_xaxes(title=label)
    fig.update_layout(
        template="plotly_white",
        coloraxis_showscale=False,
        title_font=dict(size=18),
        font=dict(size=13),
        height=420
    )
    return fig

# -----------------------------
# 7. DEMOGRAPHIC PATTERNS
# -----------------------------
st.write("### Demographic Patterns")
st.caption(
    "These charts show how high-cost prevalence differs across patient context and access groups within the current filtered population."
)

col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(
        context_chart(filtered_df, "income_band", "Income Band", "Tealgrn"),
        use_container_width=True
    )
with col4:
    st.plotly_chart(
        context_chart(filtered_df, "language_pref", "Language Preference", "PuBuGn"),
        use_container_width=True
    )

# -----------------------------
# 8. KEY PREDICTOR PATTERNS
# -----------------------------
st.write("### Key Predictor Patterns")
st.caption(
    "Bars show the proportion of high-cost patients within the currently filtered group. Hover to view sample size (N) and high-cost count for each bin."
)

col5, col6 = st.columns(2)
with col5:
    st.plotly_chart(
        driver_chart(filtered_df, "total_adverse_events", "Adverse Events", 5, "Blues"),
        use_container_width=True
    )
with col6:
    st.plotly_chart(
        driver_chart(filtered_df, "chronic_condition_count", "Chronic Condition Burden", 6, "Purples"),
        use_container_width=True
    )

col7, col8 = st.columns(2)
with col7:
    st.plotly_chart(
        driver_chart(filtered_df, "event_count_refill", "Refill Activity", 5, "Oranges"),
        use_container_width=True
    )
with col8:
    st.plotly_chart(
        driver_chart(filtered_df, "unique_drug_classes", "Medication Complexity", 6, "Reds"),
        use_container_width=True
    )

# -----------------------------
# 9. BASELINE ENROLLMENT RISK SCORE DISTRIBUTION
# -----------------------------
st.write("### Baseline Enrollment Risk Score Distribution")

high_cost_only_df = filtered_df[filtered_df["high_cost_flag"] == 1]

fig_risk = px.histogram(
    high_cost_only_df,
    x="risk_score_initial",
    nbins=20,
    title="Baseline Enrollment Risk Score Distribution",
    labels={
        "risk_score_initial": "Initial Risk Score",
        "count": "Number Of Patients"
    },
    color_discrete_sequence=["#1F4E79"]
)

fig_risk.update_layout(
    template="plotly_white",
    title_font=dict(size=18),
    font=dict(size=13),
    height=450
)
fig_risk.update_xaxes(title="Initial Risk Score")
fig_risk.update_yaxes(title="Number Of Patients")

st.plotly_chart(fig_risk, use_container_width=True)

# -----------------------------
# 10. PRIORITY PATIENT LIST
# -----------------------------
st.write("### Priority Patient List")
st.caption(
    "Showing the top 10 patients in the current filtered view, ranked by predicted high-cost risk from the final logistic regression model. Displayed predictors provide context only; the score reflects all predictors in the final model."
)

table_df = filtered_df.copy().sort_values(by="predicted_high_cost_risk", ascending=False)

display_df = table_df[
    [
        "patient_id",
        "predicted_high_cost_pct",
        "total_adverse_events",
        "chronic_condition_count",
        "event_count_refill",
        "unique_drug_classes",
        "risk_score_initial",
        "site_id",
        "region"
    ]
].head(10).copy()

display_df = display_df.rename(columns={
    "patient_id": "Patient ID",
    "predicted_high_cost_pct": "Predicted High-Cost Risk (%)",
    "total_adverse_events": "Total Adverse Events",
    "chronic_condition_count": "Chronic Condition Count",
    "event_count_refill": "Refill Event Count",
    "unique_drug_classes": "Unique Drug Classes",
    "risk_score_initial": "Initial Risk Score",
    "site_id": "Site",
    "region": "Region"
})

display_df["Predicted High-Cost Risk (%)"] = display_df["Predicted High-Cost Risk (%)"].round(1)
display_df["Initial Risk Score"] = display_df["Initial Risk Score"].round(1)

st.dataframe(display_df, use_container_width=True, hide_index=True)

# -----------------------------
# 11. WHAT-IF RISK EXPLORER
# -----------------------------
st.write("### Risk Explorer")
st.caption(
    "Choose a patient and test simple service-delivery scenarios. "
    "You can adjust wait time, shipment delays, and missed interactions to see how the model-predicted high-cost risk changes. "
    "Patient characteristics stay fixed in the background. This is for scenario exploration only."
)

if filtered_df.empty:
    st.warning("No patients available in the current filtered view.")
else:
    patient_options = (
        filtered_df.sort_values("predicted_high_cost_risk", ascending=False)["patient_id"]
        .astype(str)
        .tolist()
    )

    selected_patient_id = st.selectbox(
        "Select Patient ID",
        patient_options,
        key="what_if_patient"
    )

    selected_patient_row = filtered_df[
        filtered_df["patient_id"].astype(str) == selected_patient_id
    ].iloc[0].copy()

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.markdown("#### Edit Service Scenario")
        st.caption(
            f"Current profile: Site = {selected_patient_row['site_id']}, "
            f"Region = {selected_patient_row['region']}"
        )

        st.markdown("**Patient context stays fixed**")
        st.caption(
            f"Chronic conditions: {int(selected_patient_row['chronic_condition_count'])} | "
            f"Drug classes: {int(selected_patient_row['unique_drug_classes'])} | "
            f"Initial risk score: {float(selected_patient_row['risk_score_initial']):.1f}"
        )

        edited_wait_days = st.number_input(
            "Average Wait Days",
            min_value=0.0,
            value=float(selected_patient_row["avg_wait_days"]),
            step=0.1
        )

        edited_delayed_shipments = st.number_input(
            "Delayed Shipments Count",
            min_value=0,
            value=int(selected_patient_row["delayed_shipments_count"]),
            step=1
        )

        edited_missed_or_not_attended = st.number_input(
            "Missed / Not Attended Events",
            min_value=0,
            value=int(selected_patient_row["total_missed_or_not_attended"]),
            step=1
        )

    updated_patient_row = selected_patient_row.copy()
    updated_patient_row["avg_wait_days"] = edited_wait_days
    updated_patient_row["delayed_shipments_count"] = edited_delayed_shipments
    updated_patient_row["total_missed_or_not_attended"] = edited_missed_or_not_attended

    updated_input = pd.DataFrame([updated_patient_row])[model_features]
    updated_predicted_risk = float(model.predict_proba(updated_input)[:, 1][0])
    current_predicted_risk = float(selected_patient_row["predicted_high_cost_risk"])
    risk_change_points = (updated_predicted_risk - current_predicted_risk) * 100

    with right_col:
        st.markdown("#### Model Output")

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric(
                "Current Predicted Risk",
                f"{current_predicted_risk * 100:.1f}%"
            )
        with metric_col2:
            st.metric(
                "Updated Predicted Risk",
                f"{updated_predicted_risk * 100:.1f}%"
            )
        with metric_col3:
            st.metric(
                "Change",
                f"{risk_change_points:+.1f} pts"
            )

        comparison_df = pd.DataFrame({
            "Variable": [
                "Average Wait Days",
                "Delayed Shipments Count",
                "Missed / Not Attended Events"
            ],
            "Current Value": [
                round(float(selected_patient_row["avg_wait_days"]), 1),
                int(selected_patient_row["delayed_shipments_count"]),
                int(selected_patient_row["total_missed_or_not_attended"])
            ],
            "Updated Value": [
                round(float(edited_wait_days), 1),
                int(edited_delayed_shipments),
                int(edited_missed_or_not_attended)
            ]
        })

        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
