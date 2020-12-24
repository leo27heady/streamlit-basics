# import os
import streamlit as st
import pandas as pd

# import numpy as np
# import matplotlib.pyplot as plt
import seaborn as sns


def load_data(data_path):
    return pd.read_csv(data_path)


def title_write():
    st.title("Heart Disease")
    st.write(
        """
    This dataset gives a number of feachures along with a
    target condition of having or not having heart disease.
    \n
    Heart disease dataset feachures explain:
    * `age`: The person's age in years
    * `sex`: The person's sex (1: male, 0: female)
    * `cp`: The chest pain experienced (1: typical angina,
    2: atypical angina, 3: non-anginal pain, 4: asymptomatic)
    * `trestbps`: The person's resting blood pressure
    (mm Hg on admission to the hospital)
    * `chol`: The person's cholesterol measurement in mg/dl
    * `fbs`: The person's fasting blood sugar (> 120 mg/dl, 1: true; 0: false)
    * `restecg`: Resting electrocardiographic measurement (0: normal,
    1: having ST-T wave abnormality, 2: showing probable
    or definite left ventricular hypertrophy by Estes' criteria)
    * `thalach`: The person's maximum heart rate achieved
    * `exang`: Exercise induced angina (1: yes; 0: no)
    * `oldpeak`: ST depression induced by exercise relative to rest
    ('ST' relates to positions on the ECG plot. See more here)
    * `slope`: the slope of the peak exercise ST segment
    (1: upsloping, 2: flat, 3: downsloping)
    * `ca`: The number of major vessels (0-3)
    * `thal`: A blood disorder called thalassemia
    (3 = normal; 6 = fixed defect; 7 = reversable defect)
    * `target`: Heart disease (0: no, 1: yes)
    """
    )


def explore_write(df):
    st.write(
        """
    # Explore dataset
    ## First let's check how the dataset looks like and how big it is
    """
    )
    st.dataframe(df)

    st.write("Shape of the dataset:", df.shape)

    st.write(
        """
    ## Describe our data
    and look at the calculated parameters for each feachure
    * `count`: number of feature values
    * `mean`: average feature value
    * `std`: measure of the amount of variation of a set of values
    * `min`: minimum feature value
    * `25%`: found by ordering all data points
    and picking out the one in the 25% of data
    * `50%`: the middle number.
    Found by ordering all data points and picking out the one in the middle
    * `75%`: found by ordering all data points
    and picking out the one in the 75% of data
    * `max`: maximum feature value
    """
    )
    st.dataframe(df.describe())

    st.write(
        """
    ## Dataset distribution
    As you can see from the plot, the dataset is not balanced,
    but the imbalance is not critical.
    """
    )

    st.bar_chart(df.target.value_counts())
    st.write(
        "Dataset disbalance: ",
        df.target.value_counts()[0] / df.target.value_counts()[1],
    )


def analysis_write(df):
    st.write(
        """
        # Dataset analysis
    """
    )

    age_analysis_write(df)
    sex_analysis_write(df)
    cp_analysis_write(df)


def age_analysis_write(df):
    st.write(
        """
    ## Сheck how `age` feachure affects on the target value
    ### First let's see how the `age` is distributed in the dataset
    """
    )

    st.bar_chart(df.age.value_counts())
    st.write("`age` mean value:", df.describe().age["mean"])
    st.write("`age` median value:", df.describe().age["25%"])

    st.write(
        """
    ### Build a distribution graph for `age` depending on target value
    Looking at the graph, we can come to the conclusion,
    that those people who have heart disease do not live long
    """
    )
    figure = sns.displot(
        df[["age", "target"]], x="age", hue="target", kind="kde", fill=True
    )
    st.pyplot(figure)


def sex_analysis_write(df):
    st.write(
        """
    ## Сheck how `sex` feachure affects on the target value
    ### First let's see how the `sex` is distributed in the dataset
    """
    )
    st.bar_chart(df.sex.value_counts())
    st.write(
        "Feachure `sex` disbalance: ",
        df.sex.value_counts()[0] / df.sex.value_counts()[1],
    )
    st.write(
        """
    ### Build a distribution graph for `sex` depending on target value
    Looking at the graph, we can come to the conclusion,
    that those people who have heart disease do not live long
    """
    )
    figure = sns.catplot(
        data=df,
        kind="bar",
        x="target",
        y="age",
        hue="sex",
        ci="sd",
        palette="dark",
        alpha=0.6,
        height=6,
    )
    figure.despine(left=True)
    # figure.set_axis_labels("Sex", "Age")

    # figure = sns.displot(
    #     df[["sex", "target"]], x="sex", hue="target", kind="kde", fill=True
    # )
    st.pyplot(figure)


def cp_analysis_write(df):
    st.write(
        """
    ## Сheck how `cp` feachure affects on the target value
    ### First let's see how the `cp` is distributed in the dataset
    """
    )
    st.bar_chart(df.cp.value_counts())
    st.write(
        """
    ### Build a distribution graph for `cp` depending on target value
    Looking at the graph, we can come to the conclusion,
    that those people who have atypical angina(`cp`: 2) also have heart disease
    """
    )
    figure = sns.catplot(
        data=df,
        kind="bar",
        x="target",
        y="age",
        hue="cp",
        ci="sd",
        palette="dark",
        alpha=0.6,
        height=6,
    )
    figure.despine(left=True)
    st.pyplot(figure)


if __name__ == "__main__":
    df = load_data(data_path="./resources/heart.csv")
    title_write()
    explore_write(df)
    analysis_write(df)
