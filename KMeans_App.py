from sklearn.metrics import silhouette_score
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import ast

def load_data():
  iris = load_iris()

  dta = pd.concat([pd.DataFrame(iris['data']),
                  pd.DataFrame(iris['target'])],
                  axis = 1)
  dta.columns = ['sepal_length', 'sepal_width',
                'petal_length', 'petal_width', 'target']

  st.session_state.df = dta
  return 

def create_dataframe_from_columns():
    if 'df' not in st.session_state:
       st.write('Please Load the Dataset.')
       return
    else:
      columns_input = st.session_state.options
      # Select the specified columns from the original dataframe
      selected_columns = [st.session_state.df.columns[int(i - 1)] for i in columns_input]

      # Create a new dataframe containing only the selected columns
      new_dataframe = st.session_state.df[selected_columns]

      n_list = list(range(1,10))
      inertias = []
      sils = []
      for n in n_list:
        kmeans = KMeans(n_clusters=n, random_state=0, n_init=10).fit(new_dataframe)
        labels = kmeans.fit_predict(new_dataframe)
        inertias.append(kmeans.inertia_)
        if n < 2:
          sils.append(0)
        else:
          silhouette_avg = silhouette_score(new_dataframe, labels)
          sils.append(silhouette_avg)
      st.session_state.pdf1 = pd.DataFrame({'K': n_list, 'Inertia': inertias})
      st.session_state.pdf2 = pd.DataFrame({'K': n_list, 'Silhouette': sils})
      st.session_state.ndf = new_dataframe
      st.session_state.plotted = True
    return new_dataframe

def output_model():
  st.session_state.kmeans = KMeans(n_clusters=st.session_state.k, 
                                   random_state=0, 
                                   n_init=10).fit(st.session_state.ndf)
  return

# create a sidebar to hold the input widgets
with st.sidebar:

  # Add the button to load the Iris dataset: will change to upload in the future
  st.button('Load Dataset', 
            on_click=load_data, 
            help=None, 
            type="secondary", 
            disabled=False, 
            use_container_width=False)

  if 'df' in st.session_state:
    # Add the input textbox for the columns containing data
      # Add the input textbox for the columns containing data
    st.session_state.options = st.multiselect(
      'Columns numbers containing clustering data',
      list(range(1,len(st.session_state.df.columns) + 1)))
    
    # Add the button to generate the elbow plot
    st.button('Generate Plots', 
              on_click=create_dataframe_from_columns, 
              help=None, 
              type="secondary", 
              disabled=False, 
              use_container_width=False)

  if 'plotted' in st.session_state:
    st.session_state.k = st.slider('Select Ideal K', 
                                   min_value=1, 
                                   max_value=10, 
                                   value=None, 
                                   step=1, 
                                   label_visibility="visible")
    st.button('Generate Model', 
              on_click=output_model, 
              help=None, 
              type="secondary", 
              disabled=False, 
              use_container_width=False)
    
  if 'kmeans' in st.session_state:
    st.download_button(
                      "Download Model",
                      data=pickle.dumps(st.session_state.kmeans),
                      file_name="model.pkl",
                      )

if 'df' in st.session_state and 'plotted' not in st.session_state:
  st.write('Full Dataset')
  st.dataframe(st.session_state.df)

if 'plotted' in st.session_state:
  # Add the charts and new dataframe
  st.write('Elbow Plot')
  st.session_state.line_chart = st.line_chart(st.session_state.pdf1, x = 'K', y = 'Inertia', use_container_width = True)
  st.write('Silhouette Plot')
  st.session_state.sil_chart = st.line_chart(st.session_state.pdf2, x = 'K', y = 'Silhouette', use_container_width = True)
  st.write('Selected Data')
  st.dataframe(st.session_state.ndf, use_container_width = True)
      