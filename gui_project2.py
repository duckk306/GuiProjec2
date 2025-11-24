# gui_project2.py
# File path: GUI/gui_project2.py
"""
Streamlit app:
- Đề xuất xe máy dựa trên nội dung, phân cụm xe máy (file upload + manual check)
Requirements:
- utils
- consine_sim_matrix.pkl (chứa model)
- xe_may_cu.jpg
"""

from io import BytesIO
import streamlit as st
from streamlit_option_menu import option_menu
#from streamlit_navigation_bar import st_navbar

import pandas as pd
import numpy as np
import joblib


# ================== CONFIG ==================
st.image("xe_may_cu.jpg", use_container_width=True)
st.title("Welcome Home!")
st.markdown("### Gợi ý xe máy tương tự & Dự đoán giá xe máy cũ")
st.markdown("Upload file `data_motobikes.xlsx` hoặc nhập tay để dùng model đã train.")


# ================== NAVIGATION ==================
pages = {'Giới thiệu':'pages/intro.py',
         'Gợi ý xe máy tương tự':'pages/suggest_bikes.py',
         'Dự đoán giá xe máy cũ':'pages/predictprice_byclustering.py',
         'Thông tin tác giả':'pages/author.py'}

page_list = ["Home"] + list(pages.keys())
st.sidebar.header("Navigation Menu")
st.sidebar.page_link("gui_project2.py", label="Home", icon="🏠")
st.sidebar.page_link("pages/intro.py", label="Giới thiệu", icon="📃")
st.sidebar.page_link('pages/suggest_bikes.py', label='Gợi ý xe máy tương tự', icon="⭐")
st.sidebar.page_link('pages/predictprice_byclustering.py', label='Dự đoán giá xe máy cũ', icon="💵")
st.sidebar.page_link("pages/author.py", label="Thông tin tác giả", icon="ℹ️")



