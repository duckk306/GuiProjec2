import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


# Using menu
st.title("Trung Tâm Tin Học")
st.image("xe_may_cu.jpg", caption="Xe máy cũ")
# tạo dataframe mẫu, có 3 cột: Thương hiệu, số lượng xe, Giá trung bình
data = {
    'Thương hiệu': ['Honda', 'Yamaha', 'Suzuki', 'Piaggio', 'SYM'],
    'Số lượng xe': [150, 120, 90, 60, 80],
    'Giá trung bình (triệu VND)': [15.5, 14.0, 13.5, 16.0, 12.5]
}
df = pd.DataFrame(data)
st.subheader("Dữ liệu xe máy cũ")
st.dataframe(df)


# Vẽ biểu đồ số lượng xe theo thương hiệu
st.subheader("Biểu đồ số lượng xe theo thương hiệu")
fig, ax = plt.subplots()
sns.barplot(x='Thương hiệu', y='Số lượng xe', data=df, ax=ax)
st.pyplot(fig)

st.image("thong_ke.png", caption="Thong ke xe may cu")