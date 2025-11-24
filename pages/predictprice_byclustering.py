import streamlit as st
import joblib
from joblib import load
import pickle
import pandas as pd

st.header("💵 Dự đoán giá xe máy cũ")

# ================== Feature lists ==================
# Default brands fallback (in case user doesn't upload a dataset)
BRANDS = ['Aprilia','Bmw','Bazan','Benelli','Brixton','Cr&S','Daelim','Detech','Ducati','Gpx','Halim',
          'Harley Davidson','Honda','Hyosung','Hãng Khác','Ktm','Kawasaki','Keeway','Kengo','Kymco',
          'Moto Guzzi','Nioshima','Peugeot','Piaggio','Rebelusa','Royal Enfield','Sym','Sachs','Sanda',
          'Suzuki','Taya','Triumph','Vento','Victory','Vinfast','Visitor','Yamaha']

# ================== FUNCTIONS ==================     
@st.cache_resource 
def load_pipeline(suggest_model='model/kmeans_k2.joblib',
                  vectorizer='model/onehot_encoder_clustering.joblib'):
    """
    Load cosine similarity matrix và vectorizer đã lưu trước đó.
    """
    cosine_matrix = joblib.load(suggest_model)
    vectorizer_loaded = joblib.load(vectorizer)
    print("Đã load cosine matrix & vectorizer thành công.")
    return cosine_matrix, vectorizer_loaded

# ================== NAVIGATION ==================
st.sidebar.header("Navigation Menu")
st.sidebar.page_link("gui_project2.py", label="Home", icon="🏠")
st.sidebar.page_link("pages/intro.py", label="Giới thiệu", icon="📃")
st.sidebar.page_link('pages/suggest_bikes.py', label='Gợi ý xe máy tương tự', icon="⭐")
st.sidebar.page_link('pages/predictprice_byclustering.py', label='Dự đoán giá xe máy cũ', icon="💵")
st.sidebar.page_link("pages/author.py", label="Thông tin tác giả", icon="ℹ️")


# ================== LOAD PIPELINE /  MODEL ==================
# 1. Load clustering model / vectorizer
with open('model/kmeans_k2.pkl', 'rb') as f:
    clustering_model = pickle.load(f)
with open('model/onehot_encoder_clustering.pkl', 'rb') as f:
    ohe = pickle.load(f)

# 2. Cluster 0: RFR model / vectorizer
with open('model/rfr_0.pkl', 'rb') as f:
    model_cluster0  = pickle.load(f)
    
# 3. Cluster 1: RFR model / vectorizer
with open('model/rfr_1.pkl', 'rb') as f:
    model_cluster1  = pickle.load(f)
    
# 4. Importance ohe
with open("model/importance_ohe.txt", "r", encoding="utf-8") as imp_ohe:
    importance_ohe_L = imp_ohe.readlines()
importance_ohe_L = [i.strip() for i in importance_ohe_L]

# ================== RECOMMENDATION FROM USER ==================
# 1. Only focus on important features. 
# Input: 'Xuất_xứ', 'Phân_khúc_giá', 'Dòng_xe', 'price_segment_code', 'Loại_xe', 'Dung_tích_xe', 'Thương_hiệu', 'cc_numeric'
data = pd.read_excel("data_motobikes.xlsx")
last = st.session_state.get("last_clean")

def options(feature, feature_list):
    opts = sorted(last[feature].dropna().unique().tolist()) if last is not None and feature in last.columns else feature_list
    return opts

origin_opts = options("origin", ["Việt Nam","Nhập Khẩu"])
segment_opts = options("segment", ["Phổ thông","Cận cao cấp","Cao cấp"])
models_opts = options("model", ["Wave","Exciter","Sirius"])
vehicle_types_opts = options("vehicle_type", ["Xe số","Xe tay ga","Xe côn"])
brands_opts = options("brand", BRANDS)
segment_opts = options("segment", ["Phổ thông","Cận cao cấp","Cao cấp"])
cc_numberic_opts = options("cc_numeric", ["75", "137", "200", "40"])

#      Input Options
origin_inp = st.selectbox("Xuất xứ (origin)", options=origin_opts)
segment_inp = st.selectbox("Phân khúc (segment)", options=segment_opts)
model_inp = st.selectbox("Dòng xe (model)", options=models_opts)

segment_map = {"Phổ Thông": "1",
                "Tầm Trung": "2",
                "Cao Cấp": "3"}
price_segment_code = segment_map.get(segment_inp, "1") 
age = st.number_input("Tuổi xe (age)", min_value=0.1, step=0.1, value=3.0, format="%.1f")
vehicle_type_inp = st.selectbox("Loại xe (vehicle_type)", options=vehicle_types_opts)
price_min = st.number_input("Khoảng giá min (triệu VND)", min_value=0.0, value=8.0, step=0.1)
price_max = st.number_input("Khoảng giá max (triệu VND)", min_value=0.0, value=12.0, step=0.1)
engine_size_sel = st.selectbox("Dung tích xe (nhãn)", options=["Dưới 50","50 - 100","100 - 175","Trên 175"], index=2)
brand_inp = st.selectbox("Thương hiệu (brand)", options=brands_opts)
cc_numeric = st.text_input("Dung tích numeric (cc_numeric)", value="137")


# 2. Encoder and Scaling
input_df = pd.DataFrame({'Xuất_xứ':origin_inp,
                         'Phân_khúc_giá':segment_inp,
                         'Dòng_xe':model_inp,
                         "price_segment_code":price_segment_code,
                         'age': age,
                         "Loại_xe":vehicle_type_inp,
                         'Khoảng_giá_min':price_min,
                         'Khoảng_giá_max':price_max,
                         'Năm_đăng_ký':2025 - age,
                         "Dung_tích_xe":engine_size_sel,
                         "Thương_hiệu":brand_inp,
                         "cc_numeric": cc_numeric
                         }, index=[0])

df_input_cate = input_df[['Dung_tích_xe', 'Thương_hiệu', 'Phân_khúc_giá', 'Xuất_xứ',
                          'price_segment_code', 'Dòng_xe', 'Loại_xe', 'cc_numeric']]
df_input_num = input_df[['Khoảng_giá_max', 'age', 'Khoảng_giá_min', 'Năm_đăng_ký']]

encoded_array = ohe.transform(df_input_cate)
encoded_cols = ['E_' + name for name in ohe.get_feature_names_out(df_input_cate.columns)]
encoded_input_cate = pd.DataFrame(encoded_array, columns=encoded_cols, index=df_input_cate.index)
encoded_input_cate = encoded_input_cate.reset_index(drop=True)

X = encoded_input_cate.join(df_input_num)[importance_ohe_L].to_numpy()

cluster_label = clustering_model.predict(X)[0]

if cluster_label == 0:
    predict = model_cluster0.predict(X)
elif cluster_label == 1:
    predict = model_cluster1.predict(X)
else:
    st.write("Something wrong. Cannot predict")

st.text(f"Giá dự đoán: {predict[0]:.2f} triệu")