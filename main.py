import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import arabic_reshaper
from bidi.algorithm import get_display

st.set_page_config(
    page_title="Estimator - تخمینگر",
    page_icon=":bar_chart:",
    layout="wide"
)

# Load CSS
try:
    if os.path.exists("src/style.css"):
        with open("src/style.css", "r", encoding="utf-8") as f:
            custom_css = f.read()
        st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)
except Exception as e:
    st.warning(f"Could not load custom CSS: {e}")


def reshape_persian(text):
    """Helper to reshape Persian text for Matplotlib rendering."""
    try:
        return get_display(arabic_reshaper.reshape(str(text)))
    except Exception:
        return str(text)


@st.cache_data
def load_data(type_data: str):
    """Cached dataset loader."""
    clean_path = os.path.join("CSV_data", f"clean_data_{type_data}.csv")
    bad_path = os.path.join("CSV_data", f"bad_data_{type_data}.csv")

    enc = 'latin1' if type_data == 'laptop' else 'utf-8'
    clean_data = pd.read_csv(clean_path, encoding=enc)
    bad_data = pd.read_csv(bad_path, encoding=enc)
    return bad_data, clean_data


@st.cache_resource
def get_laptop_model():
    """Cached Random Forest training for laptop prices."""
    _, clean_data = load_data('laptop')
    x = clean_data.drop('Price_euros', axis=1)
    y = clean_data['Price_euros']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    forest = RandomForestRegressor(n_estimators=100, random_state=42)
    forest.fit(x_train_scaled, y_train)

    y_pred = forest.predict(x_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    accuracy = forest.score(x_test_scaled, y_test) * 100

    return forest, scaler, x.columns.tolist(), mae, mse, r2, accuracy


@st.cache_resource
def get_housing_model():
    """Cached Decision Tree training for Iranian housing prices."""
    bad_data, clean_data = load_data('home_IR')
    x = clean_data.drop(['all_to_deposit', 'district'], axis=1, errors='ignore').copy()
    y = clean_data['all_to_deposit'].copy()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    tree = DecisionTreeRegressor(max_depth=15, random_state=42)
    tree.fit(x_train, y_train)

    y_pred = tree.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    accuracy = tree.score(x_test, y_test) * 100

    # Fit LabelEncoder on regions
    le = LabelEncoder()
    valid_regions = bad_data['region'].dropna().astype(str).unique()
    le.fit(valid_regions)

    return tree, x.columns.tolist(), mae, mse, r2, accuracy, le


def do_home():
    """Home landing page."""
    try:
        if os.path.exists("src/index.html"):
            with open("src/index.html", "r", encoding="utf-8") as f:
                home_page = f.read()
            st.markdown(home_page, unsafe_allow_html=True)
        else:
            st.title("به سایت تخمینگر خوش آمدید")
            st.info("سایت Estimator - تخمینگر برای تخمین قیمت لپ تاپ و مسکن با مدل‌های یادگیری ماشین طراحی شده است.")
    except Exception as e:
        st.error(f"Error loading homepage: {e}")


def upload_file():
    return st.file_uploader("فایل داده خود را آپلود کنید", type=["csv", "xlsx"])


def count_empty_values(df):
    return df.isnull().sum()


def count_text_values(df):
    text_columns = df.select_dtypes(include=['object']).columns
    return df[text_columns].isna().sum()


def plot_bad_data_chart(df):
    text_empty_counts = count_text_values(df)
    if text_empty_counts.empty:
        st.info("هیچ ستون متنی در این دیتاست یافت نشد.")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([reshape_persian(c) for c in text_empty_counts.index], text_empty_counts.values, color='salmon')
    ax.set_xlabel(reshape_persian('ستون‌ها'))
    ax.set_ylabel(reshape_persian('تعداد داده‌های خالی'))
    ax.set_title(reshape_persian('نمودار داده ناسالم'))
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    plt.close(fig)


def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        st.warning("هیچ ستون عددی برای تحلیل همبستگی یافت نشد.")
        return
    correlation_matrix = numeric_df.corr()
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.warning(f"نمایش نمودار به دلیل مشکلات فنی امکانپذیر نیست: {e}")


def show_other_chart(user_df):
    show_chart_user = st.checkbox("نمایش نمودار توزیع ستون")
    if show_chart_user:
        column_name_user = st.selectbox("انتخاب ستون:", list(user_df.columns))
        fig_u, ax_u = plt.subplots(figsize=(9, 5))
        sns.countplot(x=column_name_user, data=user_df, ax=ax_u)
        ax_u.tick_params(axis='x', rotation=90)
        st.pyplot(fig_u)
        plt.close(fig_u)

    show_empty_values = st.checkbox("نمایش تعداد داده‌های خالی")
    if show_empty_values:
        empty_values = count_empty_values(user_df)
        st.bar_chart(empty_values)

    show_heatmap = st.checkbox("نمایش ماتریس همبستگی")
    if show_heatmap:
        plot_correlation_heatmap(user_df)

    show_bad_data = st.checkbox("نمودار مقادیر تهی متنی")
    if show_bad_data:
        plot_bad_data_chart(user_df)


def initialize_session_state():
    if 'remove' not in st.session_state:
        st.session_state.remove = False
    if 'clicked' not in st.session_state:
        st.session_state.clicked = False
    if 'prediction_laptop' not in st.session_state:
        st.session_state.prediction_laptop = False
    if 'prediction_home' not in st.session_state:
        st.session_state.prediction_home = False


def do_laptop():
    initialize_session_state()
    bad_data, clean_data = load_data('laptop')
    forest, scaler, x_columns, mae, mse, r2, accuracy = get_laptop_model()

    st.subheader("💻 تخمین هوشمند قیمت لپ‌تاپ")

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("🚀 اجرای تخمین قیمت", key="laptop_pred_btn"):
            st.session_state.prediction_laptop = True

    if st.session_state.prediction_laptop:
        st.divider()
        st.markdown("#### مشخصات لپ‌تاپ مورد نظر را انتخاب کنید:")

        c1, c2, c3 = st.columns(3)
        with c1:
            os_category = st.selectbox('سیستم عامل', ['Windows', 'Linux', 'No OS'])
            company_category = st.selectbox('برند سازنده', ['Acer', 'Razer', 'MSI'])
            laptop_type = st.selectbox('نوع لپ‌تاپ', ['Notebook', 'Ultrabook', 'Gaming', 'Workstation'])

        with c2:
            cpu_category = st.selectbox('پردازنده (CPU)', ['Intel', 'AMD'])
            gpu_category = st.selectbox('کارت گرافیک (GPU)', ['NVIDIA', 'Intel', 'AMD'])
            cpu_frequency = st.selectbox('فرکانس پردازنده (GHz)', sorted(clean_data['CPU Frequency'].unique()))

        with c3:
            ram = st.selectbox('حافظه رم (GB)', sorted(clean_data['Ram'].unique()))
            weight = st.number_input('وزن (kg)', min_value=0.5, max_value=5.0, value=1.8, step=0.1)
            screen_res = st.selectbox('رزولوشن صفحه نمایش', ['1920x1080', '1366x768', '2560x1440', '3840x2160'])

        width_val, height_val = [int(v) for v in screen_res.split('x')]

        # Build feature dictionary aligned with model features
        feature_dict = {col: 0.0 for col in x_columns}

        # OS
        if os_category == 'Windows' and 'Windows 7' in feature_dict:
            feature_dict['Windows 7'] = 1.0
        elif os_category == 'Linux' and 'Linux' in feature_dict:
            feature_dict['Linux'] = 1.0
        elif os_category == 'No OS' and 'No OS' in feature_dict:
            feature_dict['No OS'] = 1.0

        # Company
        if company_category in feature_dict:
            feature_dict[company_category] = 1.0

        # CPU
        if cpu_category == 'AMD' and 'AMD_CPU' in feature_dict:
            feature_dict['AMD_CPU'] = 1.0
        elif cpu_category == 'Intel' and 'Intel_CPU' in feature_dict:
            feature_dict['Intel_CPU'] = 1.0

        # GPU
        if gpu_category == 'NVIDIA' and 'Nvidia_GPU' in feature_dict:
            feature_dict['Nvidia_GPU'] = 1.0
        elif gpu_category == 'Intel' and 'Intel_GPU' in feature_dict:
            feature_dict['Intel_GPU'] = 1.0
        elif gpu_category == 'AMD' and 'AMD_GPU' in feature_dict:
            feature_dict['AMD_GPU'] = 1.0

        # Laptop Type
        if laptop_type in feature_dict:
            feature_dict[laptop_type] = 1.0

        # Numerical fields
        if 'Weight' in feature_dict: feature_dict['Weight'] = float(weight)
        if 'CPU Frequency' in feature_dict: feature_dict['CPU Frequency'] = float(cpu_frequency)
        if 'Screen Height' in feature_dict: feature_dict['Screen Height'] = float(height_val)
        if 'Screen Width' in feature_dict: feature_dict['Screen Width'] = float(width_val)
        if 'Ram' in feature_dict: feature_dict['Ram'] = float(ram)

        user_input_df = pd.DataFrame([feature_dict])[x_columns]
        user_input_scaled = scaler.transform(user_input_df)

        predicted_price_eur = forest.predict(user_input_scaled)[0]
        eur_to_toman_rate = 65000  # Updated standard approximate EUR/TMN rate
        predicted_price_toman = predicted_price_eur * eur_to_toman_rate

        st.success(f"💰 **قیمت تخمینی:** {predicted_price_eur:,.2f} یورو  |  {predicted_price_toman:,.0f} تومان")

        with st.expander("📊 ارزیابی دقت مدل یادگیری ماشین (Random Forest)"):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("دقت مدل (Accuracy)", f"{accuracy:.1f}%")
            m2.metric("ضریب تعیین (R²)", f"{r2:.3f}")
            m3.metric("خطای میانگین مطلق (MAE)", f"{mae:,.1f} €")
            m4.metric("خطای میانگین مربعات (MSE)", f"{mse:,.1f}")

    st.divider()
    if st.button('📂 آپلود فایل داده جدید', key="laptop_upload_btn"):
        st.session_state.clicked = True

    user_df = None
    if st.session_state.clicked:
        uploaded_file = upload_file()
        if uploaded_file is not None:
            st.session_state.remove = True
            try:
                if uploaded_file.name.endswith('.csv'):
                    user_df = pd.read_csv(uploaded_file)
                else:
                    user_df = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"خطا در خواندن فایل: {e}")
                st.session_state.remove = False

    if not st.session_state.remove:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h4 style='color:red;'>نمونه داده خام (قبل از پیش‌پردازش)</h4>", unsafe_allow_html=True)
            st.dataframe(bad_data.head(), use_container_width=True)
            if st.checkbox("نمایش نمودار داده خام", key="bad_laptop_chart"):
                col_name = st.selectbox("انتخاب ستون:", list(bad_data.columns), key="bad_lap_col")
                fig_bad, ax_bad = plt.subplots(figsize=(8, 4))
                sns.countplot(x=col_name, data=bad_data, ax=ax_bad)
                ax_bad.tick_params(axis='x', rotation=80)
                st.pyplot(fig_bad)
                plt.close(fig_bad)

        with col2:
            st.markdown("<h4 style='color:green;'>نمونه داده تمیز (پردازش شده)</h4>", unsafe_allow_html=True)
            st.dataframe(clean_data.head(), use_container_width=True)
            if st.checkbox("نمایش نمودار داده صحیح", key="clean_laptop_chart"):
                col_name = st.selectbox("انتخاب ستون:", list(clean_data.columns), key="clean_lap_col")
                fig_clean, ax_clean = plt.subplots(figsize=(8, 4))
                sns.countplot(x=col_name, data=clean_data, ax=ax_clean)
                ax_clean.tick_params(axis='x', rotation=80)
                st.pyplot(fig_clean)
                plt.close(fig_clean)

    elif user_df is not None:
        st.markdown("<h4 style='color:blue;'>داده بارگزاری شده توسط شما:</h4>", unsafe_allow_html=True)
        st.dataframe(user_df.head(), use_container_width=True)
        with st.expander("📈 تحلیل و نمودارهای داده بارگذاری شده"):
            show_other_chart(user_df)


def do_estimation_iran():
    initialize_session_state()
    bad_data, clean_data = load_data('home_IR')
    tree, x_columns, mae, mse, r2, accuracy, le = get_housing_model()

    st.subheader("🏠 تخمین قیمت مسکن در ایران (تهران)")

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("🚀 اجرای تخمین قیمت مسکن", key="home_pred_btn"):
            st.session_state.prediction_home = True

    if st.session_state.prediction_home:
        st.divider()
        st.markdown("#### مشخصات ملک را وارد نمایید:")

        c1, c2, c3 = st.columns(3)
        with c1:
            regions_list = sorted(list(le.classes_))
            selected_region = st.selectbox('محله / منطقه', regions_list)
            area = st.number_input('متراژ (متر مربع)', min_value=20, max_value=1000, value=80, step=5)
            rooms = st.selectbox('تعداد اتاق خواب', sorted(clean_data['rooms'].unique()))

        with c2:
            floor = st.selectbox('طبقه', sorted(clean_data['floor'].unique()), help='0 به معنی همکف است')
            age = st.selectbox('سن بنا (سال)', sorted(clean_data['age'].unique()))
            elavator = 1 if st.selectbox('آسانسور', ['دارد', 'ندارد']) == 'دارد' else 0

        with c3:
            parking = 1 if st.selectbox('پارکینگ', ['دارد', 'ندارد']) == 'دارد' else 0
            warehouse = 1 if st.selectbox('انباری', ['دارد', 'ندارد']) == 'دارد' else 0

        encoded_region = le.transform([selected_region])[0]

        feature_dict = {
            'floor': float(floor),
            'area': float(area),
            'age': float(age),
            'rooms': float(rooms),
            'elavator': float(elavator),
            'parking': float(parking),
            'Warehouse': float(warehouse),
            'address': float(encoded_region),
            'rent': float(clean_data['rent'].median()),
            'deposit': float(clean_data['deposit'].median())
        }

        user_input = pd.DataFrame([feature_dict])[x_columns]
        predicted_price = tree.predict(user_input)[0]

        # Present formatted estimated price
        st.success(f"🏡 **ارزش تخمینی ودیعه کامل ملک:** {predicted_price * 1000:,.0f} تومان")

        with st.expander("📊 ارزیابی دقت مدل مسکن (Decision Tree)"):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("دقت مدل", f"{accuracy:.1f}%")
            m2.metric("ضریب تعیین (R²)", f"{r2:.3f}")
            m3.metric("خطای میانگین مطلق (MAE)", f"{mae:,.0f}")
            m4.metric("خطای میانگین مربعات (MSE)", f"{mse:,.0f}")

    st.divider()
    if st.button('📂 آپلود فایل داده جدید', key="home_upload_btn"):
        st.session_state.clicked = True

    user_df = None
    if st.session_state.clicked:
        uploaded_file = upload_file()
        if uploaded_file is not None:
            st.session_state.remove = True
            try:
                if uploaded_file.name.endswith('.csv'):
                    user_df = pd.read_csv(uploaded_file)
                else:
                    user_df = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"خطا در خواندن فایل: {e}")
                st.session_state.remove = False

    if not st.session_state.remove:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h4 style='color:red;'>نمونه داده خام مسکن</h4>", unsafe_allow_html=True)
            st.dataframe(bad_data.head(), use_container_width=True)
            if st.checkbox("نمایش نمودار داده خام مسکن", key="bad_home_chart"):
                col_name = st.selectbox("انتخاب ستون:", list(bad_data.columns), key="bad_home_col")
                fig_bad, ax_bad = plt.subplots(figsize=(8, 4))
                sns.countplot(x=col_name, data=bad_data, ax=ax_bad)
                ax_bad.tick_params(axis='x', rotation=80)
                st.pyplot(fig_bad)
                plt.close(fig_bad)

        with col2:
            st.markdown("<h4 style='color:green;'>نمونه داده تمیز مسکن</h4>", unsafe_allow_html=True)
            st.dataframe(clean_data.head(), use_container_width=True)
            if st.checkbox("نمایش نمودار داده تمیز مسکن", key="clean_home_chart"):
                col_name = st.selectbox("انتخاب ستون:", list(clean_data.columns), key="clean_home_col")
                fig_clean, ax_clean = plt.subplots(figsize=(8, 4))
                sns.countplot(x=col_name, data=clean_data, ax=ax_clean)
                ax_clean.tick_params(axis='x', rotation=80)
                st.pyplot(fig_clean)
                plt.close(fig_clean)

    elif user_df is not None:
        st.markdown("<h4 style='color:blue;'>داده بارگزاری شده توسط شما:</h4>", unsafe_allow_html=True)
        st.dataframe(user_df.head(), use_container_width=True)
        with st.expander("📈 تحلیل و نمودارهای داده بارگذاری شده"):
            show_other_chart(user_df)


def do_about_us():
    st.markdown("""
    <div style="direction: rtl; text-align: justify; padding: 20px;">
        <h2>تخمینگر هوشمند (Estimator)</h2>
        <p>اپلیکیشن وب <strong>Estimator</strong> بستری برای تخمین هوشمند قیمت‌ها با به‌کارگیری الگوریتم‌های یادگیری ماشین (Machine Learning) است.</p>
        <h4>قابلیت‌های اصلی:</h4>
        <ul>
            <li><strong>تخمین قیمت لپ‌تاپ:</strong> پیش‌بینی قیمت دلاری و تومانی بر اساس سخت‌افزار با مدل Random Forest.</li>
            <li><strong>تخمین ارزش مسکن در ایران:</strong> محاسبه ارزش رهن و قیمت با مدل رگرسیون درخت تصمیم (Decision Tree) بر روی بیش از ۱۶۰ هزار آگهی.</li>
            <li><strong>تحلیل و اعتبارسنجی داده‌ها (EDA):</strong> مقایسه داده‌های خام و پردازش‌شده به همراه هیت‌مپ ماتریس همبستگی و نمودارهای توزیع.</li>
        </ul>
        <hr/>
        <p><strong>توسعه‌دهنده:</strong> ابوالفضل عباسی</p>
    </div>
    """, unsafe_allow_html=True)


menu = {
    'items': {
        'خانه': {'action': do_home, 'item_icon': 'house-door-fill'},
        'تخمین قیمت لپ تاپ': {'action': do_laptop, 'item_icon': 'laptop-fill'},
        'تخمین قیمت خانهIR': {'action': do_estimation_iran, 'item_icon': 'building-fill'},
        'درباره ما': {'action': do_about_us, 'item_icon': 'info-circle-fill'},
    },
    'default_index': 0,
    'orientation': 'horizontal',
    'styles': {
        "container": {"margin": "0px !important", "padding": "0!important", "background-color": "#fafafa", "direction": "rtl"},
        "icon": {"color": "black", "font-size": "18px"},
        "nav-link": {"font-size": "18px", "direction": "rtl", "margin": "0px", "--hover-color": "#eee"},
        "nav-link-selected": {"font-size": "18px", "font-weight": "bold", "color": "#0066cc", "direction": "rtl"},
    }
}


def show_menu(menu_config):
    options = list(menu_config['items'].keys())
    icons = [v['item_icon'] for v in menu_config['items'].values()]

    menu_selection = option_menu(
        menu_title=None,
        options=options,
        icons=icons,
        default_index=menu_config['default_index'],
        orientation=menu_config['orientation'],
        styles=menu_config['styles']
    )

    action = menu_config['items'][menu_selection]['action']
    if action:
        action()


if __name__ == '__main__':
    show_menu(menu)
