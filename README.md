# 📊 Estimator - تخمینگر قیمت هوشمند

یک وب‌اپلیکیشن تعاملی مبتنی بر **Streamlit** و الگوریتم‌های **Machine Learning** برای تخمین هوشمند قیمت لپ‌تاپ و املاک در ایران، همراه با ابزارهای تحلیل و مصورسازی داده‌ها (EDA).

---

## ✨ قابلیت‌ها (Features)

1. **💻 تخمین قیمت لپ‌تاپ (Laptop Price Estimator)**:
   - مدل رگرسیون **Random Forest** آموزش‌دیده بر روی دیتاست جامع لپ‌تاپ‌های بازار.
   - ورودی مشخصات سخت‌افزاری (برند، پردازنده، کارت گرافیک، رم، وزن، رزولوشن).
   - نمایش قیمت به هر دو واحد **یورو (€)** و **تومان ایران**.
   - گزارش معیارهای ارزیابی مدل ($R^2$ Score, MAE, MSE, Accuracy).

2. **🏠 تخمین ارزش رهن و مسکن در ایران (Iran Real Estate Estimator)**:
   - مدل **Decision Tree Regressor** آموزش‌دیده بر روی بیش از ۱۶۰ هزار رکورد آگهی مسکن در تهران.
   - انتخاب محله با نگاشت هوشمند برچسب‌ها (`LabelEncoder`).
   - ورودی متراژ، طبقه، سن بنا، تعداد اتاق، آسانسور، پارکینگ و انباری.

3. **📈 تحلیل و اعتبارسنجی داده‌ها (Data Cleaning & EDA)**:
   - مقایسه بلادرنگ داده‌های خام و پیش‌پردازش‌شده.
   - نمودارهای توزیع ویژگی‌ها و ماتریس همبستگی (Correlation Heatmap).
   - پشتیبانی از بارگذاری فایل داده دلخواه کاربر (`.csv` و `.xlsx`).

---

## 🛠️ نصب و راه‌اندازی محلی (Local Setup)

```bash
# ۱. کلون کردن مخزن
git clone https://github.com/abbasi0abolfazl/streamlit_machine_learning.git
cd streamlit_machine_learning

# ۲. ساخت و فعال‌سازی محیط مجازی
python -m venv env
source env/bin/activate  # در ویندوز: env\Scripts\activate

# ۳. نصب پیش‌نیازها
pip install -r requirements.txt

# ۴. اجرای برنامه
streamlit run main.py
```

---

## 🚀 استقرار در فضای ابری (Cloud Deployment)

این پروژه برای استقرار با یک کلیک بر روی **Streamlit Community Cloud** بهینه‌سازی شده است:
1. مخزن را در حساب GitHub خود قرار دهید.
2. به سایت [share.streamlit.io](https://share.streamlit.io) بروید.
3. مخزن `streamlit_machine_learning` و فایل `main.py` را برای انتشار انتخاب کنید.

---

## 👨‍💻 توسعه‌دهنده
**ابوالفضل عباسی**
