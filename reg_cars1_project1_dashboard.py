import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

#-----sidebar-------
st.sidebar.image(r'C:\Users\omar\Desktop\reg_cars_streamlit\imge\cars.png')

#-----header-------
st.image(r"C:\Users\omar\Desktop\reg_cars_streamlit\imge\cars2.png")

st.sidebar.title("🧮 إدخال البيانات")

# ثابت: قائمة الماركات


brands_list = ['Maruti', 'Hyundai', 'Toyota', 'Ford', 'Honda', 'BMW', 'Mercedes', 'Audi', 'Nissan', 'Volkswagen']
brand = st.sidebar.selectbox("🏷️ اختر الماركة", brands_list, key="brand")

year = st.sidebar.number_input("📅 سنة الصنع", min_value=1990, max_value=2025, value=2015, key="year")
km_driven = st.sidebar.number_input("🚗 عدد الكيلومترات المقطوعة", min_value=0, value=50000, step=1000, key="km_driven")
fuel = st.sidebar.selectbox("⛽ نوع الوقود", ["Petrol", "Diesel", "CNG", "LPG", "Electric"], key="fuel")
seller_type = st.sidebar.selectbox("🧍 نوع البائع", ["Individual", "Dealer", "Trustmark Dealer"], key="seller_type")
transmission = st.sidebar.selectbox("⚙️ ناقل الحركة", ["Manual", "Automatic"], key="transmission")
owner = st.sidebar.selectbox("👥 عدد المالكين السابقين", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"], key="owner")
mileage = st.sidebar.number_input("📏 المسافة المقطوعة (كم/لتر)", min_value=0.0, value=18.0, key="mileage")
engine = st.sidebar.number_input("🧠 سعة المحرك (cc)", min_value=500, max_value=6000, value=1500, key="engine")
max_power = st.sidebar.number_input("⚡ أقصى قوة (حصان)", min_value=30.0, value=100.0, key="max_power")
seats = st.sidebar.selectbox("💺 عدد المقاعد", [2, 4, 5, 6, 7, 8, 9, 10], key="seats")
torque = st.sidebar.number_input("🌀 عزم الدوران (Nm)", min_value=20.0, max_value=1000.0, value=113.8, key="torque")
# تعليمات بجانب الحقول
st.sidebar.markdown("### تعليمات الإدخال:")
st.sidebar.markdown("""
- اختر الماركة من القائمة المنسدلة.
- أدخل سنة الصنع بين 1990 و 2025.
- أدخل عدد الكيلومترات المقطوعة. تأكد من أن الرقم معقول.
- اختر نوع الوقود من الخيارات المتاحة (البنزين، الديزل، إلخ).
- اختر نوع البائع بين فردي أو تاجر.
- اختر نوع ناقل الحركة (يدوي أو أوتوماتيكي).
- اختر عدد المالكين السابقين.
- أدخل سعة المحرك (بـ "سي سي").
- أدخل أقصى قوة للمحرك (بـ "حصان").
- أدخل عزم الدوران (بـ "نيوتن متر").
""")

# رسائل التنبيه في حال كانت المدخلات غير صحيحة أو ناقصة
if not (year and km_driven and fuel and seller_type and transmission and owner and mileage and engine and max_power and torque and seats and brand):
    st.warning("⚠️ تأكد من إدخال جميع الحقول بشكل صحيح.")

# إدخال المستخدم كـ DataFrame
input_data = pd.DataFrame({
    "year": [year],
    "km_driven": [km_driven],
    "fuel": [fuel],
    "seller_type": [seller_type],
    "transmission": [transmission],
    "owner": [owner],
    "mileage": [mileage],
    "engine": [engine],
    "max_power": [max_power],
    "torque": [torque],
    "seats": [seats],
    "brand": [brand]
})

st.write("🔢 المدخلات التي اخترتها:")
st.write(input_data)

# تحميل البيانات الأصلية
df_car = pd.read_csv(r"C:\Users\omar\Desktop\reg_cars_streamlit\data\reg_cars_selling.csv")

data = df_car.to_csv(index=False)

#------- cleaning -------
df_car.drop_duplicates(inplace=True)
df_car = df_car.dropna()

# تنظيف وتحويل الأعمدة الرقمية النصية
df_car['max_power'] = df_car['max_power'].str.replace(r' bhp$', '', regex=True).astype(float)

df_car['mileage'] = df_car['mileage'].str.replace(r' kmpl$', '', regex=True)
df_car['mileage'] = df_car['mileage'].str.replace(r' km/kg$', '', regex=True)
df_car['mileage'] = df_car['mileage'].astype(float)  # تحويله إلى رقم

df_car['engine'] = df_car['engine'].str.replace(r' CC$', '')
df_car['engine'] = df_car['engine'].str.replace(r'CC', '')
df_car['engine'] = df_car['engine'].astype(float)  # تحويله إلى رقم

# حفظ عمود العلامة التجارية
df_car['brand'] = df_car['name'].apply(lambda x: str(x).split()[0])  # استخراج أول كلمة كعلامة تجارية
df_car.drop(columns=['name'] ,axis=1 , inplace=True)

df_car['torque'] = df_car['torque'].astype(str)

# معالجة عمود العزم torque
def extract_torque(value):
    if pd.isna(value): return None
    value = str(value).lower().strip()
    matches = re.findall(r'([\d\.]+)\s*(nm|kgm)?', value, re.IGNORECASE)
    torque_values = []
    for match in matches:
        torque_value = float(match[0])
        unit = match[1].lower() if match[1] else 'nm'
        if unit == "kgm":
            torque_value *= 9.80665
        torque_values.append(torque_value)
    if torque_values:
        return sum(torque_values) / len(torque_values)
    return None

df_car['torque'] = df_car['torque'].apply(extract_torque)

df_car['selling_price_INR'] = df_car['selling_price']
df_car.drop('selling_price' , axis=1 , inplace=True)


# ترميز الأعمدة النصية المهمة فقط، لتفادي ترميز غير متوقع
# ترميز بيانات الإدخال باستخدام نفس الـ encoders
from sklearn.preprocessing import LabelEncoder


# تعريف الأعمدة النصية التي تحتاج إلى تشفير
categorical_columns = ['fuel', 'seller_type', 'transmission', 'owner', 'brand']
encoder = LabelEncoder()

# تأكد من عدم وجود قيم مفقودة في الأعمدة النصية
input_data_encoded = input_data.copy()
for col in categorical_columns:
    # التعامل مع القيم المفقودة بإزالتها أو استبدالها (استبدال بـ "Unknown" في هذا المثال)
    input_data_encoded[col] = input_data_encoded[col].fillna("Unknown")
    
    # تدريب الـ encoder على بيانات التدريب فقط
    encoder.fit(df_car[col])
    
    # استخدام transform فقط لتجنب الفئات غير المعروفة
    input_data_encoded[col] = encoder.transform(input_data_encoded[col])

# تأكد من عدم وجود قيم مفقودة في بيانات التدريب
df_car[categorical_columns] = df_car[categorical_columns].fillna("Unknown")

# تشفير بيانات التدريب باستخدام fit
for col in categorical_columns:
    encoder.fit(df_car[col])  # تدريب الـ encoder فقط على بيانات التدريب
    df_car[col] = encoder.transform(df_car[col])  # تحويل بيانات التدريب إلى قيم عددية


#LocalOutlierFactor()
lof = LocalOutlierFactor()
outlier_labels = lof.fit_predict(df_car)
df_car['outlier_labels'] = outlier_labels
df_car = df_car[df_car['outlier_labels'] !=-1 ]
df_car.drop(columns=['outlier_labels'], inplace=True)


# تقسيم البيانات
X = df_car.drop(['selling_price_INR'], axis=1)
y = df_car['selling_price_INR']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# تدريب النموذج
model = RandomForestRegressor()
model.fit(X_train, y_train)

# التنبؤ بالسعر بناءً على المدخلات
predicted_price = model.predict(input_data_encoded)[0]

# عرض السعر المتوقع
st.title("💰 السعر المتوقع بناءً على المدخلات:")
st.subheader(f"₹ {predicted_price:,.0f}")
y_pred = model.predict(X_test)
# حساب R² (مؤشر القوة التفسيرية)
r2 = r2_score(y_test, y_pred)
st.write(f"R² Score: {r2:.2f}")

# حساب Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
st.write(f"Mean Squared Error: {mse:.2f}")






#  الرسم البياني  
st.subheader("📊 مقارنة أسعار السيارات حسب نوع الوقود")

# نعرض المتوسطات لكل نوع وقود
fuel_avg_prices = df_car.groupby("fuel")["selling_price_INR"].mean().sort_values(ascending=False)
# Data Analysis 📊 
st.write("### 💡 متوسط الأسعار حسب نوع الوقود:")
st.dataframe(fuel_avg_prices.reset_index().rename(columns={"fuel": "نوع الوقود", "selling_price_INR": "متوسط السعر"}))

df_car_numeric_analysis = df_car
# Bar Plot 
fig1, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=fuel_avg_prices.index, y=fuel_avg_prices.values, palette="viridis", ax=ax)  # استخدام اللون الفخم
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

# عرض الرسم البياني
st.pyplot(fig1)

# Distribution of Cars by Brand countplot
fig2 = plt.figure(figsize=(14, 8))
sns.countplot(
    x='brand', 
    data=df_car_numeric_analysis, 
    palette='mako',
    edgecolor='black', 
    linewidth=1.5
)
plt.title('Distribution of Cars by Brand', fontsize=18, weight='bold', color='navy')
plt.xlabel('Car Brand', fontsize=14, weight='bold', color='darkblue')
plt.ylabel('Number of Cars', fontsize=14, weight='bold', color='darkblue')
plt.xticks(rotation=80, fontsize=12)
# عرض الرسم داخل Streamlit
st.pyplot(fig2)

transmission_avg_price = df_car_numeric_analysis.groupby('transmission')['selling_price_INR'].mean()
transmission_avg_price

fig3 = plt.figure(figsize=(12,8))
transmission_avg_price.plot(kind='pie' ,autopct = "%.2f%%"  )
plt.title('Average Selling Price Distribution by Transmission Type', fontsize=16, weight='bold', color='darkblue')
plt.ylabel('')
plt.legend(title='Transmission Type', loc='upper left', fontsize=12, title_fontsize=14)
st.pyplot(fig3)

fig4 = plt.figure(figsize=(12,8))
sns.barplot(data=df_car_numeric_analysis, x='year', y='selling_price_INR', palette='mako', ci=None, edgecolor='black', linewidth=1.5)
plt.title('Average Selling Price by Year', fontsize=16, weight='bold', color='darkblue')
plt.xlabel('Year', fontsize=14)
plt.ylabel('Average Selling Price', fontsize=14)
plt.xticks(rotation=66)
st.pyplot(fig4)


fig5 = plt.figure(figsize=(12,8))
sns.barplot(data=df_car_numeric_analysis, x='seats', y='selling_price_INR',palette='viridis', ci=None, edgecolor='black', linewidth=1.5)
plt.title('Average Selling Price by Number of Seats', fontsize=16, weight='bold', color='darkblue')
plt.xlabel('Number of Seats', fontsize=14)
plt.ylabel('Average Selling Price', fontsize=14)
st.pyplot(fig5)

fig6 = plt.figure(figsize=(12,8))
sns.lineplot(data=df_car_numeric_analysis, x='year', y='selling_price_INR', hue='fuel',marker='o', palette='Set2',linewidth=2, markersize=8)
plt.title('Car Price Trends Over the Years', fontsize=16, weight='bold', color='darkblue')
plt.xlabel('Year', fontsize=14)
plt.ylabel('Selling Price', fontsize=14)
st.pyplot(fig6)

fig7 = plt.figure(figsize=(12,8))
sns.countplot(data=df_car_numeric_analysis, x='fuel', hue='transmission', palette='Set2')
st.pyplot(fig7)


df_grouped = df_car_numeric_analysis.groupby(['fuel', 'transmission'])['selling_price_INR'].mean().reset_index()
df_grouped

fig8 = plt.figure(figsize=(12,8))
sns.barplot(x='fuel', y='selling_price_INR', hue='transmission', data=df_grouped, palette='Set1')
plt.title('Average Selling Price by Fuel Type and Transmission Type', fontsize=16, weight='bold', color='darkblue')
plt.xlabel('Fuel Type', fontsize=14)
plt.ylabel('Average Selling Price', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.legend(title='Transmission Type', title_fontsize=12, fontsize=12, loc='upper left')
plt.tight_layout()
plt.show()
st.pyplot(fig8)


fig9 = plt.figure(figsize=(12,8))
sns.boxplot(x='year', y='selling_price_INR', data=df_car_numeric_analysis, palette='Set2', width=0.8)
plt.title('Distribution of Car Prices by Year', fontsize=18, weight='bold', color='darkblue')
plt.xlabel('Year', fontsize=14, color='darkblue')
plt.ylabel('Car Price (Thousands)', fontsize=14, color='darkblue')
plt.xticks(rotation=66, fontsize=12, color='black')
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.show()
st.pyplot(fig9)

fig10 = plt.figure(figsize=(12,8))
plt.bar(df_car['seller_type'] ,df_car['selling_price_INR'] , color = 'brown')
plt.title('seller_type', fontsize=18, weight='bold', color='darkblue')
plt.xlabel('seller_type', fontsize=14, color='darkblue')
plt.ylabel('selling_price', fontsize=14, color='darkblue')
plt.xticks(rotation=66, fontsize=12, color='black')
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.show()
st.pyplot(fig10)



st.write('البيانات التي استخدمناها وحللناها لتوقع اسعار السيارات هي ')
st.download_button(
    label="Download file",
    data=data,
    file_name="cars_data.csv",
    mime="text/csv"
)


# streamlit run "C:\Users\omar\Desktop\reg_cars_streamlit\code\reg_cars1_project1_dashboard.py"