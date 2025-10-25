import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

print("🤖 MODELO SIMPLE Y SEGURO")
print("=" * 40)

# Cargar datos
df = pd.read_csv('ventas.csv')

# Crear target binario simple (arriba/abajo de la mediana)
precio_mediano = df['precio'].median()
df['es_caro'] = (df['precio'] > precio_mediano).astype(int)

print(f"🎯 Target: precio > ${precio_mediano:,.0f}")
print(f"📊 Distribución: {df['es_caro'].value_counts().to_dict()}")

# Features simples
le_region = LabelEncoder()
le_metodo = LabelEncoder()

X = pd.DataFrame({
    'region': le_region.fit_transform(df['region']),
    'metodo_pago': le_metodo.fit_transform(df['metodo_pago'])
})

y = df['es_caro']

# División simple (sin estratificar para evitar errores)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modelo simple
modelo = RandomForestClassifier(n_estimators=50, random_state=42)
modelo.fit(X_train, y_train)

# Evaluar
y_pred = modelo.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"🎯 Accuracy: {accuracy:.3f}")

# Guardar
joblib.dump(modelo, 'modelos/modelo_simple.pkl')
joblib.dump({'region': le_region, 'metodo_pago': le_metodo}, 'modelos/encoders_simple.pkl')

print("✅ Modelo simple guardado!")
