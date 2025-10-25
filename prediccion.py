import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

print("🤖 MODELO SIMPLE Y SEGURO")
print("=" * 40)

# Cargar datos
df = pd.read_csv('ventas.csv')

# ANÁLISIS POR REGIÓN
print("\n🌎 ANÁLISIS POR REGIÓN:")
analisis_region = df.groupby('region').agg({
    'precio': ['count', 'mean', 'median'],
    'producto': 'nunique'
}).round(0)

print(analisis_region)

# Crear target binario simple (arriba/abajo de la mediana)
precio_mediano = df['precio'].median()
df['es_caro'] = (df['precio'] > precio_mediano).astype(int)

print(f"\n🎯 OBJETIVO DEL MODELO:")
print(f"   Precio mediano: ${precio_mediano:,.0f}")
print(f"   'Caro' = precio > ${precio_mediano:,.0f}")
print(f"   Distribución: {df['es_caro'].value_counts().to_dict()}")

# Features simples
le_region = LabelEncoder()
le_metodo = LabelEncoder()

X = pd.DataFrame({
    'region': le_region.fit_transform(df['region']),
    'metodo_pago': le_metodo.fit_transform(df['metodo_pago'])
})

y = df['es_caro']

print(f"\n📊 REGIONES EN EL MODELO:")
for i, region in enumerate(le_region.classes_):
    count = (df['region'] == region).sum()
    precio_promedio = df[df['region'] == region]['precio'].mean()
    print(f"   {i}: {region} ({count} ventas, promedio: ${precio_promedio:,.0f})")

print(f"\n💳 MÉTODOS DE PAGO EN EL MODELO:")
for i, metodo in enumerate(le_metodo.classes_):
    count = (df['metodo_pago'] == metodo).sum()
    print(f"   {i}: {metodo} ({count} ventas)")

# División simple
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📈 ENTRENANDO MODELO...")
print(f"   Datos entrenamiento: {X_train.shape[0]} registros")
print(f"   Datos prueba: {X_test.shape[0]} registros")

# Modelo simple
modelo = RandomForestClassifier(n_estimators=50, random_state=42)
modelo.fit(X_train, y_train)

# Evaluar
y_pred = modelo.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 RESULTADOS:")
print(f"   Accuracy: {accuracy:.3f} ({accuracy:.1%})")

if accuracy > 0.5:
    print(f"   ✅ Mejor que adivinar al azar (50%)")
else:
    print(f"   ⚠️  Similar a adivinar al azar")

print(f"\n📋 REPORTE DETALLADO:")
print(classification_report(y_test, y_pred, target_names=['Barato', 'Caro']))

# IMPORTANCIA DE CARACTERÍSTICAS
importancia = modelo.feature_importances_
print(f"\n🔍 IMPORTANCIA DE CARACTERÍSTICAS:")
print(f"   Región: {importancia[0]:.3f}")
print(f"   Método de pago: {importancia[1]:.3f}")

# Guardar
joblib.dump(modelo, 'modelos/modelo_simple.pkl')
joblib.dump({'region': le_region, 'metodo_pago': le_metodo}, 'modelos/encoders_simple.pkl')

print(f"\n💾 MODELO GUARDADO:")
print(f"   ✅ modelos/modelo_simple.pkl")
print(f"   ✅ modelos/encoders_simple.pkl")

# FUNCIÓN DE PREDICCIÓN MEJORADA
print(f"\n🔮 FUNCIÓN DE PREDICCIÓN MEJORADA:")

def predecir_venta(region, metodo_pago):
    """Predice si una venta será cara o barata"""
    
    # Codificar inputs
    try:
        region_enc = le_region.transform([region])[0]
        metodo_enc = le_metodo.transform([metodo_pago])[0]
    except ValueError as e:
        print(f"❌ Error: {e}")
        return None, None
    
    # Predecir
    X_input = [[region_enc, metodo_enc]]
    probabilidad = modelo.predict_proba(X_input)[0][1]  # Probabilidad de "caro"
    prediccion = modelo.predict(X_input)[0]
    
    return prediccion, probabilidad

# EJEMPLOS DE PREDICCIÓN
print(f"\n🎯 EJEMPLOS DE PREDICCIÓN:")
ejemplos = [
    ('Metropolitana', 'Credito'),
    ('Araucania', 'Debito'),
    ('Biobio', 'Transferencia'),
    ('Los Lagos', 'Credito'),
    ('Los Rios', 'Debito')
]

print(f"\n{'Región':<15} {'Método Pago':<12} {'Predicción':<10} {'Prob. Caro':<12} {'Interpretación'}")
print("-" * 80)

for region, metodo in ejemplos:
    pred, prob = predecir_venta(region, metodo)
    
    if pred is not None:
        prediccion_texto = "🔴 CARO" if pred == 1 else "🟢 BARATO"
        interpretacion = f"Probabilidad {prob:.1%} de producto caro"
        
        print(f"{region:<15} {metodo:<12} {prediccion_texto:<10} {prob:>10.1%}   {interpretacion}")

print("-" * 80)

# RESUMEN POR REGIÓN
print(f"\n📊 RESUMEN POR REGIÓN (comportamiento general):")
for region in le_region.classes_:
    ventas_region = df[df['region'] == region]
    proporcion_caras = ventas_region['es_caro'].mean()
    precio_promedio = ventas_region['precio'].mean()
    
    nivel = "🔴 ALTO" if proporcion_caras > 0.6 else "🟡 MEDIO" if proporcion_caras > 0.4 else "🟢 BAJO"
    
    print(f"   {region:<15}: {proporcion_caras:.1%} ventas caras (avg: ${precio_promedio:,.0f}) {nivel}")

print(f"\n🎉 MODELO COMPLETADO!")
print(f"💡 Este modelo predice si una venta será CARA o BARATA basándose solo en:")
print(f"   • Región de la venta")
print(f"   • Método de pago utilizado")
