# -*- coding: utf-8 -*-
"""
PRÁCTICA CALIFICADA - SENATI
EJERCICIO 1: ¿El cliente está SATISFECHO o NO con el servicio?

Algoritmo: KNN (K-Nearest Neighbors = K Vecinos más Cercanos)
Idea simple: para clasificar a un cliente nuevo, buscamos los "k" clientes
más parecidos a él en los datos y vemos qué opinaron la mayoría.
Si la mayoría de sus vecinos estaba satisfecha -> lo predecimos como satisfecho.

Librerías:
  - pandas       -> leer y revisar los datos (CSV)
  - scikit-learn -> dividir datos, escalar y entrenar el modelo KNN
  - matplotlib   -> hacer y guardar los gráficos
"""

import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Solo guardamos imágenes, no abrimos ventanas
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)

# Rutas: así el script funciona sin importar desde qué carpeta lo ejecutes
CARPETA = os.path.dirname(os.path.abspath(__file__))
RUTA_CSV = os.path.join(CARPETA, "datos", "clientes_satisfaccion.csv")
CARPETA_GRAFICOS = os.path.join(CARPETA, "graficos")
os.makedirs(CARPETA_GRAFICOS, exist_ok=True)

# =============================================================================
# 1. CARGAR Y REVISAR LOS DATOS (pandas)
# =============================================================================
print("=" * 70)
print("EJERCICIO 1: PREDICCIÓN DE SATISFACCIÓN DEL CLIENTE CON KNN")
print("=" * 70)

datos = pd.read_csv(RUTA_CSV)

print("\n--- Primeras 5 filas (head) ---")
print(datos.head())

print("\n--- Información de las columnas (info) ---")
datos.info()  # info() imprime directamente: tipos de dato y cantidad de filas

print("\n--- Estadísticas básicas (describe) ---")
print(datos.describe().round(2))

print("\n--- Valores nulos por columna ---")
print(datos.isnull().sum())
# Si hubiera nulos, una forma sencilla de tratarlos es eliminar esas filas:
if datos.isnull().sum().sum() > 0:
    datos = datos.dropna()
    print("Se eliminaron filas con valores nulos.")
else:
    print("No hay valores nulos. ¡Los datos están completos!")

print("\n--- Cantidad de clientes por clase (1 = satisfecho, 0 = no) ---")
print(datos["satisfecho"].value_counts())

# =============================================================================
# 2. PREPARACIÓN DE LOS DATOS (scikit-learn)
# =============================================================================
# X = las características (lo que usamos para predecir)
# y = la variable objetivo (lo que queremos predecir)
COLUMNAS_X = ["edad", "tiempo_espera_min", "calidad_atencion",
              "precio_percibido", "resolvio_problema"]
X = datos[COLUMNAS_X]
y = datos["satisfecho"]

# Dividimos: 80% para entrenar y 20% para probar.
# stratify=y mantiene la misma proporción de satisfechos/no satisfechos en ambos grupos.
# random_state=42 hace que la división sea siempre la misma (resultados repetibles).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nDatos de entrenamiento: {len(X_train)} | Datos de prueba: {len(X_test)}")

# ¿Por qué escalar? KNN mide DISTANCIAS entre clientes.
# La edad va de 18 a 70 y la calidad de 1 a 5: sin escalar, la edad "pesaría" más
# solo por tener números más grandes. StandardScaler deja todas las columnas
# en la misma escala (media 0 y desviación 1).
# OJO: el escalador se ajusta (fit) SOLO con los datos de entrenamiento.
escalador = StandardScaler()
X_train_esc = escalador.fit_transform(X_train)
X_test_esc = escalador.transform(X_test)

# =============================================================================
# 3. MODELO KNN: BUSCAR EL MEJOR VALOR DE k (scikit-learn)
# =============================================================================
print("\n--- Probando valores de k del 1 al 20 ---")
valores_k = range(1, 21)
exactitudes = []

for k in valores_k:
    modelo_k = KNeighborsClassifier(n_neighbors=k)
    modelo_k.fit(X_train_esc, y_train)
    exactitud = accuracy_score(y_test, modelo_k.predict(X_test_esc))
    exactitudes.append(exactitud)
    print(f"k = {k:2d}  ->  exactitud = {exactitud:.4f}")

# Elegimos el k con mayor exactitud (si hay empate, se queda el k más pequeño)
mejor_exactitud = max(exactitudes)
mejor_k = list(valores_k)[exactitudes.index(mejor_exactitud)]
print(f"\n>>> Mejor k = {mejor_k} con exactitud = {mejor_exactitud:.4f}")

# Entrenamos el modelo final con el mejor k
modelo = KNeighborsClassifier(n_neighbors=mejor_k)
modelo.fit(X_train_esc, y_train)
y_pred = modelo.predict(X_test_esc)

# Métricas
print(f"\nAccuracy (exactitud): {accuracy_score(y_test, y_pred):.4f}"
      f"  ->  acierta el {accuracy_score(y_test, y_pred) * 100:.1f}% de los casos de prueba")

matriz = confusion_matrix(y_test, y_pred)
print("\n--- Matriz de confusión ---")
print("               Predicho: No   Predicho: Sí")
print(f"Real: No       {matriz[0][0]:>10}   {matriz[0][1]:>12}")
print(f"Real: Sí       {matriz[1][0]:>10}   {matriz[1][1]:>12}")

print("\n--- Reporte de clasificación ---")
print(classification_report(y_test, y_pred,
                            target_names=["No satisfecho", "Satisfecho"]))

# =============================================================================
# 4. GRÁFICOS (matplotlib) -> se guardan en la carpeta graficos/
# =============================================================================

# 4.1 Exactitud vs. valor de k
plt.figure(figsize=(9, 5))
plt.plot(list(valores_k), exactitudes, marker="o", color="#1f77b4")
plt.scatter([mejor_k], [mejor_exactitud], color="red", s=120, zorder=3,
            label=f"Mejor k = {mejor_k} ({mejor_exactitud:.2f})")
plt.title("Ejercicio 1: Exactitud del modelo según el valor de k")
plt.xlabel("Valor de k (número de vecinos)")
plt.ylabel("Exactitud (accuracy)")
plt.xticks(list(valores_k))
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, "ej1_exactitud_vs_k.png"), dpi=120)
plt.close()

# 4.2 Matriz de confusión
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(matriz, display_labels=["No satisfecho", "Satisfecho"]).plot(
    ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Ejercicio 1: Matriz de confusión (k = {mejor_k})")
ax.set_xlabel("Predicción del modelo")
ax.set_ylabel("Valor real")
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, "ej1_matriz_confusion.png"), dpi=120)
plt.close()

# 4.3 Dispersión de dos variables coloreada por clase
plt.figure(figsize=(8, 6))
no_sat = datos[datos["satisfecho"] == 0]
si_sat = datos[datos["satisfecho"] == 1]
plt.scatter(no_sat["tiempo_espera_min"], no_sat["calidad_atencion"],
            color="#d62728", alpha=0.6, label="No satisfecho (0)")
plt.scatter(si_sat["tiempo_espera_min"], si_sat["calidad_atencion"],
            color="#2ca02c", alpha=0.6, label="Satisfecho (1)")
plt.title("Ejercicio 1: Tiempo de espera vs. calidad de atención")
plt.xlabel("Tiempo de espera (minutos)")
plt.ylabel("Calidad de atención (1 a 5)")
plt.yticks([1, 2, 3, 4, 5])
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, "ej1_dispersion.png"), dpi=120)
plt.close()

print("Gráficos guardados en la carpeta 'graficos/':")
print("  - ej1_exactitud_vs_k.png\n  - ej1_matriz_confusion.png\n  - ej1_dispersion.png")


# =============================================================================
# 5. PREDICCIÓN DE UN CLIENTE NUEVO (por consola)
# =============================================================================
def pedir_numero(mensaje, minimo, maximo):
    """Pide un número por teclado y repite la pregunta si el dato no es válido."""
    while True:
        texto = input(mensaje).strip().replace(",", ".")
        try:
            valor = float(texto)
        except ValueError:
            print("   Dato inválido: escribe solo un número.")
            continue
        if minimo <= valor <= maximo:
            return valor
        print(f"   El valor debe estar entre {minimo} y {maximo}.")


print("\n" + "=" * 70)
print("PREDICCIÓN PARA UN CLIENTE NUEVO")
print("=" * 70)
try:
    edad = pedir_numero("Edad del cliente (18-90): ", 18, 90)
    espera = pedir_numero("Tiempo de espera en minutos (0-180): ", 0, 180)
    calidad = pedir_numero("Calidad de la atención (1-5): ", 1, 5)
    precio = pedir_numero("Precio percibido (1 = barato ... 5 = muy caro): ", 1, 5)
    resolvio = pedir_numero("¿Se resolvió su problema? (1 = sí, 0 = no): ", 0, 1)

    # Creamos una tabla de 1 fila con las mismas columnas que usamos al entrenar
    nuevo = pd.DataFrame([[edad, espera, calidad, precio, resolvio]], columns=COLUMNAS_X)
    nuevo_esc = escalador.transform(nuevo)  # Se escala igual que los datos de entrenamiento

    prediccion = modelo.predict(nuevo_esc)[0]
    probabilidad = modelo.predict_proba(nuevo_esc)[0][1]  # % de vecinos satisfechos

    resultado = "SATISFECHO" if prediccion == 1 else "NO SATISFECHO"
    print(f"\nResultado: el cliente estaría {resultado}")
    print(f"({probabilidad * 100:.0f}% de sus {mejor_k} vecinos más cercanos estaban satisfechos)")
except (EOFError, KeyboardInterrupt):
    print("\nPredicción cancelada.")
