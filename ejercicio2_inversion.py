# -*- coding: utf-8 -*-
"""
PRÁCTICA CALIFICADA - SENATI
EJERCICIO 2: ¿El empresario INVIERTE o NO en un software para automatizar procesos?

Algoritmo: KNN (K-Nearest Neighbors = K Vecinos más Cercanos)
Idea simple: para un empresario nuevo, buscamos los "k" empresarios más
parecidos a él (edad, tamaño de empresa, ingresos, etc.) y vemos qué decidió
la mayoría. Si la mayoría invirtió -> predecimos que invierte.

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
RUTA_CSV = os.path.join(CARPETA, "datos", "empresarios_inversion.csv")
CARPETA_GRAFICOS = os.path.join(CARPETA, "graficos")
os.makedirs(CARPETA_GRAFICOS, exist_ok=True)

# =============================================================================
# 1. CARGAR Y REVISAR LOS DATOS (pandas)
# =============================================================================
print("=" * 70)
print("EJERCICIO 2: PREDICCIÓN DE INVERSIÓN EN SOFTWARE CON KNN")
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

print("\n--- Cantidad de empresarios por clase (1 = invierte, 0 = no) ---")
print(datos["invierte"].value_counts())

# =============================================================================
# 2. PREPARACIÓN DE LOS DATOS (scikit-learn)
# =============================================================================
# X = las características (lo que usamos para predecir)
# y = la variable objetivo (lo que queremos predecir)
COLUMNAS_X = ["edad", "tamano_empresa", "ingresos_mensuales",
              "procesos_manuales", "conoce_automatizacion"]
X = datos[COLUMNAS_X]
y = datos["invierte"]

# Dividimos: 80% para entrenar y 20% para probar.
# stratify=y mantiene la misma proporción de "invierte / no invierte" en ambos grupos.
# random_state=42 hace que la división sea siempre la misma (resultados repetibles).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nDatos de entrenamiento: {len(X_train)} | Datos de prueba: {len(X_test)}")

# ¿Por qué escalar? KNN mide DISTANCIAS entre empresarios.
# Los ingresos están en miles (ej. 45000) y "conoce_automatizacion" es 0 o 1:
# sin escalar, los ingresos dominarían por completo la distancia.
# StandardScaler deja todas las columnas en la misma escala (media 0 y desviación 1).
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
                            target_names=["No invierte", "Invierte"]))

# =============================================================================
# 4. GRÁFICOS (matplotlib) -> se guardan en la carpeta graficos/
# =============================================================================

# 4.1 Exactitud vs. valor de k
plt.figure(figsize=(9, 5))
plt.plot(list(valores_k), exactitudes, marker="o", color="#1f77b4")
plt.scatter([mejor_k], [mejor_exactitud], color="red", s=120, zorder=3,
            label=f"Mejor k = {mejor_k} ({mejor_exactitud:.2f})")
plt.title("Ejercicio 2: Exactitud del modelo según el valor de k")
plt.xlabel("Valor de k (número de vecinos)")
plt.ylabel("Exactitud (accuracy)")
plt.xticks(list(valores_k))
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, "ej2_exactitud_vs_k.png"), dpi=120)
plt.close()

# 4.2 Matriz de confusión
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(matriz, display_labels=["No invierte", "Invierte"]).plot(
    ax=ax, cmap="Greens", colorbar=False)
ax.set_title(f"Ejercicio 2: Matriz de confusión (k = {mejor_k})")
ax.set_xlabel("Predicción del modelo")
ax.set_ylabel("Valor real")
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, "ej2_matriz_confusion.png"), dpi=120)
plt.close()

# 4.3 Dispersión de dos variables coloreada por clase
plt.figure(figsize=(8, 6))
no_inv = datos[datos["invierte"] == 0]
si_inv = datos[datos["invierte"] == 1]
plt.scatter(no_inv["tamano_empresa"], no_inv["procesos_manuales"],
            color="#d62728", alpha=0.6, label="No invierte (0)")
plt.scatter(si_inv["tamano_empresa"], si_inv["procesos_manuales"],
            color="#2ca02c", alpha=0.6, label="Invierte (1)")
plt.xscale("log")  # Escala logarítmica: hay empresas de 2 y de 400 empleados
plt.title("Ejercicio 2: Tamaño de empresa vs. procesos manuales")
plt.xlabel("Tamaño de la empresa (n.º de empleados, escala logarítmica)")
plt.ylabel("Nivel de procesos manuales (1 a 10)")
plt.yticks(range(1, 11))
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, "ej2_dispersion.png"), dpi=120)
plt.close()

print("Gráficos guardados en la carpeta 'graficos/':")
print("  - ej2_exactitud_vs_k.png\n  - ej2_matriz_confusion.png\n  - ej2_dispersion.png")


# =============================================================================
# 5. PREDICCIÓN DE UN EMPRESARIO NUEVO (por consola)
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
print("PREDICCIÓN PARA UN EMPRESARIO NUEVO")
print("=" * 70)
try:
    edad = pedir_numero("Edad del empresario (18-90): ", 18, 90)
    empleados = pedir_numero("Tamaño de la empresa (n.º de empleados, 1-5000): ", 1, 5000)
    ingresos = pedir_numero("Ingresos mensuales en soles (ej. 45000): ", 0, 100_000_000)
    procesos = pedir_numero("Nivel de procesos manuales (1 = pocos ... 10 = muchos): ", 1, 10)
    conoce = pedir_numero("¿Conoce sobre automatización? (1 = sí, 0 = no): ", 0, 1)

    # Creamos una tabla de 1 fila con las mismas columnas que usamos al entrenar
    nuevo = pd.DataFrame([[edad, empleados, ingresos, procesos, conoce]], columns=COLUMNAS_X)
    nuevo_esc = escalador.transform(nuevo)  # Se escala igual que los datos de entrenamiento

    prediccion = modelo.predict(nuevo_esc)[0]
    probabilidad = modelo.predict_proba(nuevo_esc)[0][1]  # % de vecinos que invirtieron

    resultado = "INVIERTE" if prediccion == 1 else "NO INVIERTE"
    print(f"\nResultado: el empresario {resultado} en el software")
    print(f"({probabilidad * 100:.0f}% de sus {mejor_k} vecinos más cercanos invirtieron)")
except (EOFError, KeyboardInterrupt):
    print("\nPredicción cancelada.")
