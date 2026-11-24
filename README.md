# Práctica Calificada SENATI: Clasificación con KNN

Dos ejercicios con el algoritmo **KNN (K-Nearest Neighbors, o K vecinos más cercanos)**, usando:

- **pandas**: cargar y revisar los datos
- **scikit-learn**: dividir los datos, escalarlos y entrenar el modelo
- **matplotlib**: generar los gráficos

## ¿Cómo funciona KNN?

Para clasificar un caso nuevo, el modelo busca los **k** casos más parecidos (los más cercanos) en los datos de entrenamiento y elige la clase que tenga la mayoría de ellos. Como se basa en **distancias**, antes de entrenar hay que **escalar** los datos con `StandardScaler`. Si no se escalan, una columna con números grandes (por ejemplo, los ingresos) pesaría más que otra solo por el tamaño de sus valores.

## Estructura

```
practica_knn_senati/
├── requirements.txt
├── ejercicio1_satisfaccion.py   -> ¿Cliente satisfecho o no?
├── ejercicio2_inversion.py      -> ¿Empresario invierte en software o no?
├── datos/
│   ├── clientes_satisfaccion.csv    (200 filas)
│   └── empresarios_inversion.csv    (200 filas)
└── graficos/                    -> aquí se guardan los PNG al ejecutar
```

## Ejercicio 1: Satisfacción del cliente

| Columna | Significado |
|---|---|
| edad | Edad del cliente |
| tiempo_espera_min | Minutos que esperó para ser atendido |
| calidad_atencion | 1 (mala) a 5 (excelente) |
| precio_percibido | 1 (barato) a 5 (muy caro) |
| resolvio_problema | 1 = sí, 0 = no |
| **satisfecho** | **Objetivo:** 1 = sí, 0 = no |

## Ejercicio 2: Inversión en software de automatización

| Columna | Significado |
|---|---|
| edad | Edad del empresario |
| tamano_empresa | Número de empleados |
| ingresos_mensuales | Ingresos mensuales (soles) |
| procesos_manuales | 1 (pocos) a 10 (casi todo es manual) |
| conoce_automatizacion | 1 = sí, 0 = no |
| **invierte** | **Objetivo:** 1 = sí, 0 = no |

## Instalación y ejecución

```bash
pip install -r requirements.txt
python ejercicio1_satisfaccion.py
python ejercicio2_inversion.py
```

### Interfaz gráfica (opcional)

```bash
python interfaz.py       # abre en el Ejercicio 1
python interfaz.py 2     # abre en el Ejercicio 2
python interfaz.py 3     # abre en la Comparación
```

La ventana tiene 3 pestañas:

- **Ejercicio 1** y **Ejercicio 2**: tarjetas con el mejor k, la exactitud y el total de datos; un formulario con los botones **Predecir** y **Limpiar**; y 4 sub-pestañas: Exactitud vs k, Matriz de confusión, Dispersión y Reporte. El resultado aparece en verde (sí) o rojo (no), y el caso nuevo se marca con una ★ de ese mismo color en el gráfico de dispersión.
- **Comparación**: una tabla con las métricas de los dos modelos, una conclusión automática y gráficos lado a lado: exactitud vs k, barras de métricas y las dos matrices de confusión. La interfaz usa `tkinter`, que viene incluido con Python, así que no hay que instalar nada más.

Al final, cada script pide por consola los datos de un caso nuevo y muestra la predicción.

## Cómo interpretar los resultados

- **Exactitud (accuracy):** porcentaje de casos de prueba que el modelo acertó. Por ejemplo, 0.90 significa que acertó el 90 %.
- **Gráfico "Exactitud vs. k":** muestra cómo cambia la exactitud al probar k = 1 hasta 20. El punto rojo marca el mejor k. Un k muy pequeño se deja engañar por casos raros, y uno muy grande no distingue bien entre clases.
- **Matriz de confusión:**
  - La diagonal (arriba a la izquierda y abajo a la derecha) son los **aciertos**.
  - Fuera de la diagonal están los **errores**: casos de "No" que el modelo predijo como "Sí", y casos de "Sí" que predijo como "No".
- **classification_report:**
  - **precision:** de todos los casos que el modelo predijo como una clase, cuántos eran de esa clase de verdad.
  - **recall:** de todos los casos reales de una clase, cuántos encontró el modelo.
  - **f1-score:** combina precision y recall en un solo número. Cuanto más cerca de 1, mejor.
- **Gráfico de dispersión:** muestra dos variables y pinta cada punto según su clase (verde = sí, rojo = no). Si los colores aparecen agrupados en zonas distintas, KNN podrá clasificar bien.
- **Predicción nueva:** además del resultado, el script muestra qué porcentaje de los k vecinos pertenece a la clase "sí". Ese porcentaje indica qué tan segura es la predicción.

> Nota: los datasets son simulados, pero sus reglas son realistas. Por ejemplo, esperar más tiempo baja la satisfacción, y una empresa más grande con más procesos manuales tiene más probabilidad de invertir.
