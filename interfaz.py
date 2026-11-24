# -*- coding: utf-8 -*-
"""
PRÁCTICA CALIFICADA - SENATI
INTERFAZ GRÁFICA para los dos ejercicios de KNN.

Ejecutar con:
    python interfaz.py       -> abre en el Ejercicio 1
    python interfaz.py 2     -> abre en el Ejercicio 2
    python interfaz.py 3     -> abre en la Comparación

Herramientas:
  - tkinter (viene incluido con Python) -> ventanas, botones y cajas de texto
  - pandas       -> leer los CSV
  - scikit-learn -> entrenar el modelo KNN (igual que en los ejercicios)
  - matplotlib   -> gráficos dentro de la ventana

La ventana tiene 3 pestañas:
  1. Ejercicio 1: Satisfacción del cliente
  2. Ejercicio 2: Inversión en software
  3. Comparación: los dos modelos lado a lado
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox

import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  # Permite dibujar los gráficos dentro de tkinter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)

CARPETA = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# PALETA DE COLORES Y FUENTES (cambia aquí para personalizar la interfaz)
# =============================================================================
AZUL_OSCURO = "#0f2a4a"   # cabecera
AZUL = "#1d6fd8"          # botón principal y acentos
AZUL_CLARO = "#e8f1fd"
FONDO = "#eef2f7"         # fondo general
TARJETA = "#ffffff"       # fondo de las tarjetas
BORDE = "#d9e1ec"
TEXTO = "#1f2937"
TEXTO_SUAVE = "#6b7280"
VERDE = "#16a34a"         # clase "sí"
ROJO = "#dc2626"          # clase "no"
NARANJA = "#f59e0b"       # Ejercicio 2 en la comparación

FUENTE = "Segoe UI"

# Estilo general de los gráficos de matplotlib (más limpio)
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#9ca3af",
    "axes.titleweight": "bold",
    "axes.titlesize": 11,
    "axes.titlepad": 10,
    "grid.color": "#e5e7eb",
    "figure.facecolor": TARJETA,
    "axes.facecolor": TARJETA,
})

# =============================================================================
# CONFIGURACIÓN DE CADA EJERCICIO
# Así usamos el MISMO código para los dos ejercicios: solo cambian estos datos.
# Cada campo: (columna, texto que ve el usuario, mínimo, máximo, valor inicial)
# =============================================================================
EJERCICIOS = [
    {
        "titulo": "Ejercicio 1 · Satisfacción del cliente",
        "corto": "Ej. 1 Satisfacción",
        "pregunta": "¿El cliente está satisfecho con el servicio?",
        "icono": "😊",
        "csv": "clientes_satisfaccion.csv",
        "objetivo": "satisfecho",
        "clases": ["No satisfecho", "Satisfecho"],
        "texto_si": "SATISFECHO",
        "texto_no": "NO SATISFECHO",
        "campos": [
            ("edad", "Edad del cliente", 18, 90, 30),
            ("tiempo_espera_min", "Tiempo de espera (min)", 0, 180, 10),
            ("calidad_atencion", "Calidad de atención (1-5)", 1, 5, 4),
            ("precio_percibido", "Precio percibido (1 barato - 5 caro)", 1, 5, 3),
            ("resolvio_problema", "¿Resolvió su problema?", 0, 1, 1),
        ],
        "dispersion": ("tiempo_espera_min", "calidad_atencion"),
        "log_x": False,
        "color": AZUL,
    },
    {
        "titulo": "Ejercicio 2 · Inversión en software",
        "corto": "Ej. 2 Inversión",
        "pregunta": "¿El empresario invierte en un software de automatización?",
        "icono": "💼",
        "csv": "empresarios_inversion.csv",
        "objetivo": "invierte",
        "clases": ["No invierte", "Invierte"],
        "texto_si": "INVIERTE",
        "texto_no": "NO INVIERTE",
        "campos": [
            ("edad", "Edad del empresario", 18, 90, 40),
            ("tamano_empresa", "Tamaño de empresa (n.º empleados)", 1, 5000, 30),
            ("ingresos_mensuales", "Ingresos mensuales (S/)", 0, 100_000_000, 60000),
            ("procesos_manuales", "Procesos manuales (1-10)", 1, 10, 7),
            ("conoce_automatizacion", "¿Conoce automatización?", 0, 1, 1),
        ],
        "dispersion": ("tamano_empresa", "procesos_manuales"),
        "log_x": True,  # hay empresas de 2 y de 400 empleados
        "color": NARANJA,
    },
]


# =============================================================================
# PARTE DE MACHINE LEARNING (los mismos pasos que en los scripts)
# =============================================================================
def entrenar_modelo(config):
    """Carga el CSV, escala los datos, prueba k=1..20 y entrena el mejor KNN."""
    datos = pd.read_csv(os.path.join(CARPETA, "datos", config["csv"])).dropna()
    columnas = [campo[0] for campo in config["campos"]]
    X = datos[columnas]
    y = datos[config["objetivo"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    escalador = StandardScaler()
    X_train_esc = escalador.fit_transform(X_train)
    X_test_esc = escalador.transform(X_test)

    valores_k = list(range(1, 21))
    exactitudes = []
    for k in valores_k:
        m = KNeighborsClassifier(n_neighbors=k).fit(X_train_esc, y_train)
        exactitudes.append(accuracy_score(y_test, m.predict(X_test_esc)))

    mejor_exactitud = max(exactitudes)
    mejor_k = valores_k[exactitudes.index(mejor_exactitud)]

    modelo = KNeighborsClassifier(n_neighbors=mejor_k).fit(X_train_esc, y_train)
    y_pred = modelo.predict(X_test_esc)
    matriz = confusion_matrix(y_test, y_pred)

    return {
        "datos": datos, "columnas": columnas, "escalador": escalador, "modelo": modelo,
        "valores_k": valores_k, "exactitudes": exactitudes,
        "mejor_k": mejor_k, "mejor_exactitud": mejor_exactitud,
        "matriz": matriz,
        "aciertos": int(matriz.trace()), "errores": int(matriz.sum() - matriz.trace()),
        "reporte": classification_report(y_test, y_pred, target_names=config["clases"]),
        "reporte_dict": classification_report(y_test, y_pred, output_dict=True,
                                              target_names=config["clases"]),
        "n_train": len(X_train), "n_test": len(X_test),
    }


# =============================================================================
# ESTILOS (ttk) -> le dan el aspecto moderno a botones, pestañas y tablas
# =============================================================================
def configurar_estilos(ventana):
    estilo = ttk.Style(ventana)
    estilo.theme_use("clam")  # tema que permite cambiar colores fácilmente

    estilo.configure(".", background=FONDO, foreground=TEXTO, font=(FUENTE, 10))
    estilo.configure("Tarjeta.TFrame", background=TARJETA)
    estilo.configure("Tarjeta.TLabel", background=TARJETA, foreground=TEXTO)
    estilo.configure("Suave.TLabel", background=TARJETA, foreground=TEXTO_SUAVE, font=(FUENTE, 9))
    estilo.configure("TituloTarjeta.TLabel", background=TARJETA, foreground=TEXTO,
                     font=(FUENTE, 11, "bold"))

    # Pestañas principales
    estilo.configure("Principal.TNotebook", background=FONDO, borderwidth=0, tabmargins=(12, 8, 12, 0))
    estilo.configure("Principal.TNotebook.Tab", padding=(18, 8), font=(FUENTE, 10, "bold"),
                     background="#dbe3ee", foreground=TEXTO_SUAVE, borderwidth=0)
    estilo.map("Principal.TNotebook.Tab",
               background=[("selected", TARJETA)], foreground=[("selected", AZUL)])

    # Sub-pestañas de los gráficos
    estilo.configure("Graficos.TNotebook", background=TARJETA, borderwidth=0)
    estilo.configure("Graficos.TNotebook.Tab", padding=(12, 5), font=(FUENTE, 9),
                     background="#f1f5f9", foreground=TEXTO_SUAVE, borderwidth=0)
    estilo.map("Graficos.TNotebook.Tab",
               background=[("selected", AZUL_CLARO)], foreground=[("selected", AZUL)])

    # Botones
    estilo.configure("Primario.TButton", font=(FUENTE, 10, "bold"), foreground="white",
                     background=AZUL, borderwidth=0, padding=(10, 8))
    estilo.map("Primario.TButton", background=[("active", "#155bb5")])
    estilo.configure("Secundario.TButton", font=(FUENTE, 10), foreground=TEXTO,
                     background="#e5e7eb", borderwidth=0, padding=(10, 8))
    estilo.map("Secundario.TButton", background=[("active", "#d1d5db")])

    # Cajas de texto y listas desplegables
    estilo.configure("TEntry", fieldbackground="#f9fafb", bordercolor=BORDE,
                     lightcolor=BORDE, padding=5)
    estilo.map("TEntry", bordercolor=[("focus", AZUL)], lightcolor=[("focus", AZUL)])
    estilo.configure("TCombobox", fieldbackground="#f9fafb", bordercolor=BORDE, padding=4,
                     arrowcolor=AZUL, background="#f9fafb")
    estilo.map("TCombobox", fieldbackground=[("readonly", "#f9fafb")],
               background=[("readonly", "#f9fafb")], foreground=[("readonly", TEXTO)])

    # Tabla de la comparación
    estilo.configure("Treeview", background=TARJETA, fieldbackground=TARJETA,
                     rowheight=24, font=(FUENTE, 10), bordercolor=BORDE, borderwidth=0)
    estilo.configure("Treeview.Heading", background=AZUL_OSCURO, foreground="white",
                     font=(FUENTE, 10, "bold"), relief="flat", padding=6)
    estilo.map("Treeview.Heading", background=[("active", AZUL_OSCURO)])


# =============================================================================
# PIEZAS REUTILIZABLES DE LA INTERFAZ
# =============================================================================
def crear_tarjeta(padre, titulo=None):
    """Crea un recuadro blanco con borde suave (una 'tarjeta') y lo devuelve."""
    borde = tk.Frame(padre, bg=BORDE)  # el marco gris de 1 px hace de borde
    interior = ttk.Frame(borde, style="Tarjeta.TFrame", padding=14)
    interior.pack(fill="both", expand=True, padx=1, pady=1)
    if titulo:
        ttk.Label(interior, text=titulo, style="TituloTarjeta.TLabel").pack(anchor="w", pady=(0, 8))
    return borde, interior


def crear_indicador(padre, titulo, valor, color):
    """Pequeña tarjeta con un número grande (ej. 'Exactitud 90 %')."""
    caja = tk.Frame(padre, bg=TARJETA, highlightbackground=BORDE, highlightthickness=1)
    tk.Frame(caja, bg=color, width=5).pack(side="left", fill="y")  # franja de color
    contenido = tk.Frame(caja, bg=TARJETA, padx=12, pady=8)
    contenido.pack(side="left", fill="both", expand=True)
    tk.Label(contenido, text=titulo, bg=TARJETA, fg=TEXTO_SUAVE, font=(FUENTE, 9)).pack(anchor="w")
    tk.Label(contenido, text=valor, bg=TARJETA, fg=TEXTO, font=(FUENTE, 18, "bold")).pack(anchor="w")
    return caja


def agregar_figura(padre, figura):
    """Mete una figura de matplotlib dentro de un marco de tkinter."""
    canvas = FigureCanvasTkAgg(figura, master=padre)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    return canvas


# =============================================================================
# PESTAÑA DE UN EJERCICIO
# =============================================================================
class PestanaEjercicio(ttk.Frame):
    def __init__(self, padre, config, resultados):
        super().__init__(padre, padding=14)
        self.config_ej = config
        self.res = resultados
        self.entradas = {}
        self.punto_nuevo = None  # el punto del caso nuevo en el gráfico de dispersión

        # Encabezado de la pestaña: pregunta que responde el modelo
        ttk.Label(self, text=f"{config['icono']}  {config['pregunta']}",
                  font=(FUENTE, 13, "bold")).pack(anchor="w", pady=(0, 10))

        cuerpo = ttk.Frame(self)
        cuerpo.pack(fill="both", expand=True)

        # Columna izquierda: indicadores + formulario + reporte
        izquierda = ttk.Frame(cuerpo, width=430)
        izquierda.pack(side="left", fill="y", padx=(0, 14))
        # Columna derecha: gráficos
        derecha = ttk.Frame(cuerpo)
        derecha.pack(side="left", fill="both", expand=True)

        self.crear_indicadores(izquierda)
        self.crear_formulario(izquierda)
        self.crear_graficos(derecha)

    # ---------- Fila de indicadores (KPI) ----------
    def crear_indicadores(self, padre):
        r = self.res
        fila = ttk.Frame(padre)
        fila.pack(fill="x", pady=(0, 12))
        indicadores = [
            ("Mejor k", str(r["mejor_k"]), AZUL),
            ("Exactitud", f"{r['mejor_exactitud'] * 100:.1f} %", VERDE),
            ("Datos", str(len(r["datos"])), NARANJA),
        ]
        for i, (titulo, valor, color) in enumerate(indicadores):
            crear_indicador(fila, titulo, valor, color).grid(
                row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 6, 0))
            fila.columnconfigure(i, weight=1)

    # ---------- Formulario para el caso nuevo ----------
    def crear_formulario(self, padre):
        borde, marco = crear_tarjeta(padre, "Nuevo caso a predecir")
        borde.pack(fill="x")

        campos = ttk.Frame(marco, style="Tarjeta.TFrame")
        campos.pack(fill="x")
        for fila, (columna, texto, minimo, maximo, inicial) in enumerate(self.config_ej["campos"]):
            ttk.Label(campos, text=texto, style="Tarjeta.TLabel").grid(
                row=fila, column=0, sticky="w", pady=4)
            if (minimo, maximo) == (0, 1):
                # Para preguntas de sí/no usamos una lista desplegable
                caja = ttk.Combobox(campos, values=["1 = Sí", "0 = No"], width=14, state="readonly")
                caja.set("1 = Sí" if inicial == 1 else "0 = No")
            else:
                caja = ttk.Entry(campos, width=17)
                caja.insert(0, str(inicial))
            caja.grid(row=fila, column=1, padx=(10, 0), pady=4, sticky="e")
            self.entradas[columna] = (caja, texto, minimo, maximo)
        campos.columnconfigure(0, weight=1)

        # Botones: "Predecir" y "Limpiar"
        botones = ttk.Frame(marco, style="Tarjeta.TFrame")
        botones.pack(fill="x", pady=(12, 10))
        ttk.Button(botones, text="🔍  Predecir", style="Primario.TButton",
                   command=self.predecir).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(botones, text="🧹  Limpiar", style="Secundario.TButton",
                   command=self.limpiar).pack(side="left", fill="x", expand=True, padx=(5, 0))

        # Recuadro del resultado
        self.lbl_resultado = tk.Label(marco, text="Ingresa los datos y presiona Predecir",
                                      font=(FUENTE, 10), bg="#f3f4f6", fg=TEXTO_SUAVE, pady=12)
        self.lbl_resultado.pack(fill="x")
        self.lbl_detalle = ttk.Label(marco, text="", style="Suave.TLabel")
        self.lbl_detalle.pack(pady=(6, 0))

        # Enter = Predecir (más cómodo)
        for caja, *_ in self.entradas.values():
            caja.bind("<Return>", lambda _e: self.predecir())

    # ---------- Reporte de clasificación (sub-pestaña de texto) ----------
    def crear_reporte(self, cuaderno):
        r = self.res
        marco = ttk.Frame(cuaderno, style="Tarjeta.TFrame", padding=20)
        cuaderno.add(marco, text="📋 Reporte")
        ttk.Label(marco, text="Reporte de clasificación (datos de prueba)",
                  style="TituloTarjeta.TLabel").pack(anchor="w")
        ttk.Label(marco, text=f"Entrenamiento: {r['n_train']} datos  ·  Prueba: {r['n_test']} datos  ·  "
                              f"Aciertos: {r['aciertos']} de {r['n_test']}  ·  Errores: {r['errores']}",
                  style="Suave.TLabel").pack(anchor="w", pady=(2, 10))
        texto = tk.Text(marco, height=9, font=("Consolas", 12), relief="flat",
                        bg="#f9fafb", fg=TEXTO, padx=14, pady=10)
        texto.insert("1.0", r["reporte"])
        texto.configure(state="disabled")  # solo lectura
        texto.pack(fill="x")
        ayuda = ("¿Cómo se lee?\n"
                 "• precision: de los casos que el modelo predijo como esa clase, cuántos acertó.\n"
                 "• recall: de los casos reales de esa clase, cuántos encontró el modelo.\n"
                 "• f1-score: combina precision y recall (más cerca de 1 = mejor).\n"
                 "• support: cuántos casos de esa clase había en los datos de prueba.")
        tk.Label(marco, text=ayuda, bg=AZUL_CLARO, fg=TEXTO, font=(FUENTE, 10), justify="left",
                 anchor="w", padx=14, pady=10).pack(fill="x", pady=(14, 0))

    # ---------- Gráficos (3 sub-pestañas) ----------
    def crear_graficos(self, padre):
        r, cfg = self.res, self.config_ej
        borde, marco = crear_tarjeta(padre)
        borde.pack(fill="both", expand=True)
        cuaderno = ttk.Notebook(marco, style="Graficos.TNotebook")
        cuaderno.pack(fill="both", expand=True)

        # 1) Exactitud vs k
        fig1 = Figure(figsize=(6.5, 4.5), dpi=100)
        ax1 = fig1.add_subplot(111)
        ax1.plot(r["valores_k"], r["exactitudes"], marker="o", color=AZUL, linewidth=2)
        ax1.fill_between(r["valores_k"], r["exactitudes"], min(r["exactitudes"]) - 0.02,
                         color=AZUL, alpha=0.08)
        ax1.scatter([r["mejor_k"]], [r["mejor_exactitud"]], color=ROJO, s=140, zorder=3,
                    label=f"Mejor k = {r['mejor_k']} ({r['mejor_exactitud'] * 100:.0f} %)")
        ax1.set_title("Exactitud según el valor de k")
        ax1.set_xlabel("k (número de vecinos)")
        ax1.set_ylabel("Exactitud")
        ax1.set_xticks(r["valores_k"])
        ax1.grid(alpha=0.6)
        ax1.legend(frameon=False)
        fig1.tight_layout()
        self.agregar_subpestana(cuaderno, fig1, "📈 Exactitud vs k")

        # 2) Matriz de confusión
        fig2 = Figure(figsize=(6.5, 4.5), dpi=100)
        ax2 = fig2.add_subplot(111)
        ConfusionMatrixDisplay(r["matriz"], display_labels=cfg["clases"]).plot(
            ax=ax2, cmap="Blues", colorbar=False)
        ax2.set_title(f"Matriz de confusión (k = {r['mejor_k']})")
        ax2.set_xlabel("Predicción del modelo")
        ax2.set_ylabel("Valor real")
        fig2.tight_layout()
        self.agregar_subpestana(cuaderno, fig2, "🔲 Matriz de confusión")

        # 3) Dispersión coloreada por clase (aquí también se dibuja el caso nuevo)
        col_x, col_y = cfg["dispersion"]
        datos = r["datos"]
        nombres = {campo[0]: campo[1] for campo in cfg["campos"]}
        fig3 = Figure(figsize=(6.5, 4.5), dpi=100)
        self.ax3 = fig3.add_subplot(111)
        for valor, color, nombre in [(0, ROJO, cfg["clases"][0]), (1, VERDE, cfg["clases"][1])]:
            grupo = datos[datos[cfg["objetivo"]] == valor]
            # En la leyenda mostramos cuántos datos hay de cada clase
            self.ax3.scatter(grupo[col_x], grupo[col_y], color=color, alpha=0.5,
                             edgecolor="white", linewidth=0.5, s=45,
                             label=f"{nombre}: {len(grupo)} datos")
        if cfg["log_x"]:
            self.ax3.set_xscale("log")
        self.ax3.set_title(f"{nombres[col_x]}  vs.  {nombres[col_y]}")
        self.ax3.set_xlabel(nombres[col_x] + (" (escala logarítmica)" if cfg["log_x"] else ""))
        self.ax3.set_ylabel(nombres[col_y])
        self.ax3.grid(alpha=0.6)
        self.ax3.legend(loc="best", framealpha=0.95)
        fig3.tight_layout()
        self.canvas_dispersion = self.agregar_subpestana(cuaderno, fig3, "🔵 Dispersión")

        # 4) Reporte de clasificación en texto
        self.crear_reporte(cuaderno)

    def agregar_subpestana(self, cuaderno, figura, titulo):
        marco = ttk.Frame(cuaderno, style="Tarjeta.TFrame")
        cuaderno.add(marco, text=titulo)
        return agregar_figura(marco, figura)

    # ---------- Botón "Limpiar" ----------
    def limpiar(self):
        """Borra lo escrito en el formulario, el resultado y la estrella del gráfico."""
        for caja, *_ in self.entradas.values():
            if isinstance(caja, ttk.Combobox):
                caja.set("")            # lista desplegable sin selección
            else:
                caja.delete(0, tk.END)  # caja de texto vacía

        self.lbl_resultado.config(text="Ingresa los datos y presiona Predecir",
                                  font=(FUENTE, 10), fg=TEXTO_SUAVE, bg="#f3f4f6")
        self.lbl_detalle.config(text="")

        if self.punto_nuevo is not None:  # quitamos la estrella del caso anterior
            self.punto_nuevo.remove()
            self.punto_nuevo = None
            self.ax3.legend(loc="best", framealpha=0.95)
            self.canvas_dispersion.draw()

        # Ponemos el cursor en la primera caja para escribir de nuevo
        next(iter(self.entradas.values()))[0].focus_set()

    # ---------- Botón "Predecir" ----------
    def predecir(self):
        valores = {}
        for columna, (caja, texto, minimo, maximo) in self.entradas.items():
            contenido = caja.get().strip()
            if contenido == "":
                messagebox.showwarning("Falta un dato", f"Completa el campo: '{texto}'.")
                caja.focus_set()
                return
            contenido = contenido.split("=")[0].strip()  # "1 = Sí" -> "1"
            try:
                valor = float(contenido.replace(",", "."))
            except ValueError:
                messagebox.showerror("Dato inválido", f"'{texto}' debe ser un número.")
                caja.focus_set()
                return
            if not minimo <= valor <= maximo:
                messagebox.showerror("Fuera de rango",
                                     f"'{texto}' debe estar entre {minimo} y {maximo}.")
                caja.focus_set()
                return
            valores[columna] = valor

        r, cfg = self.res, self.config_ej
        nuevo = pd.DataFrame([valores], columns=r["columnas"])
        nuevo_esc = r["escalador"].transform(nuevo)  # mismo escalado que en el entrenamiento
        prediccion = r["modelo"].predict(nuevo_esc)[0]
        prob_si = r["modelo"].predict_proba(nuevo_esc)[0][1]

        if prediccion == 1:
            texto_resultado, color = cfg["texto_si"], VERDE
        else:
            texto_resultado, color = cfg["texto_no"], ROJO
        self.lbl_resultado.config(text=("✔  " if prediccion == 1 else "✖  ") + texto_resultado,
                                  font=(FUENTE, 17, "bold"), fg="white", bg=color)
        self.lbl_detalle.config(
            text=f"{prob_si * 100:.0f} % de sus {r['mejor_k']} vecinos más cercanos son "
                 f"\"{cfg['clases'][1]}\"")

        # Dibujamos el caso nuevo como una ESTRELLA del color del resultado.
        # No se agrega al CSV: solo se muestra dónde cae.
        col_x, col_y = cfg["dispersion"]
        if self.punto_nuevo is not None:
            self.punto_nuevo.remove()
        self.punto_nuevo = self.ax3.scatter(
            [valores[col_x]], [valores[col_y]], marker="*", s=550, color=color,
            edgecolor="black", linewidth=1.5, zorder=5,
            label=f"Caso nuevo ★: {texto_resultado}")
        self.ax3.legend(loc="best", framealpha=0.95)
        self.canvas_dispersion.draw()


# =============================================================================
# PESTAÑA DE COMPARACIÓN DE LOS DOS EJERCICIOS
# =============================================================================
class PestanaComparacion(ttk.Frame):
    def __init__(self, padre, lista_resultados):
        super().__init__(padre, padding=14)
        self.lista = lista_resultados  # [(config, resultados), (config, resultados)]

        ttk.Label(self, text="⚖  Comparación de los dos modelos KNN",
                  font=(FUENTE, 13, "bold")).pack(anchor="w", pady=(0, 10))

        cuerpo = ttk.Frame(self)
        cuerpo.pack(fill="both", expand=True)
        izquierda = ttk.Frame(cuerpo)
        izquierda.pack(side="left", fill="y", padx=(0, 14))
        derecha = ttk.Frame(cuerpo)
        derecha.pack(side="left", fill="both", expand=True)

        self.crear_tabla(izquierda)
        self.crear_conclusion(izquierda)
        self.crear_graficos(derecha)

    def metricas(self, res):
        """Saca del reporte las métricas promedio (macro avg) de un modelo."""
        promedio = res["reporte_dict"]["macro avg"]
        return {
            "Exactitud": res["mejor_exactitud"],
            "Precisión": promedio["precision"],
            "Recall": promedio["recall"],
            "F1-score": promedio["f1-score"],
        }

    # ---------- Tabla comparativa ----------
    def crear_tabla(self, padre):
        borde, marco = crear_tarjeta(padre, "Tabla comparativa")
        borde.pack(fill="x")

        (c1, r1), (c2, r2) = self.lista
        m1, m2 = self.metricas(r1), self.metricas(r2)

        tabla = ttk.Treeview(marco, columns=("ind", "e1", "e2"), show="headings", height=11)
        tabla.heading("ind", text="Indicador")
        tabla.heading("e1", text=c1["corto"])
        tabla.heading("e2", text=c2["corto"])
        tabla.column("ind", width=170, anchor="w")
        tabla.column("e1", width=130, anchor="center")
        tabla.column("e2", width=130, anchor="center")

        def pct(x):
            return f"{x * 100:.1f} %"

        filas = [
            ("Total de datos", len(r1["datos"]), len(r2["datos"])),
            ("Entrenamiento / prueba", f"{r1['n_train']} / {r1['n_test']}",
             f"{r2['n_train']} / {r2['n_test']}"),
            ("Variables usadas", len(r1["columnas"]), len(r2["columnas"])),
            ("% de clase \"sí\"",
             pct(r1["datos"][c1["objetivo"]].mean()), pct(r2["datos"][c2["objetivo"]].mean())),
            ("Mejor k", r1["mejor_k"], r2["mejor_k"]),
            ("Exactitud", pct(m1["Exactitud"]), pct(m2["Exactitud"])),
            ("Precisión (prom.)", pct(m1["Precisión"]), pct(m2["Precisión"])),
            ("Recall (prom.)", pct(m1["Recall"]), pct(m2["Recall"])),
            ("F1-score (prom.)", pct(m1["F1-score"]), pct(m2["F1-score"])),
            ("Aciertos en prueba", f"{r1['aciertos']} de {r1['n_test']}",
             f"{r2['aciertos']} de {r2['n_test']}"),
            ("Errores en prueba", r1["errores"], r2["errores"]),
        ]
        tabla.tag_configure("par", background="#f5f8fc")
        for i, fila in enumerate(filas):
            tabla.insert("", "end", values=fila, tags=("par",) if i % 2 else ())
        tabla.pack(fill="x")

    # ---------- Conclusión automática ----------
    def crear_conclusion(self, padre):
        (c1, r1), (c2, r2) = self.lista
        e1, e2 = r1["mejor_exactitud"], r2["mejor_exactitud"]
        if e1 == e2:
            ganador = "Los dos modelos tienen la misma exactitud."
        else:
            mejor, peor = ((c1, r1), (c2, r2)) if e1 > e2 else ((c2, r2), (c1, r1))
            diferencia = abs(e1 - e2) * 100
            ganador = (f"El modelo del {mejor[0]['corto']} es más exacto: acierta "
                       f"{mejor[1]['mejor_exactitud'] * 100:.1f} % frente a "
                       f"{peor[1]['mejor_exactitud'] * 100:.1f} % ({diferencia:.1f} puntos más).")

        texto = (
            f"• {ganador}\n"
            f"• Mejor k: {r1['mejor_k']} vecinos (Ej. 1) y {r2['mejor_k']} vecinos (Ej. 2). "
            "Un k pequeño es más sensible a casos raros; uno grande decide de forma más estable.\n"
            "• Ambos superan de lejos el 50 % de adivinar al azar: KNN sí encontró patrones."
        )
        borde, marco = crear_tarjeta(padre, "💡 Conclusión")
        borde.pack(fill="both", expand=True, pady=(12, 0))
        tk.Label(marco, text=texto, bg=TARJETA, fg=TEXTO, font=(FUENTE, 10),
                 justify="left", wraplength=420, anchor="nw").pack(fill="both", expand=True)

    # ---------- Gráficos comparativos ----------
    def crear_graficos(self, padre):
        (c1, r1), (c2, r2) = self.lista
        borde, marco = crear_tarjeta(padre)
        borde.pack(fill="both", expand=True)

        fig = Figure(figsize=(8, 6), dpi=100)
        rejilla = fig.add_gridspec(2, 2, height_ratios=[1, 1])

        # (a) Exactitud vs k de los dos modelos en el mismo gráfico
        ax1 = fig.add_subplot(rejilla[0, 0])
        for cfg, res in self.lista:
            ax1.plot(res["valores_k"], res["exactitudes"], marker="o", markersize=4,
                     linewidth=2, color=cfg["color"], label=cfg["corto"])
            ax1.scatter([res["mejor_k"]], [res["mejor_exactitud"]], s=110,
                        color=cfg["color"], edgecolor="black", zorder=3)
        ax1.set_title("Exactitud vs k")
        ax1.set_xlabel("k")
        ax1.set_ylabel("Exactitud")
        ax1.set_xticks(range(1, 21, 2))
        ax1.grid(alpha=0.6)
        ax1.legend(frameon=False, fontsize=8)

        # (b) Barras agrupadas con las 4 métricas
        ax2 = fig.add_subplot(rejilla[0, 1])
        nombres = list(self.metricas(r1).keys())
        posiciones = range(len(nombres))
        ancho = 0.38
        for desplazamiento, (cfg, res) in zip((-ancho / 2, ancho / 2), self.lista):
            valores = list(self.metricas(res).values())
            barras = ax2.bar([p + desplazamiento for p in posiciones], valores, ancho,
                             color=cfg["color"], label=cfg["corto"])
            ax2.bar_label(barras, labels=[f"{v * 100:.0f}" for v in valores],
                          fontsize=7, padding=2)
        ax2.set_xticks(list(posiciones))
        ax2.set_xticklabels(nombres)
        ax2.set_ylim(0, 1.25)
        ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax2.set_title("Métricas (en %)")
        ax2.grid(axis="y", alpha=0.6)
        ax2.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)

        # (c) y (d) Matrices de confusión lado a lado
        for columna, (cfg, res), mapa in zip((0, 1), self.lista, ("Blues", "Oranges")):
            ax = fig.add_subplot(rejilla[1, columna])
            ConfusionMatrixDisplay(res["matriz"], display_labels=cfg["clases"]).plot(
                ax=ax, cmap=mapa, colorbar=False)
            ax.set_title(f"Matriz de confusión · {cfg['corto']}")
            ax.set_xlabel("Predicción")
            ax.set_ylabel("Real")
            ax.tick_params(labelsize=8)

        fig.tight_layout(h_pad=2.5)
        agregar_figura(marco, fig)


# =============================================================================
# VENTANA PRINCIPAL
# =============================================================================
def main():
    ventana = tk.Tk()
    ventana.title("Práctica Calificada SENATI · Clasificación con KNN")
    ventana.geometry("1280x760")
    ventana.minsize(1100, 680)
    ventana.configure(bg=FONDO)
    try:
        ventana.state("zoomed")  # En Windows abre la ventana maximizada
    except tk.TclError:
        pass
    configurar_estilos(ventana)

    # ----- Cabecera azul -----
    cabecera = tk.Frame(ventana, bg=AZUL_OSCURO, padx=24, pady=14)
    cabecera.pack(fill="x")
    tk.Label(cabecera, text="Clasificación con KNN", bg=AZUL_OSCURO, fg="white",
             font=(FUENTE, 18, "bold")).pack(anchor="w")
    tk.Label(cabecera, text="SENATI · Práctica Calificada  ·  K vecinos más cercanos  ·  "
                            "80 % entrenamiento / 20 % prueba  ·  datos escalados con StandardScaler",
             bg=AZUL_OSCURO, fg="#a9c3e6", font=(FUENTE, 10)).pack(anchor="w")

    # ----- Entrenamos los dos modelos una sola vez -----
    lista_resultados = [(cfg, entrenar_modelo(cfg)) for cfg in EJERCICIOS]

    # ----- Pestañas -----
    pestanas = ttk.Notebook(ventana, style="Principal.TNotebook")
    pestanas.pack(fill="both", expand=True)
    for cfg, res in lista_resultados:
        pestanas.add(PestanaEjercicio(pestanas, cfg, res), text=f"{cfg['icono']}  {cfg['titulo']}")
    pestanas.add(PestanaComparacion(pestanas, lista_resultados), text="⚖  Comparación")

    # "python interfaz.py 2" abre en el Ejercicio 2; "3" abre en la Comparación
    if len(sys.argv) > 1 and sys.argv[1] in ("1", "2", "3"):
        pestanas.select(int(sys.argv[1]) - 1)

    ventana.mainloop()


if __name__ == "__main__":
    main()
