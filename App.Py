import streamlit as st
import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. Cargar datos (precio real + predicciones)
# --------------------------------------------------
@st.cache_data
def load_data():
    # si el CSV está en la misma carpeta que app.py
    df = pd.read_csv("amazon_scored_powerbi.csv")
    return df

df = load_data()

# columna de error por si no existe
if "Error_Absoluto" not in df.columns:
    df["Error_Absoluto"] = abs(df["discounted_price"] - df["predicted_discounted_price"])

# --------------------------------------------------
# 2. Configuración de la página
# --------------------------------------------------
st.set_page_config(
    page_title="Predictive Pricing - Amazon",
    layout="wide"
)

st.title("Predicción del Precio con Descuento – Amazon")
st.write(
    """
    Esta aplicación permite explorar las predicciones del modelo de **Gradient Boosting**
    sobre el precio con descuento de productos de Amazon.

    Selecciona una categoría y un producto para comparar el **precio real** vs.
    el **precio estimado por el modelo**, además del error asociado.
    """
)

# --------------------------------------------------
# 3. Filtros (ingreso de datos del usuario)
# --------------------------------------------------
col_filtros, col_resultados = st.columns([1, 2])

with col_filtros:
    st.subheader("Filtros de búsqueda")

    # seleccionar categoría
    categorias = ["(todas)"] + sorted(df["category"].astype(str).unique().tolist())
    categoria = st.selectbox("Categoría", options=categorias)

    if categoria != "(todas)":
        df_filtrado = df[df["category"] == categoria]
    else:
        df_filtrado = df.copy()

    # seleccionar producto
    productos = sorted(df_filtrado["product_name"].astype(str).unique().tolist())
    producto = st.selectbox("Producto", options=productos)

    fila = df_filtrado[df_filtrado["product_name"] == producto].iloc[0]

# --------------------------------------------------
# 4. Métricas para el producto seleccionado
# --------------------------------------------------
with col_resultados:
    st.subheader("Detalle del producto seleccionado")

    precio_real = fila["discounted_price"]
    precio_pred = fila["predicted_discounted_price"]
    error_abs = abs(precio_real - precio_pred)
    error_pct = error_abs / precio_real if precio_real != 0 else np.nan

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precio real (discounted_price)", f"₹ {precio_real:,.2f}")
    m2.metric("Precio predicho por el modelo", f"₹ {precio_pred:,.2f}")
    m3.metric("Error absoluto", f"₹ {error_abs:,.2f}")
    m4.metric("Error porcentual", f"{error_pct*100:,.2f}%")

    st.write("---")
    st.markdown("### Información adicional del producto")
    st.write(f"**Categoría:** {fila['category']}")
    st.write(f"**Rating:** ⭐ {fila['rating']:.2f}")
    st.write(f"**Número de reseñas:** {int(fila['rating_count'])}")

    if "sentimiento" in fila.index:
        st.write(f"**Sentimiento promedio de reseñas:** {fila['sentimiento']:.3f}")

# --------------------------------------------------
# 5. Indicadores globales del modelo
# --------------------------------------------------
st.write("---")
st.subheader("Desempeño global del modelo")

mae_global = df["Error_Absoluto"].mean()
mape_global = (df["Error_Absoluto"] / df["discounted_price"]).mean()

c1, c2 = st.columns(2)
c1.metric("MAE global (Error absoluto medio)", f"₹ {mae_global:,.2f}")
c2.metric("MAPE global (Error porcentual medio)", f"{mape_global*100:,.2f}%")

# --------------------------------------------------
# 6. Top 10 productos con mayor error
# --------------------------------------------------
st.write("---")
st.subheader("Top 10 productos con mayor error absoluto")

top_error = df.sort_values("Error_Absoluto", ascending=False).head(10)

st.dataframe(
    top_error[[
        "product_name",
        "category",
        "discounted_price",
        "predicted_discounted_price",
        "Error_Absoluto",
        "rating",
        "rating_count"
    ]],
    use_container_width=True,
)
