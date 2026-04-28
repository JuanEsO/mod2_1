"""
Reporte de Exploración de Datos — Data Science Job Posts 2025
Prepara el terreno para un pipeline de aprendizaje automático.
"""

import ast
import re
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Configuración ──────────────────────────────────────────────────────────────
FILE     = "data_science_job_posts_2025.csv"
OUT_DIR  = Path("eda_report")
OUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 130


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_salary(s: str):
    """Devuelve (min, max, mid) en float o (None, None, None)."""
    if pd.isna(s) or s.strip() in ("", "Private", "Public"):
        return None, None, None
    nums = re.findall(r"[\d,]+", s.replace("€", ""))
    nums = [float(n.replace(",", "")) for n in nums if n]
    if len(nums) == 0:
        return None, None, None
    if len(nums) == 1:
        return nums[0], nums[0], nums[0]
    return nums[0], nums[1], (nums[0] + nums[1]) / 2


def parse_skills(s: str):
    """Convierte el string-lista de Python a lista real."""
    try:
        result = ast.literal_eval(s)
        return result if isinstance(result, list) else []
    except Exception:
        return []


def parse_company_size(s: str):
    """Extrae número de empleados si el valor es numérico."""
    try:
        return float(str(s).replace(",", ""))
    except ValueError:
        return None


def parse_post_days(s: str):
    """Aproxima días desde la publicación."""
    if pd.isna(s):
        return None
    s = s.lower().strip()
    if "hour" in s or "just" in s:
        return 0
    m = re.search(r"(\d+)\s+day", s)
    if m:
        return int(m.group(1))
    if "week" in s:
        m2 = re.search(r"(\d+)\s+week", s)
        return int(m2.group(1)) * 7 if m2 else 7
    if "month" in s:
        m3 = re.search(r"(\d+)\s+month", s)
        return int(m3.group(1)) * 30 if m3 else 30
    return None


def save(fig, name):
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [guardado] {path}")


def section(title):
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


# ── Carga y limpieza ───────────────────────────────────────────────────────────
section("1. CARGA Y ESTRUCTURA GENERAL")

df = pd.read_csv(FILE)
print(f"\nShape original : {df.shape}")
print(f"Filas          : {df.shape[0]:,}")
print(f"Columnas       : {df.shape[1]}")

# Parseo de campos especiales
df[["salary_min", "salary_max", "salary_mid"]] = pd.DataFrame(
    df["salary"].apply(parse_salary).tolist(), index=df.index
)
df["skills_list"]      = df["skills"].apply(parse_skills)
df["n_skills"]         = df["skills_list"].apply(len)
df["company_size_num"] = df["company_size"].apply(parse_company_size)
df["post_days"]        = df["post_date"].apply(parse_post_days)

# Extraer estado/país desde headquarter
df["hq_state"]   = df["headquarter"].str.extract(r",\s*([A-Z]{2}),")
df["hq_country"] = df["headquarter"].str.extract(r",\s*([A-Z]{2})\s*$")

print("\nColumnas tras ingeniería:\n", df.columns.tolist())
print("\nTipos de datos:\n", df.dtypes.to_string())


# ── Valores nulos ──────────────────────────────────────────────────────────────
section("2. VALORES NULOS Y COBERTURA")

hashable_cols = [c for c in df.columns if c != "skills_list"]
null_df = pd.DataFrame({
    "nulos"      : df.isnull().sum(),
    "% nulos"    : (df.isnull().mean() * 100).round(2),
    "únicos"     : df[hashable_cols].nunique().reindex(df.columns),
    "cobertura%" : ((1 - df.isnull().mean()) * 100).round(2),
})
print(null_df.sort_values("% nulos", ascending=False).to_string())

# Plot missingness
fig, ax = plt.subplots(figsize=(10, 5))
null_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
null_pct = null_pct[null_pct > 0]
bars = ax.barh(null_pct.index, null_pct.values, color=sns.color_palette("Reds_r", len(null_pct)))
ax.set_xlabel("% valores nulos")
ax.set_title("Porcentaje de valores nulos por columna")
for bar, val in zip(bars, null_pct.values):
    ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=9)
save(fig, "01_nulos")


# ── Variables categóricas ──────────────────────────────────────────────────────
section("3. DISTRIBUCIÓN DE VARIABLES CATEGÓRICAS")

cat_cols = ["job_title", "seniority_level", "status", "industry", "ownership"]

for col in cat_cols:
    vc = df[col].value_counts(dropna=False).head(15)
    print(f"\n── {col} ({'top 15' if df[col].nunique() > 15 else 'todos'})")
    print(vc.to_string())

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    vc = df[col].value_counts(dropna=False).head(12)
    labels = vc.index[::-1].fillna("(nulo)").astype(str)
    axes[i].barh(labels, vc.values[::-1],
                 color=sns.color_palette("Blues_r", 12))
    axes[i].set_title(col)
    axes[i].set_xlabel("frecuencia")
axes[-1].set_visible(False)
fig.suptitle("Distribución de variables categóricas", fontsize=14, y=1.01)
plt.tight_layout()
save(fig, "02_categoricas")


# ── Salario ────────────────────────────────────────────────────────────────────
section("4. ANÁLISIS DE SALARIO")

sal = df["salary_mid"].dropna()
print(f"\nSalarios con valor numérico : {len(sal):,} / {len(df):,} ({len(sal)/len(df)*100:.1f}%)")
print(f"Mínimo                      : €{sal.min():,.0f}")
print(f"Máximo                      : €{sal.max():,.0f}")
print(f"Mediana                     : €{sal.median():,.0f}")
print(f"Media                       : €{sal.mean():,.0f}")
print(f"Desv. estándar              : €{sal.std():,.0f}")
print(f"\nPercentiles:")
for p in [10, 25, 50, 75, 90, 95]:
    print(f"  P{p:2d}: €{sal.quantile(p/100):,.0f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(sal / 1000, bins=40, color="#4C72B0", edgecolor="white", linewidth=0.5)
axes[0].set_xlabel("Salario (€K)")
axes[0].set_title("Distribución del salario medio")
axes[0].axvline(sal.median() / 1000, color="red", linestyle="--", label=f"Mediana €{sal.median()/1000:.0f}K")
axes[0].legend()

axes[1].boxplot(sal / 1000, vert=False, patch_artist=True,
                boxprops=dict(facecolor="#4C72B0", alpha=0.6))
axes[1].set_xlabel("Salario (€K)")
axes[1].set_title("Boxplot del salario medio")
plt.tight_layout()
save(fig, "03_salario_dist")

# Salario por seniority
fig, ax = plt.subplots(figsize=(10, 5))
order = (df.dropna(subset=["salary_mid", "seniority_level"])
           .groupby("seniority_level")["salary_mid"]
           .median().sort_values().index)
sns.boxplot(data=df.dropna(subset=["salary_mid", "seniority_level"]),
            x="salary_mid", y="seniority_level", order=order,
            palette="Blues", ax=ax)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x/1000:.0f}K"))
ax.set_title("Salario según nivel de seniority")
ax.set_xlabel("Salario medio (€)")
ax.set_ylabel("")
save(fig, "04_salario_seniority")

# Salario por industria (top 10)
top_industries = df["industry"].value_counts().head(10).index
fig, ax = plt.subplots(figsize=(12, 6))
order2 = (df[df["industry"].isin(top_industries)]
            .dropna(subset=["salary_mid"])
            .groupby("industry")["salary_mid"]
            .median().sort_values().index)
sns.boxplot(data=df[df["industry"].isin(top_industries)].dropna(subset=["salary_mid"]),
            x="salary_mid", y="industry", order=order2,
            palette="muted", ax=ax)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x/1000:.0f}K"))
ax.set_title("Salario por industria (top 10)")
ax.set_xlabel("Salario medio (€)")
ax.set_ylabel("")
save(fig, "05_salario_industria")

# Salario por modalidad
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(data=df.dropna(subset=["salary_mid", "status"]),
            x="status", y="salary_mid", palette="Set2", ax=ax)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x/1000:.0f}K"))
ax.set_title("Salario por modalidad de trabajo")
ax.set_xlabel("")
ax.set_ylabel("Salario medio (€)")
save(fig, "06_salario_modalidad")


# ── Habilidades ────────────────────────────────────────────────────────────────
section("5. ANÁLISIS DE HABILIDADES (SKILLS)")

all_skills = [sk for lst in df["skills_list"] for sk in lst]
skill_counts = Counter(all_skills)

print(f"\nHabilidades únicas mencionadas: {len(skill_counts):,}")
print(f"Total menciones               : {len(all_skills):,}")
print(f"\nTop 30 habilidades:")
for skill, count in skill_counts.most_common(30):
    pct = count / len(df) * 100
    print(f"  {skill:<25} {count:4d}  ({pct:.1f}% de ofertas)")

print(f"\nEstadísticas de n_skills por oferta:")
print(df["n_skills"].describe().to_string())

top_skills = pd.Series(skill_counts).nlargest(25)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
top_skills.sort_values().plot.barh(ax=axes[0], color="#4C72B0")
axes[0].set_title("Top 25 habilidades más demandadas")
axes[0].set_xlabel("número de ofertas")

df["n_skills"].hist(bins=range(0, df["n_skills"].max() + 2), ax=axes[1],
                    color="#DD8452", edgecolor="white")
axes[1].set_title("Distribución de nº de skills por oferta")
axes[1].set_xlabel("número de skills")
axes[1].set_ylabel("frecuencia")
plt.tight_layout()
save(fig, "07_skills")

# Skills por seniority
fig, ax = plt.subplots(figsize=(9, 5))
sns.boxplot(data=df.dropna(subset=["seniority_level"]),
            x="seniority_level", y="n_skills",
            order=["entry", "associate", "mid-senior", "senior", "director", "executive", "lead"],
            palette="Blues", ax=ax)
ax.set_title("Nº de skills requeridas por nivel de seniority")
ax.set_xlabel("")
ax.set_ylabel("skills por oferta")
plt.xticks(rotation=20)
save(fig, "08_skills_seniority")


# ── Variables numéricas adicionales ───────────────────────────────────────────
section("6. TAMAÑO DE EMPRESA Y ANTIGÜEDAD DEL ANUNCIO")

print("\nTamaño de empresa (numérico):")
print(df["company_size_num"].describe().to_string())
print(f"\nValores no numéricos en company_size: "
      f"{df['company_size'][df['company_size_num'].isna()].value_counts().head(10).to_string()}")

print(f"\nDías desde publicación:")
print(df["post_days"].describe().to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
cs = df["company_size_num"].dropna()
axes[0].hist(cs.clip(upper=cs.quantile(0.95)), bins=40, color="#55A868", edgecolor="white")
axes[0].set_title("Tamaño de empresa (empleados, hasta P95)")
axes[0].set_xlabel("empleados")

pd_days = df["post_days"].dropna()
axes[1].hist(pd_days, bins=30, color="#C44E52", edgecolor="white")
axes[1].set_title("Días desde publicación del anuncio")
axes[1].set_xlabel("días")
plt.tight_layout()
save(fig, "09_empresa_tiempo")


# ── Geografía ─────────────────────────────────────────────────────────────────
section("7. DISTRIBUCIÓN GEOGRÁFICA (HEADQUARTER)")

print("\nTop 15 estados (US):")
print(df["hq_state"].value_counts().head(15).to_string())

top_states = df["hq_state"].value_counts().head(15)
fig, ax = plt.subplots(figsize=(10, 5))
top_states.sort_values().plot.barh(ax=ax, color="#8172B2")
ax.set_title("Top 15 estados por HQ de empresa")
ax.set_xlabel("número de ofertas")
save(fig, "10_geografia")


# ── Correlaciones numéricas ────────────────────────────────────────────────────
section("8. CORRELACIONES ENTRE VARIABLES NUMÉRICAS")

num_df = df[["salary_mid", "salary_min", "salary_max", "n_skills",
             "company_size_num", "post_days"]].dropna()
print(f"\nFilas con todas las variables numéricas completas: {len(num_df):,}")
print("\nMatriz de correlación (Pearson):")
print(num_df.corr().round(3).to_string())

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, linewidths=0.5, ax=ax)
ax.set_title("Correlación entre variables numéricas")
save(fig, "11_correlacion")


# ── Resumen para ML ────────────────────────────────────────────────────────────
section("9. RESUMEN ORIENTADO A ML")

salary_coverage = df["salary_mid"].notna().mean() * 100
print(f"""
VARIABLE OBJETIVO POTENCIAL
  salary_mid  →  cobertura {salary_coverage:.1f}%  |  rango €{df['salary_mid'].min():,.0f} – €{df['salary_mid'].max():,.0f}

FEATURES CANDIDATAS
  Categóricas  : job_title, seniority_level, status, industry, ownership, hq_state
  Numéricas    : n_skills, company_size_num, post_days
  Texto/multi  : skills (multi-label o bag-of-skills)

PROBLEMAS A RESOLVER ANTES DE MODELAR
  - salary_mid   : {df['salary_mid'].isna().sum()} nulos ({df['salary_mid'].isna().mean()*100:.1f}%) — solo filas con salario válido
  - seniority    : {df['seniority_level'].isna().sum()} nulos — imputar "unknown" o excluir
  - status       : {df['status'].isna().sum()} nulos ({df['status'].isna().mean()*100:.1f}%) — relevante para WFH premium
  - company_size : valores mixtos (números + "Private") — tratar como numérica + flag
  - revenue      : mayormente categórico ("Private"/"Public") — one-hot
  - skills       : requiere one-hot / TF-IDF / embedding por skill
  - salary rango : aprox. {(df['salary_min'] != df['salary_max']).sum()} ofertas con rango — usar mid o modelar min/max por separado

CARDINALIDAD DE CATEGÓRICAS
""")
for c in ["job_title", "seniority_level", "status", "industry", "ownership", "hq_state"]:
    print(f"  {c:<18}: {df[c].nunique()} valores únicos")

print(f"""
RECOMENDACIÓN DE PIPELINE
  1. Filtrar filas sin salary_mid  →  ~{df['salary_mid'].notna().sum()} muestras útiles
  2. Imputar / encodear categóricas (OrdinalEncoder para seniority, OHE para resto)
  3. Bag-of-skills → columnas binarias (top ~50 skills cubren la mayoría)
  4. Escalar numéricas (StandardScaler / RobustScaler por outliers en salary)
  5. Baseline: Ridge / GradientBoosting  →  objetivo: RMSE y MAE en €
""")

print(f"\nTodos los gráficos guardados en: {OUT_DIR.resolve()}/")
print("─" * 70)
