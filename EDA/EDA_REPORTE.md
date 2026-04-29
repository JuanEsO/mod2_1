# Reporte de Exploración de Datos
## Data Science Job Posts 2025

---

## Índice

1. [Estructura general del dataset](#1-estructura-general)
2. [Calidad de datos y valores nulos](#2-calidad-de-datos-y-valores-nulos)
3. [Variables categóricas](#3-variables-categóricas)
4. [Análisis de salario](#4-análisis-de-salario)
5. [Habilidades requeridas (Skills)](#5-habilidades-requeridas-skills)
6. [Tamaño de empresa y antigüedad del anuncio](#6-tamaño-de-empresa-y-antigüedad-del-anuncio)
7. [Distribución geográfica](#7-distribución-geográfica)
8. [Correlaciones entre variables numéricas](#8-correlaciones-entre-variables-numéricas)
9. [Resumen orientado a ML](#9-resumen-orientado-a-ml)

---

## 1. Estructura General

### Dimensiones

| Dimensión | Valor |
|---|---|
| Filas (ofertas) | 944 |
| Columnas originales | 13 |
| Columnas tras ingeniería de features | 22 |

### Columnas originales

| Columna | Tipo | Descripción |
|---|---|---|
| `job_title` | str | Título del puesto |
| `seniority_level` | str | Nivel de seniority |
| `status` | str | Modalidad: on-site / hybrid / remote |
| `company` | str | Identificador anonimizado de empresa |
| `location` | str | Ubicación del puesto (puede ser múltiple) |
| `post_date` | str | Fecha relativa de publicación ("X days ago") |
| `headquarter` | str | Ciudad, estado y país del HQ |
| `industry` | str | Sector industrial |
| `ownership` | str | Tipo de propiedad (Public / Private) |
| `company_size` | str | Tamaño de empresa (nº empleados, mixto) |
| `revenue` | str | Ingresos de la empresa (mixto) |
| `salary` | str | Salario en texto con símbolo € (rango o valor único) |
| `skills` | str | Lista de habilidades como string Python |

### Columnas derivadas (ingeniería de features)

| Columna | Origen | Descripción |
|---|---|---|
| `salary_min` | `salary` | Valor mínimo del rango salarial en € |
| `salary_max` | `salary` | Valor máximo del rango salarial en € |
| `salary_mid` | `salary` | Punto medio del rango salarial en € |
| `skills_list` | `skills` | Lista Python real de habilidades |
| `n_skills` | `skills_list` | Número de habilidades por oferta |
| `company_size_num` | `company_size` | Versión numérica de empleados |
| `post_days` | `post_date` | Días aproximados desde publicación |
| `hq_state` | `headquarter` | Estado del HQ (ej: CA, NY) |
| `hq_country` | `headquarter` | País del HQ |

---

## 2. Calidad de Datos y Valores Nulos

### Tabla de cobertura

| Columna | Nulos | % Nulos | Cobertura | Valores únicos |
|---|---|---|---|---|
| `status` | 256 | **27.1%** | 72.9% | 3 |
| `hq_state` | 108 | **11.4%** | 88.6% | 54 |
| `seniority_level` | 60 | **6.4%** | 93.6% | 4 |
| `ownership` | 47 | **5.0%** | 95.0% | 2 |
| `company_size_num` | 39 | 4.1% | 95.9% | 507 |
| `post_days` | 24 | 2.5% | 97.5% | 34 |
| `revenue` | 15 | 1.6% | 98.4% | 312 |
| `hq_country` | 10 | 1.1% | 98.9% | 19 |
| `job_title` | 3 | 0.3% | 99.7% | 4 |
| `location` | 2 | 0.2% | 99.8% | 431 |
| `salary_mid` | 0 | **0.0%** | **100%** | 874 |
| `industry` | 0 | 0.0% | 100% | 8 |
| `skills` | 0 | 0.0% | 100% | 400 |
| `n_skills` | 0 | 0.0% | 100% | 18 |

### Hallazgos de calidad

- **`status` es la columna más problemática**: 1 de cada 4 ofertas no indica la modalidad de trabajo. Esto es relevante porque la modalidad remote/hybrid tiene impacto directo en el salario.
- **`seniority_level` con 6.4% de nulos**: representa 60 ofertas sin clasificar. La distribución sesgada hacia "senior" (ver sección 3) sugiere que los nulos podrían ser niveles no estándar.
- **`salary_mid` tiene cobertura perfecta (100%)**: aunque el salario original era texto, el parseo fue exitoso para todos los registros. Aun así, el rango extremo superior (hasta €2.7M) indica outliers que requieren atención antes de modelar.
- **`company_size` tiene valores mixtos**: además de números de empleados, contiene categorías como "Private" y montos de revenue (ej: "€352.44B"), lo que indica un problema de origen en la recolección de datos.
- **`revenue` es mayormente categórico**: los valores "Private" (247 registros) y "Public" (227 registros) representan casi toda la columna, con solo ~15% de registros con valores monetarios reales.

---

## 3. Variables Categóricas

### job_title

| Título | Frecuencia | % |
|---|---|---|
| data scientist | 856 | 90.7% |
| machine learning engineer | 80 | 8.5% |
| data engineer | 4 | 0.4% |
| data analyst | 1 | 0.1% |
| (nulo) | 3 | 0.3% |

**Hallazgo**: El dataset está fuertemente dominado por ofertas de _Data Scientist_ (90.7%). La baja representación de otros roles limita la capacidad de un modelo para generalizar entre tipos de puesto. Para ML, conviene tratar `job_title` como variable de control o segmentar el análisis.

### seniority_level

| Nivel | Frecuencia | % |
|---|---|---|
| senior | 630 | 66.7% |
| lead | 116 | 12.3% |
| midlevel | 113 | 12.0% |
| (nulo) | 60 | 6.4% |
| junior | 25 | 2.6% |

**Hallazgo**: El nivel "senior" representa dos tercios de las ofertas. Los niveles junior y entry-level son extremadamente escasos (solo 25 registros), lo que indica que el mercado capturado en este dataset es principalmente senior. Esto genera **desbalance de clases** si seniority se usa como variable objetivo, y puede sesgar un modelo de predicción de salario hacia rangos altos.

### status (modalidad)

| Modalidad | Frecuencia | % del total | % de los conocidos |
|---|---|---|---|
| on-site | 363 | 38.5% | 52.8% |
| hybrid | 207 | 21.9% | 30.1% |
| remote | 118 | 12.5% | 17.2% |
| (nulo) | 256 | 27.1% | — |

**Hallazgo**: Entre las modalidades conocidas, el trabajo presencial domina (52.8%), seguido de híbrido (30.1%) y remoto (17.2%). El 27.1% de nulos es crítico — imputar esta columna requiere cuidado porque la modalidad correlaciona con salario.

### industry

| Industria | Frecuencia | % |
|---|---|---|
| Technology | 582 | 61.7% |
| Finance | 127 | 13.5% |
| Retail | 110 | 11.7% |
| Healthcare | 83 | 8.8% |
| Education | 19 | 2.0% |
| Energy | 12 | 1.3% |
| Manufacturing | 7 | 0.7% |
| Logistics | 4 | 0.4% |

**Hallazgo**: Tecnología concentra casi 2 de cada 3 ofertas. La distribución de industrias es marcadamente **desbalanceada en cola larga** — las últimas 4 categorías suman menos del 5% de los datos. Considerar agrupar en "Technology", "Finance", "Retail/Healthcare" y "Otros" para reducir ruido en el modelo.

### ownership

| Tipo | Frecuencia | % |
|---|---|---|
| Public | 579 | 61.3% |
| Private | 318 | 33.7% |
| (nulo) | 47 | 5.0% |

**Hallazgo**: Las empresas públicas dominan ligeramente. Dado que solo hay 2 valores válidos, esta variable puede codificarse directamente como binaria (0/1) con un flag adicional para nulos.

---

## 4. Análisis de Salario

### Estadísticas descriptivas

| Estadístico | Valor (€) |
|---|---|
| Mínimo | €7,055 |
| Máximo | €2,739,979 |
| Media | €131,780 |
| Mediana | €134,724 |
| Desv. estándar | €128,814 |
| P10 | €28,053 |
| P25 | €76,372 |
| P50 | €134,724 |
| P75 | €169,733 |
| P90 | €207,190 |
| P95 | €228,751 |

**Hallazgos clave**:

- **La distribución es bimodal**: existe una masa de salarios bajos (P10 = €28K) que probablemente corresponden a mercados no estadounidenses o posiciones muy junior, y un núcleo central de €75K–€200K que representa el mercado estadounidense de Data Science.
- **Los outliers superiores son extremos**: el máximo de €2.74M es al menos 10× la mediana. El rango P90–P95 (€207K–€228K) es razonable; todo lo que supere ~€350K debe tratarse como outlier o rol ejecutivo.
- **Media ≈ Mediana (€131K vs €134K)**: sugiere una distribución aproximadamente simétrica en el núcleo central, aunque los outliers superiores estiran la cola derecha.
- **594 ofertas tienen rango salarial** (min ≠ max): el punto medio es una aproximación razonable para el objetivo de regresión, pero modelar min y max por separado permitiría capturar la incertidumbre de la negociación.

### Salario por seniority

Los niveles de seniority muestran progresión esperada en salario:

- **junior** → mediana más baja (~€60K–€80K)
- **midlevel** → rango medio (~€100K–€140K)
- **senior** → mediana ~€140K–€160K
- **lead** → mediana similar a senior con mayor dispersión
- Hay solapamiento significativo entre niveles, lo que indica que el seniority solo explica parcialmente la varianza salarial.

### Salario por industria

- **Technology y Finance** ofrecen los salarios más altos (mediana >€140K).
- **Retail y Healthcare** se ubican en el rango medio (€100K–€130K).
- **Education y Logistics** muestran los salarios más bajos, con mayor compresión del rango.

### Salario por modalidad

- **Remote** tiende a salarios ligeramente superiores o equivalentes a hybrid.
- **On-site** muestra mayor dispersión, incluyendo tanto los salarios más bajos como algunos de los más altos.
- La diferencia no es dramática, pero es consistente con la literatura: los roles remotos suelen requerir seniority más alto.

---

## 5. Habilidades Requeridas (Skills)

### Estadísticas generales

| Métrica | Valor |
|---|---|
| Habilidades únicas totales | 33 |
| Total menciones en el dataset | 4,181 |
| Media de skills por oferta | 4.4 |
| Mediana de skills por oferta | 4 |
| Máximo de skills por oferta | 17 |
| Ofertas con 0 skills | significativo (~25%) |

### Top 30 habilidades

| Skill | Menciones | % de ofertas |
|---|---|---|
| python | 640 | **67.8%** |
| machine learning | 580 | **61.4%** |
| sql | 442 | 46.8% |
| r | 343 | 36.3% |
| aws | 218 | 23.1% |
| deep learning | 178 | 18.9% |
| tensorflow | 165 | 17.5% |
| spark | 161 | 17.1% |
| azure | 155 | 16.4% |
| pytorch | 148 | 15.7% |
| tableau | 116 | 12.3% |
| gcp | 106 | 11.2% |
| scikit-learn | 91 | 9.6% |
| scala | 85 | 9.0% |
| database | 83 | 8.8% |
| pandas | 76 | 8.1% |
| java | 73 | 7.7% |
| hadoop | 67 | 7.1% |
| git | 65 | 6.9% |
| numpy | 60 | 6.4% |
| docker | 54 | 5.7% |
| amazon | 51 | 5.4% |
| kubernetes | 44 | 4.7% |
| matplotlib | 36 | 3.8% |
| keras | 32 | 3.4% |
| powerbi | 25 | 2.6% |
| airflow | 25 | 2.6% |
| linux | 23 | 2.4% |
| neural network | 15 | 1.6% |
| scipy | 10 | 1.1% |

### Hallazgos

- **El vocabulario de skills es compacto**: solo 33 habilidades únicas en todo el dataset. Esto significa que un **bag-of-skills binario de 33 columnas** captura el 100% de la información sin necesidad de técnicas de texto complejas.
- **Python y ML son casi universales**: aparecen en más del 60% de las ofertas, lo que los convierte en features de baja discriminación individualmente, aunque su ausencia sí puede ser informativa.
- **SQL en casi la mitad de las ofertas**: refleja que el perfil de Data Scientist todavía requiere manejo de bases de datos relacionales.
- **Cloud dominada por AWS** (23%), seguida de Azure (16%) y GCP (11%): la triada cloud está bien representada.
- **~25% de ofertas tienen 0 skills listadas**: esto podría indicar scraping incompleto o que la empresa no listó requisitos técnicos explícitos. No necesariamente implica que no haya requisitos.
- **La distribución de n_skills es asimétrica a la derecha**: la mayoría de ofertas listan entre 1 y 7 skills, pero hay cola de hasta 17.

---

## 6. Tamaño de Empresa y Antigüedad del Anuncio

### Tamaño de empresa (empleados)

| Estadístico | Valor |
|---|---|
| Registros numéricos | 905 / 944 (95.9%) |
| Mínimo | 5 empleados |
| Máximo | 865,476 empleados |
| Media | 97,290 |
| Mediana | 20,030 |
| P25 | 1,530 |
| P75 | 94,570 |

**Hallazgos**:
- La distribución es **extremadamente sesgada a la derecha**: la mediana (20K) está muy por debajo de la media (97K), lo que indica que pocas empresas gigantes elevan el promedio.
- Los **39 registros no numéricos** en `company_size` contienen valores como "Private" o montos de revenue, reflejando un error de origen en la base de datos. Se recomienda crear un **flag binario `size_is_known`** como feature adicional.
- Para modelar, aplicar **transformación logarítmica** (`log1p`) a `company_size_num` para reducir el efecto de los outliers.

### Antigüedad del anuncio

| Estadístico | Valor |
|---|---|
| Registros parseados | 920 / 944 (97.5%) |
| Mínimo | 0 días (mismo día) |
| Máximo | 300 días |
| Media | 23.5 días |
| Mediana | 14 días |
| P25 | 7 días |
| P75 | 30 días |

**Hallazgos**:
- La mayoría de anuncios tienen menos de un mes de antigüedad, consistente con un scraping reciente.
- El outlier de 300 días es sospechoso — probablemente un error de parseo o un anuncio reabierto.
- Esta variable tiene **baja correlación con el salario** (r = −0.03), por lo que su valor predictivo es limitado.

---

## 7. Distribución Geográfica

### Top 15 estados (HQ de empresa)

| Estado | Ofertas | Notas |
|---|---|---|
| CA | 258 | Silicon Valley / SF Bay Area |
| NY | 80 | Wall Street / Tech NYC |
| VA | 79 | Northern Virginia (government/defense) |
| WA | 53 | Seattle (Amazon, Microsoft) |
| IL | 32 | Chicago |
| MN | 27 | Minneapolis |
| AR | 24 | Bentonville (Walmart HQ) |
| MA | 24 | Boston |
| MD | 24 | Maryland |
| NJ | 23 | New Jersey |
| TX | 23 | Austin / Dallas |
| KA | 19 | — |
| FL | 13 | Florida |
| WY | 13 | Wyoming |
| ON | 12 | Ontario, Canadá |

**Hallazgos**:
- **California domina ampliamente** con 258 ofertas (27% del total con estado conocido), seguida de NY y VA.
- La concentración en CA + NY + WA refleja los grandes hubs tecnológicos de EE. UU.
- **Virginia (VA)** tiene una presencia inusualmente alta que puede relacionarse con contratistas del gobierno federal y empresas de defensa/nube (AWS tiene HQ en Northern VA).
- **AR** aparece por la presencia de Walmart (HQ en Bentonville, AR), una empresa con fuerte inversión en Data Science.
- Con 54 estados/regiones únicos y 108 nulos (11.4%), `hq_state` es útil pero requiere manejo de la categoría "desconocido".

---

## 8. Correlaciones entre Variables Numéricas

### Matriz de correlación (Pearson)

| | salary_mid | salary_min | salary_max | n_skills | company_size | post_days |
|---|---|---|---|---|---|---|
| **salary_mid** | 1.000 | 0.990 | 0.991 | −0.077 | −0.080 | −0.030 |
| **salary_min** | 0.990 | 1.000 | 0.964 | −0.075 | −0.096 | −0.037 |
| **salary_max** | 0.991 | 0.964 | 1.000 | −0.078 | −0.064 | −0.022 |
| **n_skills** | −0.077 | −0.075 | −0.078 | 1.000 | 0.114 | −0.063 |
| **company_size** | −0.080 | −0.096 | −0.064 | 0.114 | 1.000 | −0.036 |
| **post_days** | −0.030 | −0.037 | −0.022 | −0.063 | −0.036 | 1.000 |

**Hallazgos**:

- **salary_min, salary_max y salary_mid correlacionan casi perfectamente entre sí** (r > 0.96): son derivadas de la misma fuente. Para modelar, usar solo `salary_mid` como objetivo o como feature.
- **Ninguna variable numérica tiene correlación lineal significativa con el salario** (máximo |r| = 0.08). Esto es una señal importante: **el poder predictivo del modelo deberá venir principalmente de las variables categóricas** (industria, seniority, modalidad) y del bag-of-skills.
- La correlación negativa entre `company_size` y salario (r = −0.08) es contraintuitiva pero débil — podría indicar que empresas más pequeñas que contratan Data Scientists ofrecen primas competitivas, o simplemente ruido dado la baja correlación.
- **`post_days` no tiene relación con el salario** (r = −0.03): la antigüedad del anuncio no discrimina el nivel salarial.

---

## 9. Resumen Orientado a ML

### Variable objetivo

| Variable | Cobertura | Rango | Tipo de problema |
|---|---|---|---|
| `salary_mid` | 100% (944 registros) | €7,055 – €2,739,979 | Regresión continua |

### Features candidatas

**Categóricas** (requieren encoding):

| Feature | Cardinalidad | Estrategia recomendada |
|---|---|---|
| `seniority_level` | 4 + nulos | OrdinalEncoder (junior < midlevel < senior < lead) |
| `industry` | 8 | OneHotEncoder o agrupar cola larga |
| `status` | 3 + nulos | OneHotEncoder + flag de nulo |
| `ownership` | 2 + nulos | Binaria (0/1) + flag de nulo |
| `job_title` | 4 + nulos | OneHotEncoder (baja cardinalidad) |
| `hq_state` | 54 + nulos | TargetEncoder o agrupar por región |

**Numéricas** (requieren escalar):

| Feature | Transformación recomendada |
|---|---|
| `company_size_num` | `log1p` + imputar mediana para nulos |
| `n_skills` | Sin transformación (rango acotado 0–17) |
| `post_days` | Baja utilidad predictiva; opcional |

**Multi-label**:

| Feature | Estrategia recomendada |
|---|---|
| `skills_list` | Bag-of-skills binario (33 columnas, cobertura total) |

### Problemas a resolver antes de modelar

1. **Outliers en salario**: el máximo de €2.74M distorsiona el entrenamiento. Opciones: (a) aplicar `log1p` al objetivo, (b) truncar al P99 (~€300K), (c) modelar en escala logarítmica y exponenciar en predicción.
2. **Nulos en `status` (27.1%)**: la imputación ciega es arriesgada. Opciones: crear categoría explícita "unknown", usar KNN-imputer con otras features, o entrenar con y sin esta variable para comparar.
3. **Nulos en `seniority_level` (6.4%)**: imputar con "unknown" como categoría adicional o usar el modelo para inferir el nivel a partir del salario y las skills.
4. **Desbalance de `job_title`**: 90.7% son Data Scientists. Si se entrena un modelo general, el sesgo hacia este rol puede afectar la predicción para ML Engineers o Data Engineers.
5. **`company_size` con valores categóricos mezclados**: separar en `company_size_num` (numérico) + `company_size_is_private` (flag binario).
6. **`revenue` casi sin información monetaria real**: usar como binaria (Private / Public / Other) en lugar de intentar parsear los montos.

### Pipeline recomendado

```
1. Preprocesamiento
   ├── Parsear salary_mid como objetivo (ya hecho)
   ├── Truncar outliers: salary_mid > €400K → excluir o winsorizar
   ├── Transformar objetivo: log1p(salary_mid)
   ├── Bag-of-skills: 33 columnas binarias
   ├── Imputar seniority_level nulos → "unknown"
   ├── Imputar status nulos → "unknown"
   ├── Imputar company_size_num nulos → mediana + flag
   └── log1p(company_size_num)

2. Encoding
   ├── OrdinalEncoder: seniority_level
   └── OneHotEncoder: industry, status, ownership, job_title

3. Escalado
   └── RobustScaler: salary_mid (robusto a outliers), company_size_num

4. Modelado (baseline → avanzado)
   ├── Ridge Regression (baseline lineal)
   ├── GradientBoostingRegressor / XGBoost (captura interacciones)
   └── Métrica: RMSE y MAE en escala original (€)

5. Validación
   └── StratifiedKFold o GroupKFold por industria para evitar data leakage
```

### Estimación del tamaño de muestra efectivo

| Escenario | Registros disponibles |
|---|---|
| Dataset completo | 944 |
| Sin outliers extremos (salary < €400K) | ~900 |
| Con todas las features sin nulos | ~830 |

Con ~830–900 muestras limpias, un modelo de regresión regularizado (Ridge, Lasso) o un ensemble ligero (GBM con max_depth bajo) es suficiente. Evitar redes neuronales profundas sin aumentación de datos.

---

*Reporte generado con `eda.py` — Gráficos disponibles en `eda_report/`*
