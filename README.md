# OptiEma

Proyecto OII464 de programación quirúrgica con bloqueo, recursos duales y comparación de algoritmos meméticos y metaheurísticos. Las instancias son sintéticas: no contienen datos de pacientes reales.

## Entregables

- [Informe final (PDF)](informe_final_QLMA.pdf) y [fuente LaTeX autónoma](informe_final_QLMA.tex).
- [Presentación de defensa](defensa_QLMA.pptx): 12 diapositivas, notas del expositor y cinco gráficos editables. Incluye un ejemplo didáctico de bloqueo identificado explícitamente.
- [Guion de defensa](guion_defensa.txt): duración objetivo de 14 minutos y 20 segundos.
- [Paquete completo de entrega](entrega_final_QLMA.zip).
- [Guía de entrega y reproducción](LEEME_entrega_final.txt).

## Organización

| Carpeta | Contenido |
| --- | --- |
| `proyecto/` | Código, 37 instancias, pruebas, configuraciones y registros experimentales. |
| `proyecto/experiments/records/production/` | Manifiesto congelado, calibraciones, 6.960 corridas finales y análisis. |
| `datos/` | Tablas, figuras y análisis complementarios usados en el informe. |
| `referencias/` | Entrega 2, pauta de Entrega 3 y material del curso suministrado. |
| `archivo/` | Respaldos históricos de preparación y ejecución. Sus estados no describen el cierre del estudio. |

## Reproducción

Se requiere Python, las dependencias de `proyecto/requirements.txt`, R 4.5.2 e irace 4.5.1. Los entornos y ejecutables locales no se distribuyen. Las versiones registradas se conservan en `proyecto/experiments/records/runtime_manifest.json`; sus rutas absolutas documentan el equipo original y deben adaptarse.

```sh
python -m venv .venv
# Activar el entorno según el sistema operativo.
python -m pip install -r proyecto/requirements.txt
cd proyecto
python experiments/pipeline.py --help
python experiments/summarize.py --help
```

La entrada del estudio es `experiments/pipeline.py`. El archivo `main.py` conserva la interfaz histórica del curso. Antes de repetir la calibración completa, consultar [experiments/LEEME.txt](proyecto/experiments/LEEME.txt), los argumentos de los programas y el manifiesto congelado. La calibración y las corridas originales se conservan para su auditoría.

Desde la raíz, la sensibilidad complementaria puede recalcularse con:

```sh
python analisis_sensibilidad_signos.py proyecto/experiments/records/production datos/supplemental_sign_tests.json
```

El informe se compila con `pdflatex` dos veces o con Tectonic. No necesita imágenes ni bibliografía externas.

## Alcance de los resultados

Se ejecutaron 6.960 corridas finales (29 instancias × 8 métodos × 30 semillas) y 3.998 llamadas reales de calibración. El método memético con probabilidades fijas calibradas obtuvo el menor RPD medio global (14,72 %); QL-MA obtuvo 22,59 %. No se afirma superioridad del aprendizaje adaptativo, optimalidad ni beneficio clínico demostrado. `max_wait` se analiza como diagnóstico y no constituye una restricción dura del modelo.

## Procedencia

El código del curso y las instancias proceden de [Saicooh/OII464_Hospitales](https://github.com/Saicooh/OII464_Hospitales), revisión `ecd30e3dbc1962d856c09b3470d1870c3b139a9a`. Este repositorio nuevo conserva una instantánea del trabajo y sus modificaciones, sin importar el historial Git del repositorio de origen. El [README original del proyecto](proyecto/README.md) y la documentación experimental explican la base y los cambios. No se atribuye una licencia nueva al material original ni al material docente.

Autores del informe: Sebastián Casas, Felipe Rojo, Benjamín Espinoza y Cristobal Martinez.
