"""Builds the 4-page project guide PDF (docs/GUIA_DEL_PROYECTO.pdf).

Regenerate after restructuring folders:

    python scripts/generate_project_guide.py
"""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "GUIA_DEL_PROYECTO.pdf"

PURPLE = (124, 58, 237)  # #7C3AED
DARK_PURPLE = (49, 46, 129)  # #312E81
TEXT = (23, 32, 51)  # #172033
MUTED = (100, 116, 139)  # #64748B


class ProjectGuidePDF(FPDF):
    def header(self) -> None:
        self.set_fill_color(*DARK_PURPLE)
        self.rect(0, 0, self.w, 22, style="F")
        self.set_xy(10, 6)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, "NeuroTrain Lab - Guia del proyecto")
        self.set_y(26)
        self.set_text_color(*TEXT)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 8, f"Pagina {self.page_no()} / 4", align="C")

    def section_title(self, text: str) -> None:
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*PURPLE)
        self.cell(0, 9, text, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*TEXT)
        self.ln(1)

    def body(self, text: str) -> None:
        self.set_font("Helvetica", "", 10.5)
        self.multi_cell(0, 5.6, text)
        self.ln(1)

    def bullet(self, text: str) -> None:
        self.set_font("Helvetica", "", 10.5)
        self.set_x(self.l_margin + 4)
        self.multi_cell(0, 5.6, f"- {text}")

    def mono_block(self, text: str) -> None:
        self.set_font("Courier", "", 8.6)
        self.set_fill_color(245, 243, 255)  # #F5F3FF
        self.multi_cell(0, 4.4, text, fill=True)
        self.set_font("Helvetica", "", 10.5)
        self.ln(1)


def build() -> ProjectGuidePDF:
    pdf = ProjectGuidePDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(14, 14, 14)

    # Page 1 -- purpose, audience, disclaimer, pillars
    pdf.add_page()
    pdf.section_title("Que es NeuroTrain Lab")
    pdf.body(
        "Un curso introductorio de redes neuronales artificiales pensado para "
        "aprender haciendo. Cubre 4 temas -- el perceptron y las funciones de "
        "activacion, la funcion de perdida y backpropagation, los optimizadores, "
        "y el entrenamiento con control de sobreajuste -- cada uno con un "
        "notebook de ejercicios reales y una pagina interactiva en la app."
    )
    pdf.section_title("A quien va dirigido")
    pdf.body(
        "A estudiantes que empiezan en deep learning y quieren construir "
        "intuicion antes de memorizar formulas. Los notebooks estan disponibles "
        "en espanol y en ingles; el codigo fuente del proyecto esta en ingles."
    )
    pdf.section_title("Aviso de uso responsable")
    pdf.body(
        "El proyecto usa el dataset real Breast Cancer Wisconsin (Diagnostic) "
        "como ejemplo de principio a fin. Es una demostracion educativa: no es "
        "un dispositivo medico, no esta validado externamente y no debe usarse "
        "para diagnostico, triaje ni ninguna decision sobre personas. Mas "
        "detalle en docs/DATASET_Y_USO_RESPONSABLE.md."
    )
    pdf.section_title("Los 4 pilares del proyecto")
    pdf.bullet("notebooks/ -- 8 notebooks (4 temas x espanol/ingles) con ejercicios reales")
    pdf.bullet("app.py + app_pages/ -- la aplicacion Streamlit: recorrido guiado + laboratorio")
    pdf.bullet("src/neurotrain/ -- la logica reutilizable (datos, modelo, metricas, graficos)")
    pdf.bullet("docs/ -- esta guia, la guia de estudio y videos, la presentacion Masterclass y el aviso de uso responsable")
    pdf.section_title("Licencia")
    pdf.body(
        "Material con licencia Creative Commons Atribucion 4.0 (CC BY 4.0). Si usas o "
        "adaptas este contenido, por favor cita a Elizabeth Sena. Detalle completo en LICENSE."
    )

    # Page 2 -- folder map
    pdf.add_page()
    pdf.section_title("Mapa de carpetas")
    pdf.mono_block(
        "neurotrain-lab/\n"
        "|-- app.py                      Punto de entrada de la app Streamlit\n"
        "|-- app_pages/                  Paginas: inicio, 4 temas, laboratorio\n"
        "|-- notebooks/\n"
        "|   |-- es/  01..04             Los 4 notebooks, en espanol\n"
        "|   |-- en/  01..04             Los 4 notebooks, en ingles\n"
        "|   `-- material_adicional/     Notebook bonus (DOTCSV / Elizabeth Sena)\n"
        "|-- scripts/\n"
        "|   |-- notebook_builders/      Generador de los 8 notebooks\n"
        "|   |-- build_notebooks.py      Regenera los notebooks desde codigo\n"
        "|   |-- verify_notebooks.py     Ejecuta cada notebook de principio a fin\n"
        "|   |-- export_dataset.py       Exporta el dataset a CSV\n"
        "|   `-- generate_project_guide.py   Genera este PDF\n"
        "|-- src/neurotrain/              Paquete reutilizable\n"
        "|   |-- config.py               Hiperparametros tipados\n"
        "|   |-- data.py                 Carga, split, escalado\n"
        "|   |-- modeling.py             Baseline + red neuronal\n"
        "|   |-- evaluation.py           Metricas de clasificacion\n"
        "|   |-- visualization.py        Graficos compartidos (notebooks + app)\n"
        "|   `-- celebrations.py         Confeti de cierre de cada notebook\n"
        "|-- data/                       Dataset real (CSV)\n"
        "|-- docs/                       Esta guia y las otras guias del proyecto\n"
        "|-- tests/                      Suite de pruebas automatizadas\n"
        "|-- Dockerfile, docker-compose.yml, .dockerignore\n"
        "`-- requirements.txt, requirements-app.txt"
    )

    # Page 3 -- recommended path
    pdf.add_page()
    pdf.section_title("Orden logico recomendado")
    pdf.bullet("1. Lee esta guia completa (4 paginas) para orientarte.")
    pdf.bullet("2. Prepara el entorno (ver pagina 4).")
    pdf.bullet(
        "3. Recorre los notebooks 1 a 4, en espanol o en ingles: cada uno "
        "termina con una autoevaluacion y una celebracion."
    )
    pdf.bullet(
        "4. Lanza la app (streamlit run app.py) y sigue el Recorrido guiado: "
        "teoria con analogias, un video por tema y una mini-demo interactiva."
    )
    pdf.bullet(
        "5. Entra en Laboratorio - Modo Experimento: entrena una red real sobre "
        "el dataset clinico, tocando arquitectura, epochs, batch size, Dropout "
        "y EarlyStopping."
    )
    pdf.bullet("6. Opcional: ejecuta la suite de tests para verificar que todo funciona.")
    pdf.section_title("Por que este orden")
    pdf.body(
        "Los notebooks contienen los ejercicios de codigo reales (marcados con "
        "el icono de lapiz) -- ahi es donde se construye la intuicion. La app "
        "reutiliza exactamente la misma logica de src/neurotrain/, asi que el "
        "laboratorio se siente como una continuacion natural, no como una "
        "herramienta aparte."
    )

    # Page 4 -- how to run it
    pdf.add_page()
    pdf.section_title("Ejecutar en local (con un entorno virtual)")
    pdf.mono_block(
        "python -m venv .venv\n"
        "source .venv/bin/activate        # En Windows: .venv\\Scripts\\Activate.ps1\n"
        "pip install --upgrade pip\n"
        "pip install -r requirements.txt\n\n"
        "streamlit run app.py"
    )
    pdf.section_title("Ejecutar con Docker")
    pdf.mono_block("docker compose up --build")
    pdf.body(
        "La imagen usa requirements-app.txt (sin Jupyter ni PyTorch, que solo "
        "hacen falta para los notebooks) y tensorflow-cpu para mantenerla ligera. "
        "La app queda disponible en http://localhost:8501"
    )
    pdf.section_title("Verificar que todo funciona")
    pdf.mono_block(
        "PYTHONPATH=src python -m unittest discover -s tests -v\n"
        "python scripts/verify_notebooks.py"
    )
    pdf.section_title("Mas informacion")
    pdf.body(
        "README.md tiene el detalle completo. docs/GUIA_ESTUDIO_Y_VIDEOS.md "
        "tiene el recorrido de videos recomendado. "
        "docs/DATASET_Y_USO_RESPONSABLE.md documenta el origen y los limites "
        "del dataset."
    )

    return pdf


def main() -> None:
    pdf = build()
    OUTPUT.parent.mkdir(exist_ok=True)
    pdf.output(str(OUTPUT))
    print(f"Generated: {OUTPUT} ({len(pdf.pages)} pages)")


if __name__ == "__main__":
    main()
