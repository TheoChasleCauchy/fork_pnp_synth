from csvtex import create_latex_table
import os

def csv_to_standalone_tex(csv_file_path):
    base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    tex_file_path = os.path.join(os.path.dirname(csv_file_path) or '.', f"{base_name}.tex")

    table = create_latex_table(csv_file_path)

    standalone_doc = f"""\\documentclass[preview,border=5pt]{{standalone}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{booktabs}}
\\usepackage{{array}}

\\begin{{document}}

{table}

\\end{{document}}
"""

    with open(tex_file_path, 'w', encoding='utf-8') as f:
        f.write(standalone_doc)

    return tex_file_path


csv_file_path = "exp_audio_fpc/generated/results/metrics_table.csv"
csv_to_standalone_tex(csv_file_path)