import os
import subprocess
import markdown
import re
import latex2mathml.converter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD_PATH = os.path.join(ROOT, "paper_academic.md")
HTML_PATH = os.path.join(ROOT, "scratch", "paper.html")
PDF_PATH = os.path.join(ROOT, "options-pricing-ml-Chaithanya-2026.pdf")

def convert_math(text):
    # Find all $$ ... $$ blocks and convert them to MathML
    def math_replacer(match):
        latex = match.group(1)
        try:
            mathml = latex2mathml.converter.convert(latex)
            return f'<div class="math-block" style="text-align: center; margin: 1em 0;">{mathml}</div>'
        except Exception as e:
            print(f"Failed to convert math: {latex}")
            return match.group(0)
    
    text = re.sub(r'\$\$(.*?)\$\$', math_replacer, text, flags=re.DOTALL)
    
    def inline_math_replacer(match):
        latex = match.group(1)
        try:
            mathml = latex2mathml.converter.convert(latex)
            return f'<span class="math-inline">{mathml}</span>'
        except Exception as e:
            return match.group(0)
            
    text = re.sub(r'(?<!\\)(?<!\$)\$(?!\$)(.*?)(?<!\\)(?<!\$)\$(?!\$)', inline_math_replacer, text)
    return text

def main():
    # Read humanized markdown
    with open(MD_PATH, "r") as f:
        md_text = f.read()

    # Convert math
    md_text = convert_math(md_text)

    # Convert to HTML
    html_body = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])

    # Wrap in our Journal of Finance / Handwriting HTML template
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Options Pricing Paper</title>
        
        <style>
            @page {{
                size: letter;
                margin: 1in;
            }}
            body {{
                font-family: 'Times New Roman', Times, serif;
                font-size: 10.5pt;
                line-height: 1.5;
                color: #111;
                margin: 0;
                padding: 0;
            }}
            h1, h2, h3 {{
                font-weight: bold;
                text-align: left;
                margin-top: 1.5em;
                margin-bottom: 0.5em;
                break-after: avoid;
            }}
            h1 {{
                font-size: 16pt;
                text-align: center;
                margin-top: 0;
            }}
            h2 {{
                font-size: 13pt;
            }}
            h3 {{
                font-size: 11pt;
            }}
            
            /* The title and abstract should be centered */
            .header-section {{
                text-align: center;
                margin-bottom: 3em;
                margin-top: 2em;
            }}
            .header-section h3 {{
                text-align: center;
                margin: 0.2em;
                font-weight: normal;
                font-size: 12pt;
            }}
            
            /* The main text is a highly professional single column */
            .main-content {{
                max-width: 6.5in;
                margin: 0 auto;
                text-align: justify;
            }}
            
            p {{
                margin-bottom: 1.2em;
                text-indent: 1.5em;
                overflow-wrap: break-word;
                word-wrap: break-word;
            }}
            p:first-of-type {{
                text-indent: 0;
            }}
            
            img {{
                max-width: 100%;
                height: auto;
                display: block;
                margin: 1em auto;
                border: 1px solid #ccc;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            }}
            
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 1.5em 0;
                font-size: 9pt;
            }}
            table, th, td {{
                border: 1px solid #000;
            }}
            th, td {{
                padding: 6px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            
            /* Prevent awkward breaks */
            table, img, h2, h3, .math-block {{
                break-inside: avoid;
            }}
            
        </style>
    </head>
    <body>
        <div class="header-section">
            <h1>A Comparative Study of Numerical Methods for European Option Pricing: Black-Scholes, CRR Binomial Trees, Monte Carlo Simulation with Variance Reduction, and the Heston Stochastic Volatility Model</h1>
            <h3>Chaithanya</h3>
            <h3>Independent Researcher</h3>
            <hr style="margin: 2em 0; width: 50%; margin-left: auto; margin-right: auto;">
            <h2>Abstract</h2>
            <p style="text-align: justify; max-width: 80%; margin: 0 auto; text-indent: 0;">
                so this paper is basically a head-to-head comparison of old school numerical methods, stochastic volatility, and some newer deep learning stuff for pricing options. i coded everything from scratch in python and benchmarked it all against live market data. we look at the black-scholes formula, the CRR binomial tree, monte carlo, and the heston (1993) model. then i compared those against neural networks like MLPs trained on log-moneyness, LSTM volatility forecasters, and VAEs. i also added a per-option nelder-mead calibration for the heston model, which gave the best accuracy overall (MAE $0.14) and actually reproduced the real volatility smile. i ran a diebold-mariano test which proved that black-scholes, CRR, and monte carlo are statistically tied if they get the same market volatility. finally, i showed why monte carlo is still the goat for path-dependent exotics like barrier options. all the code is on my github.
            </p>
            <hr style="margin: 2em 0;">
        </div>
        
        <div class="main-content">
            {{html_body_main}}
        </div>
    </body>
    </html>
    """

    # Split the body to separate the Title/Abstract from the main content
    parts = html_body.split("<h2>1. intro</h2>")
    if len(parts) > 1:
        body_content = "<h2>1. intro</h2>" + parts[1]
    else:
        body_content = html_body

    final_html = html_content.replace("{html_body_main}", body_content)

    os.makedirs(os.path.dirname(HTML_PATH), exist_ok=True)
    with open(HTML_PATH, "w") as f:
        f.write(final_html)

    print(f"Running WeasyPrint to generate PDF...")
    from weasyprint import HTML, CSS
    
    # We load images using an absolute base_url so WeasyPrint can resolve them
    HTML(string=final_html, base_url="file://" + ROOT + "/").write_pdf(PDF_PATH)
    print(f"Generated PDF: {PDF_PATH}")

if __name__ == "__main__":
    main()
