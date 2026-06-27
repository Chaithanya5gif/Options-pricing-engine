import fitz  # PyMuPDF
import os

pdf_path = "/Users/apple/Downloads/options_paper.pdf"
output_path = "/Users/apple/Desktop/options/options_paper_with_images.pdf"

# Open the PDF
doc = fitz.open(pdf_path)

# Create a new blank page at the end
page = doc.new_page(-1, width=595, height=842) # A4 size

# The images to add
images = [
    "/Users/apple/Desktop/options/pricer_comparison.png",
    "/Users/apple/Desktop/options/volatility_forecast_comparison.png",
    "/Users/apple/Desktop/options/heston_smile.png"
]

# Insert images vertically
y_offset = 50
page.insert_text((50, y_offset), "Appendix B: Figures", fontsize=16, fontname="helv")
y_offset += 30

for img_path in images:
    if os.path.exists(img_path):
        # Insert image
        # Calculate a reasonable rectangle for the image
        img_rect = fitz.Rect(50, y_offset, 545, y_offset + 200) 
        page.insert_image(img_rect, filename=img_path, keep_proportion=True)
        y_offset += 230
    else:
        print(f"Image not found: {img_path}")

# Save the output
doc.save(output_path)
print(f"Successfully added images to {output_path}")
