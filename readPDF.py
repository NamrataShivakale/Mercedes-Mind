import fitz  # PyMuPDF
from langchain.text_splitter import CharacterTextSplitter
import os
import pickle

# File paths
pdf_path = 'MbrdiVehicleManual.pdf'
output_folder = 'extracted_images'
os.makedirs(output_folder, exist_ok=True)

# Step 1: Extract text from PDF using PyMuPDF
doc = fitz.open(pdf_path)
raw_text = ''
page_texts = {}

# Loop through each page in the PDF to extract text
for i in range(len(doc)):
    page = doc.load_page(i)  # Load a specific page
    content = page.get_text("text")  # Extract text from the page
    if content:
        raw_text += content
        page_texts[i] = content  # Map page number to its text

# Step 2: Extract images from the PDF
page_images = {}  # Map page number to list of image file paths

for i in range(len(doc)):
    images = doc[i].get_images(full=True)  # Get all images from the page
    img_paths = []

    for j, img in enumerate(images):
        xref = img[0]  # The image reference
        pix = fitz.Pixmap(doc, xref)

        try:
            # Check if the pixmap is in grayscale or RGB (1 = grayscale, 3 = RGB)
            if pix.colorspace.n not in (1, 3):  # If the pixmap is not grayscale or RGB
                print(f"Converting image {i+1}_{j} to RGB...")
                pix = fitz.Pixmap(fitz.csRGB, pix)  # Convert to RGB

            # Check if the pixmap is valid and has no issues
            if pix:
                img_filename = f"page_{i+1}_img_{j}.png"
                img_path = os.path.join(output_folder, img_filename)

                # Save image as PNG
                pix.save(img_path)
                img_paths.append(img_path)
            else:
                print(f"Skipping invalid pixmap at {i+1}_{j}")

        except Exception as e:
            print(f"Error processing image {i+1}_{j}: {e}")
        finally:
            pix = None  # Always release the pixmap to free memory

    if img_paths:
        page_images[i] = img_paths  # Store the image paths for the page

# Step 3: Split the text into chunks using LangChain's CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len
)

chunks_with_images = []

# Split the text and associate images
for page_num, text in page_texts.items():
    chunks = text_splitter.split_text(text)
    images = page_images.get(page_num, [])

    for chunk in chunks:
        chunks_with_images.append({
            "text": chunk,
            "images": images  # May be an empty list
        })

# Step 4: Save the chunks with text and image paths to a pickle file
with open('vehiclemanual_with_images.pickle', 'wb') as f:
    pickle.dump(chunks_with_images, f)

# Summary output
print(f"Total chunks created: {len(chunks_with_images)}")
print("Sample chunk:")
print(chunks_with_images[0])  # Show a sample of the first chunk
