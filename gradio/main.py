import gradio as gr
import requests
from PIL import Image
from io import BytesIO
import json
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define the URL of your FastAPI endpoints
API_URL_SEARCH = "http://localhost:8000/api/search_serv/search/"
API_URL_IMAGE = "http://localhost:8000/api/image_serv/images_source/"
API_URL_METADATA = "http://localhost:8000/api/image_serv/images_metadata/"
import os
# Function to call FastAPI search endpoint
def search_indexes(file: gr.File, search_by, distance_type):
    logger.debug("Starting search_indexes function")
    # file_content = file.read()  # This will give you the byte content of the file
    
    # Now, send the file via an HTTP request (POST method)
    print(file)
    files = {'file': (file.name, open(file,'rb'))}

    data = {
        'search_by': search_by,
        'distance_type': distance_type
    }
    
    # Send the POST request to the FastAPI endpoint
    logger.info("Sending POST request to FastAPI")
    try:
        response = requests.post(API_URL_SEARCH, files=files, data=data)
        
        # Log response details
        if response.status_code == 200:
            logger.info(f"Received successful response: {response.status_code}")
            json_response = response.json()
            # formatted_response = "\n".join([f"ID: {item['id']}, Cosine Similarity: {item['cosine_similarity_embs']}" for item in json_response])
            return list(map(lambda x: x['id'], json_response))
        else:
            logger.error(f"Error response from FastAPI: {response.status_code}, {response.text}")
            return f"Error: {response.status_code}, {response.text}"
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return f"Error occurred while making request: {e}"

# Function to fetch an image by ID from FastAPI
def get_image_by_id(image_id):
    logger.debug(f"Fetching image with ID: {image_id}")
    
    # Send GET request to fetch the image
    try:
        response = requests.get(f"{API_URL_IMAGE}{image_id}")
        
        if response.status_code == 200:
            logger.info(f"Image fetched successfully with ID: {image_id}")
            image = Image.open(BytesIO(response.content))
            return image
        else:
            logger.error(f"Error fetching image: {response.status_code}, {response.text}")
            return f"Error: {response.status_code}, {response.text}"
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return f"Error occurred while making request: {e}"

# Function to fetch metadata of an image by ID from FastAPI
def get_image_metadata_by_id(image_id):
    logger.debug(f"Fetching metadata for image with ID: {image_id}")
    
    # Send GET request to fetch the metadata
    try:
        response = requests.get(f"{API_URL_METADATA}{image_id}")
        
        if response.status_code == 200:
            logger.info(f"Metadata fetched successfully for image ID: {image_id}")
            metadata = response.json()  # Assuming metadata is returned in JSON format
            formatted_metadata = json.dumps(metadata, indent=4)  # Format JSON for display
            return formatted_metadata
        else:
            logger.error(f"Error fetching metadata: {response.status_code}, {response.text}")
            return f"Error: {response.status_code}, {response.text}"
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return f"Error occurred while making request: {e}"

# Function to handle fetching multiple images and metadata
def get_images_and_metadata(ids_str):
    # Convert the comma-separated IDs string to a list of IDs
    ids = ids_str.split(',')
    images = []
    metadata = []
    
    # Fetch images and metadata for each ID
    for image_id in ids:
        image_id = image_id.strip()  # Remove extra spaces
        if image_id.isdigit():  # Check if the ID is a valid number
            # Fetch image and metadata
            image = get_image_by_id(image_id)
            meta = get_image_metadata_by_id(image_id)
            
            images.append(image)
            metadata.append(meta)
    
    # Return images and metadata
    return images, metadata

# Define the URL of your FastAPI parser endpoint
API_URL = "http://localhost:8000/api/parser/pars/"

# Function to fetch parsed image URLs
def parse_images(url: str):
    # Send the URL to the FastAPI server for parsing
    response = requests.post(API_URL, params={"text": url})
    
    if response.status_code == 200:
        data = response.json()  # Parse the JSON response
        image_urls = [entry['id'] for entry in data]
        return image_urls
    else:
        return f"Error: {response.status_code} - {response.text}"

# Function to display parsed image data in gallery format
def gallery_display(url: str):
    image_urls = parse_images(url)
    if isinstance(image_urls, list):  # If the output is a list of image URLs
        return image_urls
    else:
        return image_urls  # Return error message if not a list

# Create Gradio Interface for Image Parser
def image_parser_page():
    with gr.Row():
        gr.Markdown("### Image Parser")
        url_input = gr.Textbox(label="Enter URL to parse")
        gallery_output = gr.Textbox(label="Parsed Images")
        url_input.submit(gallery_display, inputs=url_input, outputs=gallery_output)

# Create main Gradio Interface with multiple pages

# Create Gradio interface
with gr.Blocks() as demo:
    with gr.Tab("Search Indexes"):
        # Create input elements for search
        file_input = gr.File(label="Upload File")  # File input
        search_by_input = gr.Radio(["class", "one_shot_embedding"], label="Search By")  # Search type
        distance_type_input = gr.Radio(["cosine", "euclidean", "manhattan"], label="Distance Type")  # Distance type
        output_text = gr.Textbox(label="Search Results")
        
        # Set up the search functionality
        search_button = gr.Button("Search")
        search_button.click(search_indexes, inputs=[file_input, search_by_input, distance_type_input], outputs=output_text)
    
    with gr.Tab("Fetch Images and Metadata by IDs"):
        # Create input elements for multiple IDs
        ids_input = gr.Textbox(label="Enter Image IDs (comma-separated)", value="1,2,3")  # Input for image IDs
        output_gallery = gr.Gallery(label="Image Gallery")  # Gallery for images
        output_metadata = gr.Textbox(label="Images Metadata", lines=6)  # Display metadata for all images
        
        # Set up the button to fetch multiple images and metadata
        fetch_images_button = gr.Button("Fetch Images and Metadata")
        
        # Fetch both images and metadata when the button is clicked
        fetch_images_button.click(
            fn=get_images_and_metadata, 
            inputs=ids_input, 
            outputs=[output_gallery, output_metadata]
        )
    with gr.Tab("Image Parser"):
        image_parser_page()

# Launch the Gradio interface
demo.launch()
