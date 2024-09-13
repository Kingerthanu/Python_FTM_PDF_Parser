import openai                                                         # For OpenAI LLM
import os                                                             # For File Manipulation

# Suppress TensorFlow Logs By Setting The Environment Variable (Just Some Stuff About GPU Acceleration That Isn't Needed Right Now As We Only Are Using Blip..)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import fitz                                                           # PyMuPDF For PDF Reading
from PIL import Image                                                 # Handling Images Extracted From PDFs
import re                                                             # Regular Expressions
import io                                                             # File Interactions
import concurrent.futures                                             # Multi-Threading
import multiprocessing                                                # For Handling JVM (Tabula) In A Separate Process
import tabula                                                         # Table Extraction For PDFs Using Tabula
from rich.progress import Progress, BarColumn, TextColumn             # Progress Bar For File Status
from rich.live import Live                                            # Responsive Fixed Positions On The Terminal (Allow Progress Bar To Actively Update At Bottom Of Terminal Without Adding Extra Lines)
from threading import Lock                                            # Ensure Race-Conditions Are Handled From Each Worker Thread In Which Uses OCR To Scan A Image On Their Given PDF File
import time                                                           # For Logging Timestamps
import torch                                                          # For Blip (Maybe Local LLMs In Future?)
from transformers import BlipProcessor, BlipForConditionalGeneration  # Blip2 For General Image Captions (Still Kinda Iffy On It)


# Initialize BLIP-2 Processor And Model For Image Captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", clean_up_tokenization_spaces=True)
# Ensure that both the model and inputs are on the same device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Move the model to the correct device
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Config~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Initialize The OpenAI Client With Your Specific API Key
openai.api_key = "______"

# Specify The Directory Path Of Your PDFs
directory_path = r"______"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~END Config~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



'''
  Desc: Utilized To Be A UI For Users To Show PDF Parsing State Through Progress Bars,
  And Logging Information With General Tags. The Class Utilizes Mutex Locks To Ensure Thread-Safety
  When Writing To The Logger To Synchronize Access Between The Progress Bar And Log Information In The Terminal.
  Class Is Used As It's Difficult To Display Progress Bars Without Proper Synchronization From An Async Source Like Prints.

  
  Preconditions:
    1.) progress Is A Valid rich.progress Progress Bar Object.
    2.) debug_file Is A Valid Path To A Debug Log File.
  
  Postconditions:
    1.) Logs Are Synchronized And Printed Safely Even In A Multithreaded Environment.
    2.) Progress Bars Are Updated After Logging To Ensure Correct Order Of Output.
'''
class ProgressLogger:

  def __init__(self, progress: Progress, debug_file: str) -> None:
    # Set Progress Bar
    self.progress = progress
    # Setup Where log Messages Will Reside For Printing
    self.logs = []
    # Thread Lock To Synchronize Access To Logs And Progress Bars
    self.lock = Lock()
    # Path To Debug File
    self.debug_file = debug_file


  '''
  
  Desc: Will Format message Into A Log Message With A Time Stamp As Well As Warning Level
  And Print It To Console While Flushing. Had It Batching Logs To Maybe Integrate A Timer To
  Execute Prints Of The Logs Periodically But Currently Just Printing And Flushing.
  !! Know That There Is A Bug Where External Languages Libraries Are Throwing Errors In Terminal And Messing Up The Formatting Of Progress Bar !!

  
  Preconditions:
    1.) message Is The Provided Log Message To Be Posted.
    2.) level Is The Severity Of The Log (Can Be Anything You Pass, But Recommend ["INFO", "WARNING", "ERROR", "FAILED", "SUCCEEDED"]).
  
  Postconditions:
    1.) message Is Synchronized And Printed Thread-Safely.
    2.) formatted_message (The Log) Will Be Written Into The Debug File

  '''
  def log(self, message : str, level : str = "INFO") -> None:
    '''Appends A Log Message With A Specific Level And Ensures It Is Written Before Progress Updates.'''
    with self.lock:
      # Add A Timestamp To Each Log Message For Debugging Purposes
      timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
      # Format The Log Message With A Timestamp And Log Level
      formatted_message = f"{timestamp} [{level}] - {message}"
      
      # Simply Print To Console Through Progress Bar Synchronization
      self.progress.console.print(formatted_message)
        # Append The Formatted Message To The Log Buffer
        # self.logs.append(formatted_message)
        # self.flush_logs()
      # Write The Log To A Debug File
      self.write_to_debug_file(formatted_message)


  '''
  
  Desc: Is Utilized For Batch Printing self.logs And Flushing The Buffer Afterwards
  To Ensure It Has Its Contents Cleared Before We Print Our self.progress Bar(s) At The Bottom Of The Terminal.
  Improper Flushing Synchronization With The Progress Bar Can Cause The Progress Bar To Glitch Out.

  
  Preconditions:
    1.) Intend To Print All Messages In self.logs
  
  Postconditions:
    1.) Will Print Every log Contained In self.logs And Will Flush The Buffer To Ensure Everything Is Given To User
    2.) Will Clear The Contents Of self.logs After Printing Them (Will Still Be Saved In debug File)
    
  '''
  def flush_logs(self) -> None:
    # Iterate Over All Logs In The Buffer And Print Them
    for log in self.logs:
      self.progress.console.print(log)
    # Clear The Buffer After Printing
    self.logs.clear()


  '''
  
  Desc: Prints The Logs Currently Waiting To Be Printed And Then Will Update The Progress Bar With The Specified 
  task/Bar ID By Incrementing It By advance Units Forward (Negative Would Be Backwards). 

  
  Preconditions:
    1.) task_id Is A Valid ID For A Progress Bar Currently Embedded Within self.progress
    2.) advance Will Be How Many Steps We Go Forward On Our Bar Graph (I.E. If Advance == 2 We Would Go From Page 1 -> Page 3)
    3.) self.logs Will Be Printed And Cleared
  
  Postconditions:
    1.) Will Print And Clear The Contents Of self.logs
    2.) Will Update The Progress Bar Pointed To By task_id By going advance Units Forward And Changing The Caption Of The Bar With description
    
  '''
  def update_progress(self, task_id : str, advance : int = 1, description : str = None) -> None:
    with self.lock:
      # Don't Need To Flush Anymore By Printing Through self.progress.console.print(...)
        # Flush Logs Before Updating Progress To Maintain Output Order
        # self.flush_logs()
      # Update The Progress With An Optional Description
      if description:
        self.progress.update(task_id, advance=advance, description=description)
      else:
        self.progress.update(task_id, advance=advance)


  '''
  
  Desc: Used To Save Away Logging Information In A File For Easy Access Later.
  Currently Being Employed To Silently Print Parsing Results To See Actually What The Parser Is Getting Out Of Our PDFs.
  
  Preconditions:
    1.) Will Append Message Contents In File; This Can Cause Build-Up Of Log Data If Not Cleared

  Postconditions:
    1.) Will Write message's Contents To self.debug_file With A New Line At The End
    
  '''
  def write_to_debug_file(self, message : str) -> None:
    with open(self.debug_file, 'a', encoding='utf-8') as f:
      f.write(message + '\n')



'''

  Desc: Calls BLIP-2 In Order To Use Its Multi-Modal Fine-Tuned Model To Caption/Describe An Image.
  Probably Will Soon Be Deprecated As Isn't Really That Good At Thorough Descriptions. Will Probs Use
  LLaVa In Place.
  
  Preconditions:
    1.) image Must Be A Valid PIL Image Object.
  
  Postconditions:
    1.) Returns A String Caption Describing The Image Using The BLIP Model.
    2.) Returns An Empty String If An Error Occurs During BLIP Processing.

'''
def analyze_image_with_blip2(image) -> str:
  # print("Running BLIP...")
  try:
    # Process the image using BLIP-2 for captioning
    inputs = processor(image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set parameters to increase the length and detail of the generated caption
    outputs = model.generate(
      **inputs,
      max_length=450,  # Adjust to increase token length
      num_beams=25,  # Increase the number of beams for better quality
      no_repeat_ngram_size=2,  # Prevent repetitive captions
      early_stopping=True,  # Stop when a suitable caption is found
      do_sample=True,
      temperature=0.25,  # Control randomness (lower values for more deterministic results)
    )
    
    # Decode the model output into a human-readable caption
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption
  except Exception as e:
    # Log any errors encountered during BLIP processing
    print(f"BLIP-2 error: {e}")
    return ""


'''

  Desc: Function Splits A Given Text, text, Into Chunks To Prevent Exceeding OpenAI's Token Limit.
  Currently Have It Set To 2000 Tokens.
  
  Preconditions:
    1.) text Is A String That Needs To Be Chunked.
    2.) chunk_size Defines The Maximum Number Of Words Per Chunk.
    3.) Is A Generator Function 
  
  Postconditions:
    1.) Yields Chunks Of Text With At Most chunk_size Words Per Chunk.

'''
def chunk_text(text : str, chunk_size=2000):
  words = text.split()
  # Yield Chunks Of Words Up To The Specified Chunk Size
  for i in range(0, len(words), chunk_size):
    yield ' '.join(words[i:i + chunk_size])


'''

  Desc: This Function Uses Tesseract Optical Character Recognition (OCR) To Extract Text From A Provided Image.
  This Means It Will Extract Embedded Text In Images As Text Like That Is Purely A Image And Needs To Be Pattern 
  Recognized.
  
  Preconditions:
    1.) image Must Be A Valid PIL Image Object.
  
  Postconditions:
    1.) Returns A String Containing The Text Extracted From The Image.
    2.) Returns An Empty String If Tesseract OCR Encounters An Error.


Deprecated, Doesn't Work Well. Will Transfer To LLaVa
def extract_text_from_image(image: Image.Image) -> str:
  """
  Run OCR on the preprocessed image.
  """
  try:
    # Run OCR on the preprocessed image using pytesseract
    text = pytesseract.image_to_string(image, config='--psm 3')
    return text
  except Exception as e:
    print(f"OCR Error: {e}")
    return ""

'''

'''

  Desc: Extracts Text, Images, And Tables From A PDF Using PyMuPDF.

  Preconditions:
    1.) file_path Must Point To A Valid PDF File.
    2.) logger Is An Instance Of ProgressLogger For Managing Logs.
  
  Postconditions:
    1.) Returns A List Of Pages With Extracted Text, Images, Metadata, And Tables From The PDF.

'''
def extract_content_with_mupdf(file_path : str, logger : ProgressLogger) -> str:
  logger.log(f"Extracting content from PDF: {file_path}")

  # Open The PDF Using PyMuPDF
  doc = fitz.open(file_path)
  pages = []

  # Iterate Through Each Page In The PDF
  for page_num, page in enumerate(doc):
    text = page.get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES)
    image_list = page.get_images(full=True)
    images = []

    # Extract And Process Each Image In The Page
    for img_index, img in enumerate(image_list):
      xref = img[0]
      base_image = doc.extract_image(xref)
      image_bytes = base_image["image"]
      image = Image.open(io.BytesIO(image_bytes))
      images.append(image)

      # Perform BLIP-2 Image Analysis And Log The Caption
      logger.log(f"Using BLIP-2 to analyze image on page {page_num + 1} of {file_path}...")
      blip_caption = analyze_image_with_blip2(image)
      if blip_caption != "":
        logger.log(f"Blip-2 gained a caption.")
      else:
        logger.log(f"Failed gaining Blip-2 caption.")
      logger.write_to_debug_file(f"BLIP-2 Caption: {blip_caption}")

      # Perform OCR On The Image And Log The Extracted Text
      # logger.log(f"Using Tesseract OCR to process image on page {page_num + 1} of {file_path}...")
      # OCR Kinda Sucks For General Purpose.. ocr_text = extract_text_from_image(image)
      # OCR Kinda Sucks For General Purpose.. logger.write_to_debug_file(f"OCR extracted text: {ocr_text}")

    # Combine OCR Text And BLIP Captions For The Current Page
    # OCR Kinda Sucks For General Purpose.. combined_ocr_text = " ".join([extract_text_from_image(img) for img in images]) if images else ""
    blip_caption = " ".join([analyze_image_with_blip2(img) for img in images]) if images else ""

    # Extract Table Data From The Page
    logger.log(f"Attempting table extraction with Tabula on page {page_num + 1} of {file_path}...")
    table_data = extract_table_data_with_tabula_multiprocessing(file_path, page_num + 1)
    if table_data != "":
      logger.log(f"Successfully extracted table data.")
    else:
      logger.log(f"Failed extracting table data.")

    # Append The Extracted Data To The List Of Pages
    if text or blip_caption or table_data:
      pages.append({
        "page_num": page_num,
        "text": text + "  " + blip_caption + "  " + table_data,
        "images": images,
        "metadata": doc.metadata,
        "annotations": list(page.annots()) if page.annots() else [],
        "links": page.get_links()
      })

      logger.write_to_debug_file(f"Parsed raw text on page {page_num + 1} of {file_path}: \n{(pages[page_num]['text'])}")

  # Close The PDF Document After Processing
  doc.close()
  return pages


'''

  Desc: Cleans Up The Extracted Text By Removing Extra Whitespace Through Regular Expressions.
  
  Preconditions:
    1.) text Is A String Containing The Extracted PDF Text.
  
  Postconditions:
    1.) Returns A Cleaned String With Excessive Whitespace Removed.

'''
def preprocess_text(text):
  text = re.sub(r'\s+', ' ', text)  # Normalize Excessive Whitespace
  return text.strip()


'''

  Desc: Extracts Tables From A Specific Page Of A PDF Using Tabula.
  
  Preconditions:
    1.) file_path Is The Path To A Valid PDF File.
    2.) page_num Specifies The Page Number To Extract The Table From.
  
  Postconditions:
    1.) Returns A String Representation Of Extracted Tables.
    2.) Returns An Error Message If Table Extraction Fails.

'''
def extract_table_data_with_tabula(file_path : str, page_num : str) -> str:
  try:
    # Use Tabula To Read Tables From The Specified PDF Page
    tables = tabula.read_pdf(file_path, pages=page_num, multiple_tables=True)
    if tables and len(tables) > 0:
      return "\n".join([table.to_string() for table in tables])
    else:
      return ""
  except Exception as e:
    # Return An Error Message If Table Extraction Fails
    return f"Error extracting tables from page {page_num} of {file_path}: {e}"


'''

  Desc: Uses Multi-Processing To Avoid Java Virtual Machine (JVM) Race-Conditions When Extracting Tables Using Tabula.
  
  Preconditions:
    1.) file_path Is The Path To A Valid PDF File.
    2.) page_num Specifies The Page Number To Extract The Table From.
  
  Postconditions:
    1.) Returns The result Of Table Extraction Using Tabula In A Separate Process.

'''
def extract_table_data_with_tabula_multiprocessing(file_path : str, page_num : str) -> str:
  # Use A Separate Process To Extract Tables From The PDF Page
  with multiprocessing.Pool(1) as pool:
    result = pool.apply(extract_table_data_with_tabula, (file_path, page_num))
  return result


'''

  Desc: Sends A Text Chunk To OpenAI GPT-4o-mini For Processing, With Retry Logic For Transient/Future Failures.
  
  Preconditions:
    1.) text_chunk Is A String Representing The Text To Be Sent To The OpenAI API.
    2.) retries Defines The Maximum Number Of Retry Attempts.
    3.) delay Specifies The Delay Between Retry Attempts.
  
  Postconditions:
    1.) Returns The Response From OpenAI If Successful.
    2.) Returns None If All Retry Attempts Fail.

'''
def send_to_openai_with_retry(text_chunk : str, previous_context : str = "", file_path : str = "", page_num : str ="", logger = None, retries : int = 3, delay : int = 2):
  for attempt in range(retries):
    # Try To Send The Text Chunk To OpenAI
    response_text = send_to_openai(text_chunk, previous_context, file_path, page_num)
    if response_text:
      return response_text
    else:
      # Log The Retry Attempt If It Fails
      if logger:
        logger.log(f"Retrying API call for page {page_num} of {file_path}... (Attempt {attempt + 1} of {retries})", "WARNING")
      time.sleep(delay)  # Wait Before Retrying
  # Return None If All Retry Attempts Fail
  return None


'''

  Desc: Sends A Text Chunk And Context To OpenAI GPT-4o-mini For Processing And Fine-Tuning Data Generation.
  
  Preconditions:
    1.) text_chunk Contains The Text To Be Sent To The OpenAI API As Fine-Tuning Data.
    2.) previous_context Is The Context From Previous Pages Or Chunks Of Fine-Tuning Data Being Sent.
    3.) file_path And page_num Are For Debugging And Logging.
  
  Postconditions:
    1.) Returns The Training Data Response Content From The OpenAI Model.
    2.) Returns None If An Error Occurs During The API Call.

'''
def send_to_openai(text_chunk, previous_context="", file_path="", page_num=""):
  try:
    prompt = f"""
    Using the provided TEXT CONTENT from page {page_num} of {file_path} and PRIOR CONTEXT, generate **fine-tuning data** that breaks down the integration of hardware and software for the Arduino Mega 2560 Rev3 Board.

    Specifically focus on the following areas, but feel free to expand beyond these as necessary:

    - Pinmap diagrams and how each pin interacts with peripherals such as GPIO, PWM, ADC, and communication protocols (I2C, SPI, UART). Explain in great detail which pins should be used for different components and why.
    - Exact physical locations and pinouts on the board, specifying how to wire components such as sensors, motors, and communication modules to specific pins.
    - Simulating real-world hardware setups: Provide step-by-step guidance on how users can simulate the Arduino Mega 2560 Rev3 Board using software tools, including mock hardware interfaces.
    - Theoretical aspects of hardware pin functionalities: Provide voltage/current limits, signal protocols, and other hardware constraints.
    - Practical software integration: Provide detailed programming instructions (including libraries, best practices, and detailed code examples) for interfacing with hardware components.
    - Full lifecycle simulation: From low-level hardware configuration (pin mapping, timers, interrupts) to high-level abstractions, including how to create software simulations that reflect real-world hardware performance.
    - Real-world implementation challenges: Provide step-by-step solutions for connecting sensors, controlling motors, managing power efficiently, etc., and how to simulate such setups in a virtual environment.
    - Detail how pin-specific operations can be simulated, including communication protocols like I2C, SPI, UART, and how real-time system constraints (such as interrupts and timers) are handled both physically and in simulation.

    **Important**: The list above is not exhaustive, and your responses should adapt based on the technical problems presented in TEXT CONTENT and PRIOR CONTEXT. Your answers must be **extremely detailed** and cover both hardware and software implementation in a simulatory environment.

    TEXT CONTENT:
    <info>{text_chunk}</info>

    PRIOR CONTEXT:
    <info>{previous_context}</info>

    Ensure that each response provides detailed, actionable insights for both physical hardware setups and simulations, including diagrams, wiring instructions, theoretical breakdowns, and step-by-step code examples where applicable. Each response must generate at least **50 distinct, thorough questions and answers per response** to comprehensively address all aspects of the board's hardware and software integration lifecycle, ensuring no repeat questions are given.

    The format should include a detailed conversation of technical challenges and solutions (ensure at least 50 strong examples derived from TEXT CONTENT AND PRIOR CONTENT are given as "messages" strictly following this format!):

    {{
        "messages": [
          {{"role": "system", "content": "You are an expert in low-level hardware-software integration for Arduino Mega 2560 Rev3 Board, specializing in simulatory environments, embedded systems, detailed hardware pin mapping, and software integration from low-level configuration to high-level abstractions."}},
          {{"role": "user", "content": "(A highly specific technical question about the Arduino 2560 Rev3 Board board that pertains to the documentation as described in the provided TEXT CONTENT)"}},
          {{"role": "assistant", "content": "(A detailed response including technical explanations, practical examples, wiring diagrams, code snippets, and simulation details. Provide insight on both hardware and software implementation.)"}}
        ]
    }}
    {{
        "messages": [
          {{"role": "system", "content": "You are an expert in low-level hardware-software integration for Arduino Mega 2560 Rev3 Board, specializing in simulatory environments, embedded systems, detailed hardware pin mapping, and software integration from low-level configuration to high-level abstractions."}},
          {{"role": "user", "content": "(A highly specific technical question about the Arduino 2560 Rev3 Board board that pertains to the documentation as described in the provided TEXT CONTENT)"}},
          {{"role": "assistant", "content": "(A detailed response including technical explanations, practical examples, wiring diagrams, code snippets, and simulation details. Provide insight on both hardware and software implementation.)"}}]
    }}
    """

    response = openai.chat.completions.create(
      model="gpt-4o-mini-2024-07-18",
      messages=[
        {"role": "system", "content": (
            "You are an expert in simulatory hardware-software integration for Arduino Mega 2560 Rev3 Board, "
            "specializing in physical setups, simulations, low-level design, and real-time systems. You provide deeply detailed answers across all phases, from pin-level configuration to system-wide simulation and optimization."
        )},
        {"role": "user", "content": prompt}
      ],
      max_tokens=3000,
      temperature=0.5,
      top_p=0.9
    )
    return response.choices[0].message.content
  except Exception as e:
      return None  # Return None If API Call Fails


'''

  Desc: Processes A PDF File Sequentially Page-By-Page, Extracting Text, Tables, And Images, While Managing Progress Updates Using A ProgressLogger.
  
  Preconditions:
    1.) file_path Is The Path To The PDF File.
    2.) output_dir Specifies The Directory To Save The Processed Data.
    3.) progress_logger Is An Instance Of ProgressLogger For Logging And Progress Bar Updates.
    4.) overall_task_id And pdf_task_id Represent The Progress Bar Task IDs.
    5.) total_pages Specifies The Total Number Of Pages In The PDF.
  
  Postconditions:
    1.) Extracted Data Is Saved To The Output Directory As JSON Files.
    2.) Logs Progress And Any Errors Encountered During PDF Processing.

'''
def process_pdf_with_summaries(file_path : str, output_dir : str, progress_logger : ProgressLogger, overall_task_id : str, pdf_task_id : str, total_pages : str) -> None:
  progress_logger.log(f"Starting PDF processing: {file_path}")
  
  # Extract The Content From The PDF
  pages = extract_content_with_mupdf(file_path, progress_logger)
  all_responses = []
  context_summaries = []

  # Process Each Page Individually
  for page in pages:
    progress_logger.log(f"Parsing page {page['page_num'] + 1} of {total_pages} in PDF: {file_path}")
    
    # Update Progress Bars For Both Overall Progress And PDF-Specific Progress
    progress_logger.update_progress(overall_task_id, advance=1)
    progress_logger.update_progress(pdf_task_id, advance=1, description=f"Parsing page {page['page_num'] + 1}/{total_pages}")

    # Preprocess The Extracted Text
    text_content = preprocess_text(page['text'])
    # OCR Kinda Sucks For General Purpose.. ocr_texts = [extract_text_from_image(img) for img in page['images']]
    combined_text = text_content + " "

    # Use The Last Two Context Summaries For The Current Page
    previous_context = " ".join(context_summaries[-2:])

    # Send Each Chunk Of The Combined Text To OpenAI For Fine-Tuning
    for text_chunk in chunk_text(combined_text):
      response_text = send_to_openai_with_retry(text_chunk, previous_context, file_path, page['page_num'] + 1, progress_logger)
      if response_text:
        cleaned_response = clean_response(response_text)
        all_responses.append(cleaned_response)
      else:
        progress_logger.log(f"Error: No response received for page {page['page_num'] + 1} of {file_path}.", level="ERROR")

    # Generate A Summary Of The Extracted Text And Context
    page_summary = summarize_text(combined_text, file_path, page['page_num'] + 1)
    context_summaries.append(page_summary)

  # Save The Processed Data To JSON Files
  if all_responses:
    output_raw_file_path = os.path.join(output_dir, os.path.basename(file_path).replace('.pdf', '_raw_fine_tuning.json'))
    output_cleaned_file_path = os.path.join(output_dir, os.path.basename(file_path).replace('.pdf', '_cleaned_fine_tuning.json'))

    # Save Raw And Cleaned Responses
    with open(output_raw_file_path, 'w', encoding='utf-8') as output_file:
      output_file.write('\n'.join(all_responses))

    with open(output_cleaned_file_path, 'w', encoding='utf-8') as output_file:
      output_file.write('\n'.join(all_responses))

    # Log The Completion Of PDF Processing
    progress_logger.log(f"Finished processing {file_path}. Raw output saved to {output_raw_file_path}, Cleaned output saved to {output_cleaned_file_path}")
  else:
    # Log If No Data Was Extracted
    progress_logger.log(f"No data extracted from {file_path}. No files written.")


'''

  Desc: Summarizes Extracted Text Using OpenAI's GPT-4o-mini To Reduce Token Usage On Context By Summarizing Previous Pages Of Text.
  
  Preconditions:
    1.) text Is The Extracted Text From A Page.
    2.) file_path And page_num Are For Debugging And Logging.
  
  Postconditions:
    1.) Returns A Concise Summary Of The Extracted Text.
    2.) Returns The Original Text If The OpenAI API Call Fails.

'''
def summarize_text(text, file_path, page_num):
  try:
    # Use GPT-4 To Summarize The Extracted Text
    response = openai.chat.completions.create(
      model="gpt-4o-mini-2024-07-18",
      messages=[
        {"role": "system", "content": "You are an expert in summarizing technical documentation."},
        {"role": "user", "content": f"Summarize the following text from page {page_num} of {file_path}: {text}"}
      ],
      max_tokens=500,
      temperature=0.5
    )
    return response.choices[0].message.content.strip()
  except Exception as e:
      # Return The Original Text If Summarization Fails
      return text


'''

  Desc: Cleans The Response Text From OpenAI By Removing Unnecessary Formatting Or Artifacts.
  
  Preconditions:
    1.) response_text Is A String Containing A OpenAI Model's Response.
    2.) The Formatting Of .json Is As Follows:
        {{
          "messages": [
              {{"role": "system", "content": "_______"}},
              {{"role": "user", "content": "_______"}},
              {{"role": "assistant", "content": "_______"}}
          ]
        }}
        {{
            "messages": [
                {{"role": "system", "content": "_______"}},
                {{"role": "user", "content": "_______"}},
                {{"role": "assistant", "content": "_______"}}
          ]
        }}
    

  Postconditions:
    1.) Returns The Cleaned Response Text Ready For Use.
    2.) Returns The Original Response If Cleaning Fails.

'''
def clean_response(response_text):
  try:
    # Find And Clean JSON Blocks From The Response Text
    json_pattern = r'(\{\s*"messages":\s*\[.*?\]\s*\})'
    json_blocks = re.findall(json_pattern, response_text, flags=re.DOTALL)
    cleaned_text = "\n".join(json_blocks)
    cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)
    return cleaned_text.strip()
  except Exception as e:
    # Log Any Errors During Cleaning And Return The Original Response
    print(f"Error cleaning response: {e}")
    return response_text


'''
  Desc: Is Given A Folder Containing PDF Documents In Which To Fine-Tune An AI Model On.
  For Each PDF Provided, It Will Be Given A Designated Worker Thread In Which Will Get Fine-Tuning
  Data For That Specific PDF. This Includes openai Communication Of Prompts And Returning The Nested, Cleaned .json Data.
  It Also Establishes A Logger In Which Will Work With Each One Of These Threads To Ensure We Have Proper Tabbing On Their Statuses
  Through Progress Bars And Logging Messages.
  
  Preconditions:
    1.) directory_path Must Point To A Valid Directory Containing PDF Files.
  
  Postconditions:
    1.) Each PDF In The Directory Is Processed And Logs/Progress Are Updated.
    2.) Processed Data Is Saved To The Output Directory As JSON Files.
'''
def multiThreaded_process_directory(directory_path):
  # Start Of Logging
  print("\n-----------------------------------------Start Of Logging--------------------------------------------------", flush=True)

  # Create Output Directory For Processed Data
  output_dir = os.path.join(directory_path, 'processed_fine_tuning')
  os.makedirs(output_dir, exist_ok=True)

  # Create Debug Log File Directory
  debug_file_path = os.path.join(directory_path, 'debug', 'pdf_processing_debug.log')
  os.makedirs(os.path.dirname(debug_file_path), exist_ok=True)

  # Gather All PDF Files In The Directory
  pdf_files = [os.path.join(directory_path, file_name) for file_name in os.listdir(directory_path) if file_name.endswith('.pdf')]

  # Create A Progress Bar To Track The Progress Of All PDFs
  progress = Progress(
    TextColumn("[bold green]{task.description}"),
    BarColumn(bar_width=None),
    "[progress.percentage]{task.percentage:>3.0f}%",
    "• Processed: {task.completed}/{task.total} pages"
  )

  # Create Overall Progress Task For All PDFs
  overall_task = progress.add_task("[cyan]Overall Progress...", total=sum([len(fitz.open(pdf)) for pdf in pdf_files]))

  # Create ProgressLogger Object To Manage Both Progress And Logs
  progress_logger = ProgressLogger(progress, debug_file=debug_file_path)

  # Group The Progress Bars Together And Run Inside The Live Context
  with Live(progress, refresh_per_second=2):
    # Process Each PDF In Parallel Using A Thread Pool
    with concurrent.futures.ThreadPoolExecutor() as executor:
      futures = []
      for pdf_file in pdf_files:
        total_pages = len(fitz.open(pdf_file))
        pdf_task = progress.add_task(f"[yellow]Parsing {os.path.basename(pdf_file)}...", total=total_pages)
        futures.append(executor.submit(process_pdf_with_summaries, pdf_file, output_dir, progress_logger, overall_task, pdf_task, total_pages))

      # Gather Results As Each Future Completes
      for future in concurrent.futures.as_completed(futures):
        try:
          future.result()
        except Exception as e:
          progress_logger.log(f"Error processing PDF: {e}")

  # End Of Logging
  print("-----------------------------------------End Of Logging--------------------------------------------------\n", flush=True)



# Main Driver (Program Starts Here, Calling multiThreaded_process_directory(...) With A Valid File Containing Training PDFs)
if __name__ == "__main__":
  multiThreaded_process_directory(directory_path)