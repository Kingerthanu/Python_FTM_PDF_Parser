import openai                                                         # For OpenAI LLM
import os                                                             # For File Manipulation
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
import ollama                                                         # Locally Load In LLaMA 3.1 And LLaVA 1.6
import numpy as np                                                    # Matrix-Based Image Handling/Processing
import cv2                                                            # Image Processing
import pytesseract                                                    # OCR Scanning

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Config~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Initialize The OpenAI Client With Your Specific API Key
openai.api_key = "______"

# Specify The Directory Path Of Your PDFs
directory_path = r"______"

focus_context = "Real-Time Performance in RISC-V"

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
  Desc: Preprocesses An Image With OCR To Improve Text-Recognition Accuracy

  Preconditions:
    1.) image Must Be A Valid PIL Image Loaded In Memory.
  
  Postconditions:
    1.) Returns An Image In Which Is Color-Corrected From RGB To GreyScale And Brightened Through Matrix-Sharpening To Help Improve Letter Captures.
'''
def preprocess_image(image):
  # Convert PIL image to numpy array for OpenCV processing
  image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
  # Increase Image Size To Help With OCR Detection
  image_cv2 = cv2.resize(image_cv2, None, fx=1.75, fy=1.75, interpolation=cv2.INTER_LINEAR)

  # Apply Sharpening Filter
  kernel = np.array([[0, -1, 0],
                      [-1, 4.95, -1],
                      [0, -1, 0]])
  
  # Sharpen Our Image With A Min Depth Of -1 Possible (image_cv2 Will Be Set As A Matrix Obj)
  image_cv2 = cv2.filter2D(image_cv2, -1, kernel)
  
  # Translate Our Matrix Obj To A Image In Memory
  return Image.fromarray(image_cv2)


'''
  Desc: Perform OCR On The Preprocessed image To Extract Any Text.

  Preconditions:
    1.) image Must Be A Valid PIL Image Object.
  
  Postconditions:
    1.) Returns A String Containing The Text Extracted From The Image.
'''
def perform_ocr(image):
  # Convert Image To OpenCV Format
  image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
  # Define Custom Configuration For Tesseract
  custom_config = r'--oem 3 --psm 11'
  # Run OCR On The Preprocessed image Using pytesseract
  ocr_result = pytesseract.image_to_string(image_cv2, config=custom_config)
  return ocr_result


'''
  Desc: Generate A Detailed image Description Using LLaVA With The Preprocessed Image.

  Preconditions:
    1.) image Must Be A Valid Image In Memory.
    2.) ocr_context Contains The OCR-extracted Text From The Same image.
  
  Postconditions:
    1.) Returns A String With LLaVA's Detailed Analysis Of The image.
'''
def generate_image_description(image):
  # Convert CMYK images to RGB before saving as PNG
  if image.mode == "CMYK":
    image = image.convert("RGB")

  # Convert The image To The In-Memory Format Required For LLaVA
  img_byte_arr = io.BytesIO()
  
  # Save image's Contents In img_byte_arr As A PNG
  image.save(img_byte_arr, format='PNG')
  img_byte_arr = img_byte_arr.getvalue()

  # Ask LLaVA To Describe The Contents Of The Image
  res = ollama.chat(
    model="llava:latest",
    messages=[
      {
        "role": "user",
        "content": f"""
        You are provided with a technical diagram that may be a schematic, chart, graph, or blueprint from topics pertaining to {focus_context}. 
        Focus on identifying key details, explain component labels, data points, and the relationships between elements. 
        If it's a chart or graph, ensure to describe axis labels, scales, and any trends you can observe. 
        Please ensure your explanation is highly detailed and technical, assuming the reader has engineering knowledge.
        """,
        "images": [img_byte_arr],
        "temperature": 0.25  # Lower Temperature For A More Deterministic And Detailed Output Based More On The Image's Contents Then LLaVA's Interpretation
      }
    ]
  )
  return res["message"]["content"]


'''
  Desc: Craft A Detailed Prompt For LLaMA Based On The OCR And LLaVA Description.

  Preconditions:
    1.) ocr_text Contains The Text Extracted Via OCR.
    2.) image_description Contains LLaVA's Description Of The Image.
  
  Postconditions:
    1.) Returns A String Prompt For Use With LLaMA Analysis.
'''
def create_finetuning_prompt(ocr_text, image_description):
  prompt = f"""
  You are provided with both the extracted text from a technical diagram which could represent an electronic circuit, schematic, or embedded system component from topics pertaining to {focus_context}, with detailed descriptions shown in the visual elements. Your task is to generate an extremely detailed explanation of the components, their relationships, and their function in the context of the image. 

  Focus on identifying key electronic components, understanding their role in the system, and how they interact within the embedded system environment.

  Below is the extracted text from OCR scanning, which could include specific labels, component names, and values but also could have gibberish from faulty OCR scans as we are scanning very complex techincal images that could be misinterpreted:

  Extracted OCR Text:
  {ocr_text}

  Below is the AI-generated description of the image, which highlights visual elements, labels, and possible relationships between the components:

  Image Description:
  {image_description}

  Based on both the OCR text and image description, provide a detailed explanation covering the following aspects:
  1. Identify key electronic components (e.g., resistors, capacitors, transistors, microcontrollers) and their function within the system.
  2. Describe any important relationships or connections between these components (e.g., signal flow, power distribution, input/output handling).
  3. Explain the significance of any numeric values (e.g., voltage, current, resistance) extracted from the OCR, and how they affect the operation of the system.
  4. Highlight any special components or subsystems, such as sensors, actuators, communication buses (I2C, SPI, UART), or control units, and explain their role.
  5. If applicable, describe the system’s overall function or purpose based on the extracted data.

  Your explanation should be highly detailed, focusing on technical accuracy, and should be suitable for someone with a deep understanding of {focus_context}.
  """
  return prompt


'''
  Desc: Run The LLaMA Model With The Generated Prompt To Get A Detailed Technical Explanation.

  Preconditions:
    1.) prompt Is A String Containing The LLaMA Input Prompt.
  
  Postconditions:
    1.) Returns The Response From LLaMA's Analysis.
'''
def run_llama_analysis(prompt):
  llama_res = ollama.chat(
    model="llama3.1:8b",  # Using the pulled LLaMA model
    messages=[
      {
        "role": "user",
        "content": prompt,
        "temperature": 0.35  # Slightly higher temperature for more creative technical explanations
      }
    ]
  )
  return llama_res["message"]["content"]


'''

  Desc: Extracts Text, Images, And Tables From A PDF Using PyMuPDF.

  Preconditions:
    1.) file_path Must Point To A Valid PDF File.
    2.) logger Is An Instance Of ProgressLogger For Managing Logs.
  
  Postconditions:
    1.) Returns A List Of Pages With Extracted Text, Images, Metadata, And Tables From The PDF.

'''
def extract_content_with_mupdf(file_path: str, logger: ProgressLogger) -> str:
  logger.log(f"Extracting content from PDF: {file_path}")

  # Open The PDF Using PyMuPDF
  doc = fitz.open(file_path)
  pages = []

  # Iterate Through Each Page In The PDF
  for page_num, page in enumerate(doc):
    text = page.get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES)
    image_list = page.get_images(full=True)
    images = []
    realPageNum = page_num + 1

    # Extract And Process Each Image In The Page In-Memory
    for img_index, img in enumerate(image_list):
      xref = img[0]
      base_image = doc.extract_image(xref)
      image_bytes = base_image["image"]
      image = Image.open(io.BytesIO(image_bytes))

      try:
        # Perform OCR on the image
        logger.log(f"Performing OCR on image from page {realPageNum}...")
        preprocessed_image = preprocess_image(image)
        ocr_text = perform_ocr(preprocessed_image)
        logger.write_to_debug_file(f"OCR Description Of Image From Page {realPageNum}:\n {ocr_text}")
      except Exception as e:
        logger.log(f"Failed to perform OCR on image type, invalid channel count for preprocessing :/", "ERROR")
      
      # Perform LLaVA Image Analysis
      logger.log(f"Using LLaVA to analyze image from page {realPageNum}...")
      llava_description = generate_image_description(image)
      logger.write_to_debug_file(f"LLaVA Description Of Image From Page {realPageNum}:\n {llava_description}")

      # Create Prompt for LLaMA
      prompt = create_finetuning_prompt(ocr_text, llava_description)

      # Run LLaMA Analysis
      logger.log(f"Running LLaMA analysis for page {realPageNum}...")
      llama_output = run_llama_analysis(prompt)
      logger.write_to_debug_file(f"LLaMA Description Of Image From Page {realPageNum}:\n {llama_output}")

      # Each Image Will Be Associated With A Summation Of Its Contents From 2-Means Of Analysis
      images.append({
        # Could Be Faulty So Don't Pass It "OCR Of Image": ocr_text,
        # Could Be Faulty So Don't Pass It "LLaVA Description of Image": llava_description,
        "Summation Of LLaVA & OCR Text Contents": llama_output
      })

    # Extract Table Data From The Page
    logger.log(f"Attempting table extraction with Tabula on page {realPageNum} of {file_path}...")
    table_data = extract_table_data_with_tabula_multiprocessing(file_path, realPageNum)
    if table_data != "":
      logger.log(f"Successfully extracted table data.")
    else:
      logger.log(f"Failed extracting table data.", level="ERROR")

    # Append The Extracted Data To The List Of Pages
    if text or images or table_data:
      pages.append({
        "page_num": page_num,
        "text": text + "  " + table_data,
        "images": images,
        "metadata": doc.metadata,
        "annotations": list(page.annots()) if page.annots() else [],
        "links": page.get_links()
      })

      logger.write_to_debug_file(f"Parsed raw text on page {realPageNum} of {file_path}: \n{(pages[page_num]['text'])}")

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

  Desc: Sends A Text Chunk And Context To OpenAI GPT-4o-Mini For Processing And Fine-Tuning Data Generation In Which Focuses On Technically-Worded, Semantical/Code Solutions. Will Be Utilized For Mult-Domain Contexual Chaining Later With send_to_openai(...).
  
  Preconditions:
    1.) text_chunk Contains The Text To Be Sent To The OpenAI API As Fine-Tuning Data.
    2.) previous_context Is The Context From Previous Pages Or Chunks Of Fine-Tuning Data Being Sent.
    3.) file_path And page_num Are For Debugging And Logging.
  
  Postconditions:
    1.) Returns The Training Data Response Content From The OpenAI Model.
    2.) Returns None If An Error Occurs During The API Call.

'''
def send_to_openai_code(text_chunk, previous_context="", file_path="", page_num=""):
  try:
    # Streamlined Prompt For Fine-tuning Data Generation
    prompt = f"""
    Using the primary source documentation in TEXT CONTENT from page {page_num} of {file_path}, and the previous two pages in PRIOR CONTEXT, generate **extremely detailed and high-quality technical analysis** focused in the {focus_context} sphere of computer science. The response must integrate the information from the documentation, covering the full lifecycle and deployment phases, and fully utilizing all relevant data, including both quantitative (e.g., performance metrics, power consumption, memory sizes, clock speeds) and qualitative information (e.g., descriptive insights, configuration details).

    Ensure the response thoroughly addresses the following:
    1. **Core Concepts**: Provide a deep, interconnected explanation of key concepts in real-time performance, including task scheduling, memory management, and hardware-software integration, and their relevance to {focus_context}. **Incorporate specific numbers and key descriptive insights** from the documentation (e.g., performance metrics, system constraints).
    2. **Code Examples**: Provide one or two highly detailed code samples (in C, assembly, etc.) relevant to real-time optimizations, scheduling, memory handling, and resource management. Each code sample must be extensively explained line by line, with a narrative that links each part of the code to specific data points (e.g., performance metrics, memory access patterns) and other key descriptive information from the documentation.
    3. **System Integration**: Explain in depth how system components interact within an embedded system, and describe the impact of configurations (e.g., sensors, communication buses) on system performance. Use both **quantitative data (e.g., memory sizes, clock speeds, power consumption)** and **qualitative insights** from the documentation to support the explanation, especially regarding constraints like power and temperature.
    4. **Optimization Strategies**: Present specific optimization techniques, including task scheduling (e.g., round-robin, priority-based), interrupt handling, and memory management (stack vs. heap). Focus on **detailed performance trade-offs**, making sure to reference **any relevant data (e.g., memory bandwidth, power usage, clock speeds)** and **qualitative descriptions** in the documentation. Provide at least one alternative approach and explain the comparative trade-offs.
    5. **Real-World Application with Data**: Provide a **detailed, real-world example** incorporating **quantitative data (e.g., benchmarks, performance metrics, power consumption)** and qualitative insights from the documentation. Discuss practical case studies (e.g., power efficiency, task parallelism, or latency reduction) and explain the significance of the numbers and textual data for the optimization strategy.
    6. **Environmental Applicability**: Provide detailed explanations of the environments in which this technology is able to be deployed in (e.g. this sensor is not waterproof so don't use in hydroengineering)
    
    TEXT CONTENT:
    <info>{text_chunk}</info>

    PRIOR CONTEXT:
    <info>{previous_context}</info>

    Format the response as a conversation, ensuring a strong technical question and answer derived from the TEXT CONTENT and PRIOR CONTEXT for each topic. Focus on producing detailed responses. Each response should include deep technical insights, with extensive explanations that thoroughly connect back to the provided documentation, utilizing both **quantitative data** (e.g., numbers, performance metrics) and **qualitative descriptions**.

    ENSURE ALL DATA PROVIDED IN TEXT CONTENT and PRIOR CONTEXT IS UTILIZED IN THE RESPONSES; FORMAT YOUR RESPONSE STRICTLY IN THIS MANNER (ENSURE QUALITY OVER QUANTITY IN AMOUNT OF RESPONSES TRY MAKING AS MANY "messages" ENTRIES AS POSSIBLE IN THIS FORMAT):

    {{
      "messages": [
        {{"role": "system", "content": "You are an expert in {focus_context}, specializing in real-time optimizations, embedded systems, and performance tuning. Your task is to provide **extremely detailed** technical insights, focusing on solving complex problems based on the provided documentation. Prioritize depth and interconnect concepts from the provided documentation, and fully utilize both numbers (performance metrics, clock speeds, memory sizes, etc.) and qualitative descriptions."}},
        {{"role": "user", "content": "(A specific technical question or problem based on the documentation in TEXT CONTENT and PRIOR CONTEXT)"}} ,
        {{"content": "(A detailed, problem-solving response primarily written in code that incorporates relevant numbers and qualitative insights from the PDF in TEXT CONTENT and PRIOR CONTEXT. The solution should be **extensively explained**, presented primarily in code, with detailed comments that connect how the code addresses the problem and relates to the provided data. Provide alternative approaches where applicable, explaining trade-offs with reference to both quantitative data (e.g., numbers, benchmarks) and qualitative descriptions.)"}}
      ]
    }}
    {{ 
      "messages": [
        {{"role": "system", "content": "You are an expert in {focus_context}, specializing in real-time optimizations, embedded systems, and performance tuning. Your task is to provide **extremely detailed** technical insights, focusing on solving complex problems based on the provided documentation. Prioritize depth and interconnect concepts from the provided documentation, and fully utilize both numbers (performance metrics, clock speeds, memory sizes, etc.) and qualitative descriptions."}},
        {{"role": "user", "content": "(A specific technical question or problem based on the documentation in TEXT CONTENT and PRIOR CONTEXT)"}} ,
        {{"content": "(A detailed, problem-solving response primarily written in code that incorporates relevant numbers and qualitative insights from the PDF in TEXT CONTENT and PRIOR CONTEXT. The solution should be **extensively explained**, presented primarily in code, with detailed comments that connect how the code addresses the problem and relates to the provided data. Provide alternative approaches where applicable, explaining trade-offs with reference to both quantitative data (e.g., numbers, benchmarks) and qualitative descriptions.)"}}
      ]
    }}
    """



    # System-level Prompt To Define Model Behavior
    system_prompt = f"""
    You are an expert computer science with a current consultant focus in {focus_context}, specializing in embedded systems and low-lowel design/development. You provide comprehensive technical explanations, code samples, and practical insights based on the provided primary source documentation, addressing every stage of embedded system development, deployment, and optimization. Cross-reference the benchmarks, power consumption figures, and speedup statistics from the documentation when explaining optimization strategies.
    """

    # Call Openai Api With Adjusted Prompt
    response = openai.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        max_tokens=5000, # Adjusted For More Concise Responses
        temperature=0.25,
        top_p=0.75,
        presence_penalty=0.1,
        frequency_penalty=0.1
    )

    return response.choices[0].message.content

  except Exception as e:
    print(f"Error generating fine-tuned data: {e}")
    return None


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

  Desc: Sends A Text Chunk And Context To OpenAI GPT-4o-mini For Processing And Fine-Tuning Data Generation In Which Focuses On Technically-Worded, Theoretical Solutions. Will Be Utilized For Mult-Domain Contexual Chaining Later With send_to_openai_code(...).
  
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
    # Streamlined Prompt For Generating Fine-tuned Technical Analysis
    prompt = f"""
    Using the primary source documentation in TEXT CONTENT from page {page_num} of {file_path}, and the previous two pages in PRIOR CONTEXT, generate **extremely detailed and high-quality technical analysis** focused in the {focus_context} sphere of computer science. The response should integrate the information from the documentation, covering system behavior, lifecycle performance, and optimization techniques. Ensure you fully utilize both **quantitative data** (e.g., performance metrics, memory sizes, power consumption, clock speeds) and **qualitative insights** (e.g., configuration details, architectural design choices) ().

    Ensure the response thoroughly addresses the following:
    1. **Core Concepts**: Provide a comprehensive explanation of core concepts such as task scheduling, memory management, and hardware-software integration. Incorporate **specific numbers** and **qualitative insights** from the documentation (e.g., performance benchmarks, system constraints) to show how these concepts interrelate and impact performance.
    2. **System-Level Integration**: Explain in detail how system components (e.g., processors, memory, I/O interfaces) interact in an embedded system and the impact of configurations (e.g., sensors, communication buses) on system performance. Use both **quantitative data** (e.g., power consumption, clock speeds, memory sizes) and **qualitative descriptions** to support the analysis.
    3. **Optimization Strategies**: Analyze specific optimization techniques, such as task scheduling (e.g., round-robin vs. priority-based), memory management (e.g., stack vs. heap allocation), and hardware optimizations (e.g., cache utilization). Include detailed performance trade-offs and reference relevant benchmarks or statistics from the documentation.
    4. **Real-World Application**: Provide a **detailed example** of a real-world application incorporating both **quantitative data (e.g., performance metrics, power usage, benchmarks)** and **qualitative insights** from the documentation. Discuss practical cases such as power efficiency, task parallelism, or latency reduction, and explain the significance of these metrics for real-time system optimization.
    5. **Environmental Applicability**: Provide detailed explanations of the environments in which this technology is able to be deployed in (e.g. this sensor is not waterproof so don't use in hydroengineering)

    TEXT CONTENT:
    <info>{text_chunk}</info>

    PRIOR CONTEXT:
    <info>{previous_context}</info>

    Format the response as a conversation, ensuring a strong technical question and answer derived from the TEXT CONTENT and PRIOR CONTEXT for each topic. Focus on producing detailed responses. Each response should include **extensive technical insights**, connecting back to the provided documentation, utilizing both **quantitative data** (e.g., performance metrics) and **qualitative descriptions** (e.g., system design).

    ENSURE ALL DATA PROVIDED IN TEXT CONTENT and PRIOR CONTEXT IS UTILIZED IN THE RESPONSES; FORMAT YOUR RESPONSE STRICTLY IN THIS MANNER (ENSURE QUALITY OVER QUANTITY; TRY MAKING AS MANY QUALITY "messages" AS POSSIBLE):

    {{
      "messages": [
        {{"role": "system", "content": "You are an expert in {focus_context}, specializing in real-time optimizations, embedded systems, and performance tuning. Your task is to provide **extremely detailed** technical insights, focusing on solving complex problems based on the provided documentation. Prioritize depth and interconnect concepts from the provided documentation, and fully utilize both numbers (performance metrics, clock speeds, memory sizes, etc.) and qualitative descriptions."}},
        {{"role": "user", "content": "(A specific technical question or problem based on the documentation in TEXT CONTENT and PRIOR CONTEXT)"}} ,
        {{"content": "(A detailed, problem-solving response providing a technical analysis incorporating **quantitative data** and **qualitative insights**. The solution should be extensively explained, referencing both data points and descriptive information from the PDF. Offer alternative approaches where applicable and explain trade-offs using **quantitative benchmarks** and **qualitative insights**.)"}}
      ]
    }}
    {{
      "messages": [
        {{"role": "system", "content": "You are an expert in {focus_context}, specializing in real-time optimizations, embedded systems, and performance tuning. Your task is to provide **extremely detailed** technical insights, focusing on solving complex problems based on the provided documentation. Prioritize depth and interconnect concepts from the provided documentation, and fully utilize both numbers (performance metrics, clock speeds, memory sizes, etc.) and qualitative descriptions."}},
        {{"role": "user", "content": "(A specific technical question or problem based on the documentation in TEXT CONTENT and PRIOR CONTEXT)"}} ,
        {{"content": "(A detailed, problem-solving response providing a technical analysis incorporating **quantitative data** and **qualitative insights**. The solution should be extensively explained, referencing both data points and descriptive information from the PDF. Offer alternative approaches where applicable and explain trade-offs using **quantitative benchmarks** and **qualitative insights**.)"}}
      ]
    }}
    """

    # System-level Prompt For Defining Model Behavior In Technical Analysis
    system_prompt = f"""
    You are an expert computer science with a current consultant focus in {focus_context}, specializing in embedded systems and low-lowel design/development. You provide **extremely detailed** technical insights, drawing extensively from the provided documentation, focusing on solving complex problems in areas such as task scheduling, memory management, and hardware-software integration. You must integrate both **quantitative data** (e.g., performance metrics, power consumption, memory sizes) and **qualitative descriptions** (e.g., architectural design, configuration details) from the documentation in every response.

    Your responses should cover all aspects of {focus_context}, including foundational theory, system integration, real-world application, and optimization strategies. Ensure that your answers reflect the information provided in the PDF, cross-referencing benchmarks, power consumption figures, and technical insights wherever possible.
    """

    # Call OpenAI API with the adjusted prompt
    response = openai.chat.completions.create(
      model="gpt-4o-mini-2024-07-18",
      messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
      ],
      max_tokens=4500,  # Adjust For Detailed Outputs
      temperature=0.25,  # Keep The Temperature Low To Maintain Deterministic, Detailed Responses
      top_p=0.75,  # Ensures Focused And Technical Answers
      presence_penalty=0.1,  # Encourage New Concept Introductions
      frequency_penalty=0.1  # Reduces Redundancy In The Answers
    )

    # Return The Detailed Response From OpenAI
    return response.choices[0].message.content
  except Exception as e:
    print(f"Error generating fine-tuned data: {e}")
    return None


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
    progress_logger.log(f"No data extracted from {file_path}. No files written.", level="ERROR")


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
    # Use GPT-4o-Mini To Summarize The Extracted Text
    response = openai.chat.completions.create(
      model="gpt-4o-mini-2024-07-18",
      messages=[
        {"role": "system", "content": "You are an expert in summarizing technical documentation in a detailed, thorough manner upholding as much detail as possible."},
        {"role": "user", "content": f"Summarize the following text from page {page_num} of {file_path}: {text}"}
      ],
      max_tokens=1000,
      temperature=0.45
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
          progress_logger.log(f"Error processing PDF: {e}", level="ERROR")

  # End Of Logging
  print("-----------------------------------------End Of Logging--------------------------------------------------\n", flush=True)



# Main Driver (Program Starts Here, Calling multiThreaded_process_directory(...) With A Valid File Containing Training PDFs)
if __name__ == "__main__":
  multiThreaded_process_directory(directory_path)
