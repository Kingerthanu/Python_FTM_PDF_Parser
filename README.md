# Python_FTM_PDF_Parser
Python Script In Which Tries To Accelerate The PDF Parsing Stage Of Many openai Fine-Tuning Models. 

This Is From The Multi-Threaded Approach Of Scanning Specific PDF Documents, Allowing Many Worker Threads To Talk To openai About Their Own Document. This Can Allow The Process To Be Scanning Multiple Documents In Paralell In Which Can Greatly Improve Runtime. To Improve Reliability While Keeping Token Usage To A Lower Amount We Employed Chunking Of Pages Contents To Allow Us To Minimize Generalizations Of The Page Contents During Training Data Generation. The openai Request Will Be Provided A Summary Of The Last 2 Chunks/Requests To Help Them Correlate Context Between These Chunks. This Summation Will Be Generated Again By An openai Request To Generalize The Provided Chunk Context. Data Analysis Currently Is Being Done By PyMuPDF And Tabula Mainly. Tabula Is Utilized For Data Tables Embedded In A PDF In Which Is Good To Help With Properly Giving openai A Structured Way Of Knowing What A Table Is Saying. 

One of the major updates in this version is the integration of local large language models (LLMs) via ollama, specifically LLaMA (for textual analysis) and LLaVA (for image-based reasoning). LLaVA processes images embedded in PDFs, providing detailed descriptions of diagrams, schematics, and other visuals. LLaMA, on the other hand, handles technical explanations, generating in-depth textual insights from the parsed PDF content. This local processing reduces reliance on cloud-based services, minimizing API costs and improving data security by processing sensitive documents on-device.

This branch also introduces OCR functionality using pytesseract for cases where text is embedded in images, such as scanned documents or diagrams. Images are preprocessed with OpenCV to improve OCR accuracy by applying sharpening filters and resizing, ensuring that even challenging image-based text can be effectively extracted.

Refined prompt engineering strategies now ensure highly technical and context-aware responses from both OpenAI’s GPT models and local LLaMA instances. When splitting document content into chunks, the system preserves context across pages to generate more coherent fine-tuning data. In addition, prompt structures are tailored to elicit detailed technical insights, with a focus on real-time performance, hardware/software co-design, and system optimization.

Looking forward, the system aims to incorporate multi-domain contextual chaining, allowing insights from various PDFs and sources to be linked together. This will enable richer, more comprehensive datasets by cross-referencing related content across multiple documents. The goal is to enable specialized fine-tuning in domains such as real-time system optimization and embedded systems design, with integrated multimodal reasoning across text, tables, and images.

**Also BTW The Debug Messages Will Stay And Keep Adding Up So Make Sure It Doesn't Take Up Too Much Of Your Storage As I Kept It Quite Thorough As A Lot Is Said In The Debug File That Isn't Said Through The Terminal.**

----------------------------------------------

<img src="https://github.com/user-attachments/assets/b6920c17-2da4-4c3f-8efc-b94893955e04" alt="Cornstarch <3" width="55" height="49"> <img src="https://github.com/user-attachments/assets/b6920c17-2da4-4c3f-8efc-b94893955e04" alt="Cornstarch <3" width="55" height="49"> <img src="https://github.com/user-attachments/assets/b6920c17-2da4-4c3f-8efc-b94893955e04" alt="Cornstarch <3" width="55" height="49"> <img src="https://github.com/user-attachments/assets/b6920c17-2da4-4c3f-8efc-b94893955e04" alt="Cornstarch <3" width="55" height="49">


### **The Breakdown:**

  #### **- Config Beforehand**
  The Program Starts By Initially Needing 4 Things To Be Done. 

  1.) Initially Garner Some PDF Documents In Which You Want To Use To Use As Primary Source Material To Fine-Tune A Model With.
  
  2.) Establish An API Key In Which You Are Using To Communicate With _openai_ With.

  3.) Provide The Specific Context In Which Your Training Data Should Try To Emphasize.

  You Will Paste Them Here In The Script. Where `open.api_key` Is The _openai_ API Key Being Used And `directory_path` Is The Folder Containing Your PDFs To Train With. As Well As `focus_context` Being The Specific Context To Focus In Our Training Data.

```

19  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Config~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
20  
21  # Initialize The OpenAI Client With Your Specific API Key
22  openai.api_key = "________"
23  
24  # Specify The Directory Path Of Your PDFs
25  directory_path = r"______"
26  
27  # Specify The Focus Context To Utilize The Training Data In
28  focus_context = "Real-Time Performance in RISC-V"
29  
30  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~END Config~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

```

  3.) Ensure You Have All Required Libraries (I Tried Making It Nice So You Can pip Everything In):
  ```

  # For OpenAI API Client To Send And Receive Fine-Tuning Data.
  pip install --upgrade openai
  
  # For PDF Reading And Text/Image Extraction Using PyMuPDF.
  pip install --upgrade pymupdf
  
  # For Handling Image Manipulation.
  pip install --upgrade Pillow
  
  # For Extracting Tables From PDFs Using Tabula.
  pip install --upgrade tabula-py
  
  # For Displaying Progress Bars And Live Terminal Updates.
  pip install --upgrade rich
  
  # For Local LLaMA And LLaVA Model Support.
  pip install --upgrade ollama
  
  # For OCR Support (Requires Tesseract Installed Separately).
  pip install --upgrade pytesseract
  
  # For Image Processing And Preprocessing For OCR.
  pip install --upgrade opencv-python
  
  # For Matrix-Based Image Handling And Processing.
  pip install --upgrade numpy

  ```

  You Will Also Have To Download The LLM Models Locally From Ollama By Doing:
  ```

    ollama pull llama3.1:8b
    ollama pull llava:latest

  ```

  <h4>Important:</h4>
  
  For OCR to work, you must also install the Tesseract OCR engine on your system. Follow the instructions based on your operating system:
  
  Windows: Download and install Tesseract from <a href="https://tesseract-ocr.github.io/tessdoc/Installation.html" target="_blank">here</a>. You May Need To Include It In Your SYSPATH If It Doesn't Recognize It During Runtime.
  
  macOS: Install using Homebrew:
  ```
  brew install tesseract
  ```

  Linux: Install Tesseract using the package manager for your distribution:
  ```
  sudo apt install tesseract-ocr
  ```


  You Also May Need To Install Java For Tabula As This Codebase Relies On It.


#### **- Notice**
This branch uses local LLMs (Large Language Models), which require substantial system resources. For smooth operation, please ensure the following:

- CUDA Support: If your system has a CUDA-supported GPU, ensure it is configured for accelerated processing. Verify by running nvcc --version in your terminal. If CUDA is unavailable, the models will fall back to CPU execution, though this will be significantly slower.
  
- Memory Requirements: Make sure your system has enough RAM. Running these models locally can require 16GB or more system memory, and a GPU with at least 8GB of VRAM is recommended.
  
- Disk Space: Local LLMs like LLaVA and LLaMA can take up several gigabytes of storage. Ensure you have sufficient disk space available to store these models.
  
- Cooling: Running LLMs generates heat, especially on extended tasks. Ensure your system has adequate cooling to prevent overheating.
  
- Model Versions: This branch supports specific versions of LLaVA and LLaMA. Ensure you are using compatible versions to avoid compatibility issues.

  
#### **- Local LLM Support**
This Branch Introduces Support For Two Local Large Language Models (LLMs) Via Ollama, Allowing For High-Performance, On-Device AI Processing Without The Need For Cloud-Based Services. The Two Integrated Models Are:

<img src="https://github.com/user-attachments/assets/5b7de0ce-5f9b-4922-973c-a702bc92c695" alt="Cornstarch <3" width="25" height="25"> LLaVA (Large Language And Vision Assistant):
LLaVA Is Designed To Process And Understand Visual Data By Providing Detailed, Context-Rich Explanations Of Images. It Is Specifically Tuned For Multi-Modal Tasks, Meaning It Excels At Tasks Requiring Both Visual And Text-Based Understanding. In The Context Of This Pipeline, LLaVA Replaces BLIP-2 As The Primary Model Used For Generating Detailed Descriptions Of Images Embedded Within PDFs. The Move To LLaVA Significantly Enhances The Quality Of Image Interpretation, As The Model Is Capable Of Giving Far More Granular, Technical Descriptions Of Visual Elements Like Schematics, Diagrams, And Charts. This Is Particularly Useful When Fine-Tuning Models On Multi-Modal Datasets Where Image Understanding Is Crucial.

The Integration With LLaVA Allows The System To Process Images In PDFs Locally, Cutting Down On Cloud-Based API Calls, Which Results In Significant Token Usage Savings And Improved Data Security. LLaVA Also Supports Temperature-Controlled Responses, Ensuring More Deterministic Outputs Based On User Requirements, Making It Flexible For Different Use Cases.

<br> <img src="https://github.com/user-attachments/assets/39175c73-b5cd-4583-ad2d-7809c0be13ba" alt="Cornstarch <3" width="25" height="25"> **LLaMA (Large Language Model Meta AI)**: LLaMA, Developed By Meta AI, Is A State-Of-The-Art Transformer-Based LLM Designed For Generating Human-Like Text. In This Updated Pipeline, LLaMA Is Used For Generating Extremely Detailed Textual Descriptions, Technical Explanations, And Contextual Insights From Extracted PDF Content. LLaMA’s Performance Rivals That Of OpenAI’s GPT Models But Operates Fully On-Device, Providing Cost-Effective And Efficient Processing.
This Dual Integration With LLaVA And LLaMA Allows For Complex Image-To-Text And Text-Based Reasoning Tasks To Be Performed Locally, Minimizing The Reliance On OpenAI’s Cloud Services. It Enables The System To Generate Rich, High-Quality Explanations While Reducing Token Usage And Associated Costs. Additionally, The Local Nature Of The Models Allows For More Privacy-Preserving Workflows, Where Sensitive Data Need Not Be Transmitted Over The Internet.

#### **- OCR Support**
The System Now Includes Optical Character Recognition (OCR) Functionality, Implemented Through Pytesseract. OCR Is Used To Extract Text From Images In PDFs Where The Text Is Embedded As Part Of An Image, Such As Scanned Documents Or Graphical Content. The OCR Support Is Particularly Helpful When Dealing With Documents That Contain Text In Non-Textual Formats, Such As Scanned Images Or Screenshots.

However, OCR In This Version Of The Codebase Is Noted To Have Some Limitations And May Produce Inconsistent Results For Highly Complex Images, Especially In The Technical PDF Domain. The Text Extraction Accuracy Is Influenced By Image Quality And Formatting Irregularities. Currently, Pytesseract Handles Most Of The OCR Tasks, But It Is Configured To Preprocess Images By Resizing And Applying Sharpening Filters Through OpenCV To Enhance Recognition Accuracy.

The OCR Pipeline Is Scheduled For Future Enhancements, With Potential Integration Of More Robust Solutions Like LLaVA Or Additional Machine Learning-Based OCR Tools For Better Accuracy And Consistency In Complex Images.

#### **- Prompt Engineering Optimizations**
One Of The Significant Upgrades In This Branch Is The Refined Prompt Engineering Strategy, Particularly For Communicating With OpenAI's GPT Models And The Local LLaMA Model. Key Improvements Include:

Contextualized Prompts For Text Chunking: The Pipeline Now Includes Advanced Context Preservation Between Consecutive Pages Of A Document. When Splitting Text Into Smaller, Manageable Chunks (To Comply With API Token Limits), The System Intelligently Preserves Context From Previous Pages Or Sections. This Is Especially Important In Maintaining Coherence When Generating Fine-Tuning Datasets, As It Ensures That Each Chunk Is Processed With An Understanding Of The Overall Document Structure.

Optimized Prompts For Fine-Tuning: The Prompts For Sending Text Chunks To OpenAI Or LLaMA Are Crafted To Elicit Highly Technical And Detailed Responses. These Prompts Are Structured To Break Down The Integration Of Hardware And Software, Particularly In Real-Time Systems, While Asking For Code Samples, Performance Benchmarks, And Practical Optimizations. The Refined Prompt Structures Ensure Responses Remain Highly Relevant And In-Depth, Directly Addressing Key Topics With A Focus On Maximizing Actionable Insights.

Retry Logic For API Stability: The Prompt Engineering Process Is Bolstered By A Robust Retry Mechanism, Ensuring That Intermittent Failures During Communication With APIs (Such As OpenAI's GPT) Do Not Disrupt The Workflow. The System Will Retry Failed Requests, Ensuring Reliability In Generating Fine-Tuning Data And Reducing The Risk Of Data Loss During API Calls.

#### **- Looking Forward**
The Future Development Roadmap For The Parser Pipeline Focuses On Further Integrating Multi-Domain Contextual Chaining. This Would Allow The System To Create More Comprehensive, Interlinked Datasets By Chaining Contextual Information Across Different Types Of Data Sources (E.G., PDFs, Images, Tables, And Structured Data).

Multi-Domain Contextual Chaining Will Enable The System To:

Maintain Cross-Document Context: By Linking Insights And Information From Multiple PDFs And Documents, The System Can Maintain A Broader Understanding Across Different Sources. For Instance, If Multiple Technical Manuals Or Whitepapers Are Being Fine-Tuned On, The System Will Be Able To Cross-Reference Related Concepts Across These Documents, Creating A Richer Training Dataset.

Enhanced Multimodal Understanding: The Next Step Is To Refine How Text, Tables, And Images Are Processed Together. By Integrating Image Captions From LLaVA, Text-Based Reasoning From LLaMA, And Table Data From Tabula, The System Aims To Produce Datasets That Are More Interconnected, Mimicking Real-World Complex Data Interactions.

Advanced Fine-Tuning Mechanisms: Another Goal Is To Use Multi-Domain Context Chaining For More Specialized Fine-Tuning, Especially In Areas Like Real-Time Performance Optimization, Hardware/Software Co-Design, And Embedded Systems. The Vision Is To Use This Framework Not Just For Text-Based Fine-Tuning, But Also For Developing Models That Can Reason Across Modalities—Effectively Creating Models That Understand Both Technical Language And Technical Visuals.

<img src="https://github.com/user-attachments/assets/1fc90166-a62b-4d91-a087-5da5a3a7076f" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/1fc90166-a62b-4d91-a087-5da5a3a7076f" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/1fc90166-a62b-4d91-a087-5da5a3a7076f" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/1fc90166-a62b-4d91-a087-5da5a3a7076f" alt="Cornstarch <3" width="65" height="59">

----------------------------------------------

<img src="https://github.com/user-attachments/assets/2c82de29-71b0-433a-bc10-3778026b74a7" alt="Cornstarch <3" width="65" height="59">  <img src="https://github.com/user-attachments/assets/2c82de29-71b0-433a-bc10-3778026b74a7" alt="Cornstarch <3" width="65" height="59">  <img src="https://github.com/user-attachments/assets/2c82de29-71b0-433a-bc10-3778026b74a7" alt="Cornstarch <3" width="65" height="59">  <img src="https://github.com/user-attachments/assets/2c82de29-71b0-433a-bc10-3778026b74a7" alt="Cornstarch <3" width="65" height="59"> 


### **Features:**

  **Script Parsing And Training PDF Data:**
  
  ![2024-09-1300-32-48-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/d6577304-86de-41bd-a678-537d10918f6a)


 **Final Training Data Example:**

 <img width="1768" alt="image" src="https://github.com/user-attachments/assets/61479c16-fa29-4fc9-94c6-fbf885a73a4e">

  **Screenshot Of Console UI (With A Bit Of Sharpie:**

  <img width="1847" alt="image" src="https://github.com/user-attachments/assets/9552d56e-3dcf-41f5-bef2-acc4c51b8105">


<img src="https://github.com/user-attachments/assets/b568a6eb-aa1d-4379-80a8-938dd51fa068" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/b568a6eb-aa1d-4379-80a8-938dd51fa068" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/b568a6eb-aa1d-4379-80a8-938dd51fa068" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/b568a6eb-aa1d-4379-80a8-938dd51fa068" alt="Cornstarch <3" width="65" height="59">
