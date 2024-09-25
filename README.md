# Python_FTM_PDF_Parser
Python Script In Which Tries To Accelerate The PDF Parsing Stage Of Many openai Fine-Tuning Models. 

This Is From The Multi-Threaded Approach Of Scanning Specific PDF Documents, Allowing Many Worker Threads To Talk To openai About Their Own Document. This Can Allow The Process To Be Scanning Multiple Documents In Paralell In Which Can Greatly Improve Runtime. To Improve Reliability While Keeping Token Usage To A Lower Amount We Employed Chunking Of Pages Contents To Allow Us To Minimize Generalizations Of The Page Contents During Training Data Generation. The openai Request Will Be Provided A Summary Of The Last 2 Chunks/Requests To Help Them Correlate Context Between These Chunks. This Summation Will Be Generated Again By An openai Request To Generalize The Provided Chunk Context. Data Analysis Currently Is Being Done By PyMuPDF And Tabula Mainly. Tabula Is Utilized For Data Tables Embedded In A PDF In Which Is Good To Help With Properly Giving openai A Structured Way Of Knowing What A Table Is Saying. 

This Has Worked Really Well But Still Seems To Have Some Flaws With Formatting Sometimes So Will Need To Look Into It In The Future. OCR And Blip Were Being Attempted But Never Gave Consistent Results With The Warzone That PDFs Are So Are Currently Deprecated (Still Have Blip Saying Something But Is Really Useless, OCR Needed Gaussian Blur And Never Gave Consistent Results).

**Also BTW The Debug Messages Will Stay And Keep Adding Up So Make Sure It Doesn't Take Up Too Much Of Your Storage As I Kept It Quite Thorough As A Lot Is Said In The Debug File That Isn't Said Through The Terminal.**

----------------------------------------------

<img src="https://github.com/user-attachments/assets/b6920c17-2da4-4c3f-8efc-b94893955e04" alt="Cornstarch <3" width="55" height="49"> <img src="https://github.com/user-attachments/assets/b6920c17-2da4-4c3f-8efc-b94893955e04" alt="Cornstarch <3" width="55" height="49"> <img src="https://github.com/user-attachments/assets/b6920c17-2da4-4c3f-8efc-b94893955e04" alt="Cornstarch <3" width="55" height="49"> <img src="https://github.com/user-attachments/assets/b6920c17-2da4-4c3f-8efc-b94893955e04" alt="Cornstarch <3" width="55" height="49">


### **The Breakdown:**

  #### **- Config Beforehand**
  The Program Starts By Initially Needing 3 Things To Be Done. 

  1.) Initially Garner Some PDF Documents In Which You Want To Use To Use As Primary Source Material To Fine-Tune A Model With.
  
  2.) Establish An API Key In Which You Are Using To Communicate With _openai_ With.

  You Will Paste Them Here In The Script. Where `open.api_key` Is The _openai_ API Key Being Used And `directory_path` Is The Folder Containing Your PDFs To Train With.

```

33  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Config~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
34  
35  
36  # Initialize The OpenAI Client With Your Specific API Key
37  openai.api_key = "________"
38  
39  # Specify The Directory Path Of Your PDFs
40  directory_path = r"______"
41  
42  
43  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~END Config~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


```

  3.) Ensure You Have All Required Libraries (I Tried Making It Nice So You Can pip Everything In):
  ```

  # For OpenAI API Client To Send And Receive Fine-Tuning Data.
  pip install --upgrade openai
  
  # For PDF Reading And Text/Image Extraction Using PyMuPDF.
  pip install --upgrade pymupdf
  
  # For Handling Image Manipulation In The Code.
  pip install --upgrade Pillow
  
  # For Extracting Tables From PDFs Using Tabula.
  pip install --upgrade tabula-py
  
  # For Displaying Progress Bars And Live Terminal Updates.
  pip install --upgrade rich
  
  # For BLIP-2 And Other Machine Learning Tasks Using Torch.
  pip install --upgrade torch
  
  # For BLIP-2 Image Captioning And Model Processing.
  pip install --upgrade transformers

  ```

#### **- Local LLM Support**
This branch introduces support for two local large language models (LLMs) via ollama, allowing for high-performance, on-device AI processing without the need for cloud-based services. The two integrated models are:

<img src="https://github.com/user-attachments/assets/5b7de0ce-5f9b-4922-973c-a702bc92c695" alt="Cornstarch <3" width="25" height="25"> LLaVA (Large Language and Vision Assistant):
LLaVA is designed to process and understand visual data by providing detailed, context-rich explanations of images. It is specifically tuned for multi-modal tasks, meaning it excels at tasks requiring both visual and text-based understanding. In the context of this pipeline, LLaVA replaces BLIP-2 as the primary model used for generating detailed descriptions of images embedded within PDFs. The move to LLaVA significantly enhances the quality of image interpretation, as the model is capable of giving far more granular, technical descriptions of visual elements like schematics, diagrams, and charts. This is particularly useful when fine-tuning models on multi-modal datasets where image understanding is crucial.

The integration with LLaVA allows the system to process images in PDFs locally, cutting down on cloud-based API calls, which results in significant token usage savings and improved data security. LLaVA also supports temperature-controlled responses, ensuring more deterministic outputs based on user requirements, making it flexible for different use cases.

<br>
<img src="https://github.com/user-attachments/assets/39175c73-b5cd-4583-ad2d-7809c0be13ba" alt="Cornstarch <3" width="25" height="25"> LLaMA (Large Language Model Meta AI):
LLaMA, developed by Meta AI, is a state-of-the-art transformer-based LLM designed for generating human-like text. In this updated pipeline, LLaMA is used for generating extremely detailed textual descriptions, technical explanations, and contextual insights from extracted PDF content. LLaMA’s performance rivals that of OpenAI’s GPT models but operates fully on-device, providing cost-effective and efficient processing.

This dual integration with LLaVA and LLaMA allows for complex image-to-text and text-based reasoning tasks to be performed locally, minimizing the reliance on OpenAI’s cloud services. It enables the system to generate rich, high-quality explanations while reducing token usage and associated costs. Additionally, the local nature of the models allows for more privacy-preserving workflows, where sensitive data need not be transmitted over the internet.

#### **- OCR Support**
The system now includes Optical Character Recognition (OCR) functionality, implemented through pytesseract. OCR is used to extract text from images in PDFs where the text is embedded as part of an image, such as scanned documents or graphical content. The OCR support is particularly helpful when dealing with documents that contain text in non-textual formats, such as scanned images or screenshots.

However, OCR in this version of the codebase is noted to have some limitations and may produce inconsistent results for highly complex images, especially in the technical PDF domain. The text extraction accuracy is influenced by image quality and formatting irregularities. Currently, pytesseract handles most of the OCR tasks, but it is configured to preprocess images by resizing and applying sharpening filters through OpenCV to enhance recognition accuracy.

The OCR pipeline is scheduled for future enhancements, with potential integration of more robust solutions like LLaVA or additional machine learning-based OCR tools for better accuracy and consistency in complex images.

#### **- Prompt Engineering Optimizations**
One of the significant upgrades in this branch is the refined prompt engineering strategy, particularly for communicating with OpenAI's GPT models and the local LLaMA model. Key improvements include:

Contextualized Prompts for Text Chunking: The pipeline now includes advanced context preservation between consecutive pages of a document. When splitting text into smaller, manageable chunks (to comply with API token limits), the system intelligently preserves context from previous pages or sections. This is especially important in maintaining coherence when generating fine-tuning datasets, as it ensures that each chunk is processed with an understanding of the overall document structure.

Optimized Prompts for Fine-Tuning: The prompts for sending text chunks to OpenAI or LLaMA are crafted to elicit highly technical and detailed responses. These prompts are structured to break down the integration of hardware and software, particularly in real-time systems, while asking for code samples, performance benchmarks, and practical optimizations. The refined prompt structures ensure responses remain highly relevant and in-depth, directly addressing key topics with a focus on maximizing actionable insights.

Retry Logic for API Stability: The prompt engineering process is bolstered by a robust retry mechanism, ensuring that intermittent failures during communication with APIs (such as OpenAI's GPT) do not disrupt the workflow. The system will retry failed requests, ensuring reliability in generating fine-tuning data and reducing the risk of data loss during API calls.

#### **- Looking Forward**
The future development roadmap for the Python_FTM_PDF_Parser pipeline focuses on further integrating multi-domain contextual chaining. This would allow the system to create more comprehensive, interlinked datasets by chaining contextual information across different types of data sources (e.g., PDFs, images, tables, and structured data).

Multi-Domain Contextual Chaining will enable the system to:

Maintain Cross-Document Context: By linking insights and information from multiple PDFs and documents, the system can maintain a broader understanding across different sources. For instance, if multiple technical manuals or whitepapers are being fine-tuned on, the system will be able to cross-reference related concepts across these documents, creating a richer training dataset.

Enhanced Multimodal Understanding: The next step is to refine how text, tables, and images are processed together. By integrating image captions from LLaVA, text-based reasoning from LLaMA, and table data from Tabula, the system aims to produce datasets that are more interconnected, mimicking real-world complex data interactions.

Advanced Fine-Tuning Mechanisms: Another goal is to use multi-domain context chaining for more specialized fine-tuning, especially in areas like real-time performance optimization, hardware/software co-design, and embedded systems. The vision is to use this framework not just for text-based fine-tuning, but also for developing models that can reason across modalities—effectively creating models that understand both technical language and technical visuals.
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
