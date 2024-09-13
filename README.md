# Python_FTM_PDF_Parser
Python Script In Which Tries To Accelerate The PDF Parsing Stage Of Many openai Fine-Tuning Models. 

This Is From The Multi-Threaded Approach Of Scanning Specific PDF Documents, Allowing Many Worker Threads To Talk To openai About Their Own Document. This Can Allow The Process To Be Scanning Multiple Documents In Paralell In Which Can Greatly Improve Runtime. To Improve Reliability While Keeping Token Usage To A Lower Amount We Employed Chunking Of Pages Contents To Allow Us To Minimize Generalizations Of The Page Contents During Training Data Generation. The openai Request Will Be Provided A Summary Of The Last 2 Chunks/Requests To Help Them Correlate Context Between These Chunks. This Summation Will Be Generated Again By An openai Request To Generalize The Provided Chunk Context. Data Analysis Currently Is Being Done By PyMuPDF And Tabula Mainly. Tabula Is Utilized For Data Tables Embedded In A PDF In Which Is Good To Help With Properly Giving openai A Structured Way Of Knowing What A Table Is Saying. This Has Worked Really Well But Still Seems To Have Some Flaws With Formatting Sometimes So Will Need To Look Into It In The Future. OCR And Blip Were Being Attempted But Never Gave Consistent Results With The Warzone That PDFs Are So Are Currently Deprecated (Still Have Blip Saying Something But Is Really Useless, OCR Needed Gaussian Blur And Never Gave Consistent Results).

----------------------------------------------
<img src="https://github.com/user-attachments/assets/b6920c17-2da4-4c3f-8efc-b94893955e04" alt="Cornstarch <3" width="55" height="49"> <img src="https://github.com/user-attachments/assets/b6920c17-2da4-4c3f-8efc-b94893955e04" alt="Cornstarch <3" width="55" height="49"> <img src="https://github.com/user-attachments/assets/b6920c17-2da4-4c3f-8efc-b94893955e04" alt="Cornstarch <3" width="55" height="49"> <img src="https://github.com/user-attachments/assets/b6920c17-2da4-4c3f-8efc-b94893955e04" alt="Cornstarch <3" width="55" height="49">


**The Breakdown:**

  #### **Config Beforehand**
   - The program requires two things to be set up initially:
     - PDF Documents to be used as the primary source material for fine-tuning.
     - An OpenAI API key for communication with OpenAI.
  
   - After configuring these details between lines 33-43, the process is ready to begin.
  
   - The process begins by splitting PDFs into worker threads, allowing for parallel processing:
     ```bash
     O(max(X)) << O(X1 + X2 + ... + Xn)
     ```
     - Threads are tracked using a class called `ProgressLogger`, which updates progress bars using `rich.progress` and `rich.live` for asynchronous updates. Mutex locks are used to avoid race conditions when logging status messages from multiple worker threads.

#### **Worker Function**
   - Worker threads run **process_pdf_with_summaries(...)** to extract PDF content using **PyMuPDF** through **extract_content_with_mupdf(...)**.
   - **PyMuPDF** extracts structured pieces of data, ensuring ligatures and combined text are preserved to avoid formatting issues common in PDFs.
  
   - Images on the current page are extracted, and **Blip-2** is used for captioning, although Blip-2 may be deprecated in favor of **LLaVa** or more robust alternatives.
  
   - **Tabula** is used for extracting tables and formatting them as strings.
  
   - Extracted data is compiled into a list of dictionaries for each page in the following format:
     ```json
     {
         "page_num": page_num,
         "text": text + " " + blip_caption + " " + table_data,
         "images": images,
         "metadata": doc.metadata,
         "annotations": list(page.annots()) if page.annots() else [],
         "links": page.get_links()
     }
     ```

#### **Processing Text for Model Fine-Tuning**
   - After raw content is extracted, the text cleaning phase begins:
     - Unnecessary elements like headers, footers, and excessive whitespace are removed using regular expressions.
  
   - The text is chunked using **chunk_text(...)** to fit within OpenAI's token limits. Each chunk is sent to OpenAI via **send_to_openai_with_retry(...)**, which also handles retries for failed API requests.

#### **Fine-Tuning Contextualization**
   - **summarize_text(...)** is used to generate summaries of the extracted text, reducing token usage while retaining key contextual information.
  
   - These summaries are used to provide context when processing subsequent chunks of the document.

#### **Generating JSON Files for Training**
   - Once all text chunks are processed, responses are stored in two JSON file formats:
     - **Raw output**: Direct responses from OpenAI.
     - **Cleaned output**: Responses with unnecessary formatting removed to ensure consistency across the dataset.
  
   - These files serve as the core training material for fine-tuning AI models on tasks like Arduino Mega 2560 Rev3 integration.

#### **Handling Multiple PDFs**
   - The **multiThreaded_process_directory(...)** function manages the parallel processing of multiple PDFs, with each document processed independently in worker threads.
  
   - This approach significantly reduces overall processing time, particularly when working with large datasets.
  
   - **ProgressLogger** ensures that users are informed of the processing status at all times. Logs are synchronized to avoid race conditions, ensuring coherent output across threads.

<img src="https://github.com/user-attachments/assets/1fc90166-a62b-4d91-a087-5da5a3a7076f" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/1fc90166-a62b-4d91-a087-5da5a3a7076f" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/1fc90166-a62b-4d91-a087-5da5a3a7076f" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/1fc90166-a62b-4d91-a087-5da5a3a7076f" alt="Cornstarch <3" width="65" height="59">

----------------------------------------------

<img src="https://github.com/user-attachments/assets/2c82de29-71b0-433a-bc10-3778026b74a7" alt="Cornstarch <3" width="65" height="59">  <img src="https://github.com/user-attachments/assets/2c82de29-71b0-433a-bc10-3778026b74a7" alt="Cornstarch <3" width="65" height="59">  <img src="https://github.com/user-attachments/assets/2c82de29-71b0-433a-bc10-3778026b74a7" alt="Cornstarch <3" width="65" height="59">  <img src="https://github.com/user-attachments/assets/2c82de29-71b0-433a-bc10-3778026b74a7" alt="Cornstarch <3" width="65" height="59"> 


**Features:**

  **Result After Parsing:**
  
  ![image](https://github.com/user-attachments/assets/5582883a-eac1-4cae-bae1-d388cef04758)


<img src="https://github.com/user-attachments/assets/b568a6eb-aa1d-4379-80a8-938dd51fa068" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/b568a6eb-aa1d-4379-80a8-938dd51fa068" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/b568a6eb-aa1d-4379-80a8-938dd51fa068" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/b568a6eb-aa1d-4379-80a8-938dd51fa068" alt="Cornstarch <3" width="65" height="59">
