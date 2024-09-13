# Python_FTM_PDF_Parser
Python Script In Which Tries To Accelerate The PDF Parsing Stage Of Many openai Fine-Tuning Models. 

This Is From The Multi-Threaded Approach Of Scanning Specific PDF Documents, Allowing Many Worker Threads To Talk To openai About Their Own Document. This Can Allow The Process To Be Scanning Multiple Documents In Paralell In Which Can Greatly Improve Runtime. To Improve Reliability While Keeping Token Usage To A Lower Amount We Employed Chunking Of Pages Contents To Allow Us To Minimize Generalizations Of The Page Contents During Training Data Generation. The openai Request Will Be Provided A Summary Of The Last 2 Chunks/Requests To Help Them Correlate Context Between These Chunks. This Summation Will Be Generated Again By An openai Request To Generalize The Provided Chunk Context. Data Analysis Currently Is Being Done By PyMuPDF And Tabula Mainly. Tabula Is Utilized For Data Tables Embedded In A PDF In Which Is Good To Help With Properly Giving openai A Structured Way Of Knowing What A Table Is Saying. This Has Worked Really Well But Still Seems To Have Some Flaws With Formatting Sometimes So Will Need To Look Into It In The Future. OCR And Blip Were Being Attempted But Never Gave Consistent Results With The Warzone That PDFs Are So Are Currently Deprecated (Still Have Blip Saying Something But Is Really Useless, OCR Needed Gaussian Blur And Never Gave Consistent Results).

----------------------------------------------

<img src="https://github.com/user-attachments/assets/b6920c17-2da4-4c3f-8efc-b94893955e04" alt="Cornstarch <3" width="55" height="49"> <img src="https://github.com/user-attachments/assets/b6920c17-2da4-4c3f-8efc-b94893955e04" alt="Cornstarch <3" width="55" height="49"> <img src="https://github.com/user-attachments/assets/b6920c17-2da4-4c3f-8efc-b94893955e04" alt="Cornstarch <3" width="55" height="49"> <img src="https://github.com/user-attachments/assets/b6920c17-2da4-4c3f-8efc-b94893955e04" alt="Cornstarch <3" width="55" height="49">


### **The Breakdown:**

  #### **Config Beforehand**
  The Program Starts By Initially Needing 2 Things To Be Done. 

  1.) Initially Garner Some PDF Documents In Which You Want To Use To Use As Primary Source Material To Fine-Tune A Model With.
  2.) Establish An API Key In Which You Are Using To Communicate With _openai_ With.
  
  After You Fill In These Details Within The Config Located On Line 33 -> Line 43 You Will Be Set.

  #### **Parallel PDF Processing**
  When The Process Starts It Will Initially Attempt To Split The Given PDFs Into Their Own Worker Threads. This Allows Us To Process Many PDFs In Paralell Instead Of Sequentially As We Gain:
  ```
  O(max(X)) << O(X1 + X2 + ... + Xn)
  ```
  In Runtime.
  
  All Of These Threads Processing PDFs Will Be Tracked By A Built Class Called ProgressLogger--In Which We Will Add As An Argument Into Their Worker Function--Utilizing _rich.progress_ For UI Progress Bars Of How Far Along The PDF Processing Is For A Given Document As Well As The Total Progress. To Keep It In Fixed Position At The Bottom Of The Window It Uses _rich.live_ To Allow It To Asynchronous Updates When Inputs Come From Terminal. I Currently Am Using Mutex Locks To Ensure That Race-Conditions Do Not Happen With Worker Threads In Which Are All Trying To Communicate Their Log Status to Our ProgressLogger Instance At Once But May Of Over-Developed Now Realizing _rich.progress_ Has Specific Console Print Commands To Ensure Formatting And Thread-Safety And Don't Need To Keep Logs On Memory If Just Written To File.

  #### **PDF Content Extraction**
  The Workers Will Go And Run **process_pdf_with_summaries(...)** As Their Thread Function. In This Function It Will Start By Extracting The Contents Of The PDF Using _PyMuPDF_ Through A Page Extraction Function, **extract_content_with_mupdf(...)**. _PyMuPDF_ Will Work Well In Order To Grab Structured Pieces Of Data Compared To Previous Implementations With _PyPDF2_. _PyMuPDF_ Will Start By Grabbing The Raw Text Of The Page, Ensuring Ligatures (Or Combined Text) Is Properly Represented As PDFs Seem To Have Odd Formatting And Can Cause Line Skips. We Then Grab The Images On The Current Page, Ensuring _PyMuPDF_ Gives As Much Metadata On The Image As Possible To Enhance Contexting To Our AI. We Then Load These Images Into Memory And Run A Blip-2 Captioning On The Given Image (This Is Also Where OCR Is Commented Out). Blip-2 Works Kinda Good At Knowing What A Photo Is Shaped To Be But Has No Sense Of Detail So Probably Will Be Deprecated And Could Use LLaVa Or Something More Local And Proper.

  #### **Table Extraction**
  After Getting These Blip-2 Captions, We Then Scan The Document Page For Any Tables Through Tabula. This Will Quickly Format Them Into Strings. After Getting The Caption, Text, And Table Data It Will All Be Added In A List Of Dictionaries In Which Are Formatted As Follows:
  ```
  {
      "page_num": page_num,
      "text": text + " " + " " + blip_caption + " " + table_data,
      "images": images,
      "metadata": doc.metadata,
      "annotations": list(page.annots()) if page.annots() else [],
      "links": page.get_links()
  }
  ```
  This Helps Contextualize Structure Of The Page To The AI Model So It Has More Meaning Of What The Page IS. While It's Not All In-Line For When It Occurs Like table_data I May Have To Go To PDFMiner And PDFPlumber Or Change To Parsing By Dictionaries So I Can Dynamically Parse A Page By Group Sections As Right Now Its Hard To Correlate Figures To Given Images Or Tables. Still Currently Is Really Good Though Currently I've Just Been Holding Back Changes Like These For Until I Get LLaVa Set-Up For Local Multi-Modal Image Processing As Right Now I Have A Stable Foundation And Don't Want To Mess With It Before A Big Lib Deployment.
  
  #### **Processing Text for Model Fine-Tuning**
After the raw content from the PDF has been extracted and organized, the text cleaning phase begins. In this phase, the raw text from each page is preprocessed to remove unnecessary clutter, such as excessive whitespace and headers or footers that could hinder the AI model's contextual understanding. Regular expressions are used to normalize the text by eliminating redundant spaces and ensuring proper paragraph formatting.
Once the text is cleaned, it is ready to be sent to OpenAI's GPT-based models for further fine-tuning. To avoid exceeding the token limit imposed by the API, the function chunk_text(...) splits the cleaned text into smaller, manageable chunks. This step is crucial, especially for large documents, as it ensures that no data is lost while maintaining a seamless integration with the language model.

#### **Sending Data to OpenAI for Processing**
Each text chunk is processed using the send_to_openai_with_retry(...) function. This function handles communication with the OpenAI API, ensuring that each text chunk is passed along with any relevant context from previous pages. If an API request fails, a retry mechanism is in place to ensure reliability during network instability or other transient failures. The model's response is then cleaned up to remove any unnecessary artifacts, preparing it for use in training datasets.

#### **Fine-Tuning Contextualization**
After extracting the raw responses from OpenAI, the program generates summaries using summarize_text(...). These summaries provide a condensed form of the extracted text, reducing token usage while retaining key contextual details from the document. These summaries are stored and used as context when processing subsequent pages, allowing the model to maintain a coherent understanding of the document as a whole.

#### **Generating JSON Files for Training**
Once the model has processed the text chunks, all responses are stored in a JSON format. Two versions of these JSON files are created: a raw output containing all responses, and a cleaned output that strips away formatting inconsistencies and ensures consistency across the dataset. These files serve as the core training material for fine-tuning the AI model on specific tasks related to Arduino Mega 2560 Rev3 integration, covering both hardware and software perspectives.

#### **Handling Multiple PDFs**
The entire pipeline is designed to handle multiple PDF documents concurrently. The multiThreaded_process_directory(...) function manages the parallel execution of the PDF processing tasks, utilizing worker threads to ensure that each document is processed independently. This significantly reduces overall processing time, particularly when dealing with large datasets.
The program's use of structured logging through the ProgressLogger ensures that users are constantly informed about the processing status. The logs are synchronized across multiple threads to avoid race conditions, ensuring that log messages are printed in a coherent, non-overlapping manner.

Finally, the program ends once all PDFs have been processed, with both the progress bar and log information indicating completion. This pipeline ensures that fine-tuned models receive well-organized and contextualized data for high-quality training outcomes.

<img src="https://github.com/user-attachments/assets/1fc90166-a62b-4d91-a087-5da5a3a7076f" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/1fc90166-a62b-4d91-a087-5da5a3a7076f" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/1fc90166-a62b-4d91-a087-5da5a3a7076f" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/1fc90166-a62b-4d91-a087-5da5a3a7076f" alt="Cornstarch <3" width="65" height="59">

----------------------------------------------

<img src="https://github.com/user-attachments/assets/2c82de29-71b0-433a-bc10-3778026b74a7" alt="Cornstarch <3" width="65" height="59">  <img src="https://github.com/user-attachments/assets/2c82de29-71b0-433a-bc10-3778026b74a7" alt="Cornstarch <3" width="65" height="59">  <img src="https://github.com/user-attachments/assets/2c82de29-71b0-433a-bc10-3778026b74a7" alt="Cornstarch <3" width="65" height="59">  <img src="https://github.com/user-attachments/assets/2c82de29-71b0-433a-bc10-3778026b74a7" alt="Cornstarch <3" width="65" height="59"> 


**Features:**

  **Result After Parsing:**
  
  ![image](https://github.com/user-attachments/assets/5582883a-eac1-4cae-bae1-d388cef04758)


<img src="https://github.com/user-attachments/assets/b568a6eb-aa1d-4379-80a8-938dd51fa068" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/b568a6eb-aa1d-4379-80a8-938dd51fa068" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/b568a6eb-aa1d-4379-80a8-938dd51fa068" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/b568a6eb-aa1d-4379-80a8-938dd51fa068" alt="Cornstarch <3" width="65" height="59">
