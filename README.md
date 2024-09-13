# Python_FTM_PDF_Parser
Python Script In Which Tries To Accelerate The PDF Parsing Stage Of Many openai Fine-Tuning Models. 

This Is From The Multi-Threaded Approach Of Scanning Specific PDF Documents, Allowing Many Worker Threads To Talk To openai About Their Own Document. This Can Allow The Process To Be Scanning Multiple Documents In Paralell In Which Can Greatly Improve Runtime. To Improve Reliability While Keeping Token Usage To A Lower Amount We Employed Chunking Of Pages Contents To Allow Us To Minimize Generalizations Of The Page Contents During Training Data Generation. The openai Request Will Be Provided A Summary Of The Last 2 Chunks/Requests To Help Them Correlate Context Between These Chunks. This Summation Will Be Generated Again By An openai Request To Generalize The Provided Chunk Context. Data Analysis Currently Is Being Done By PyMuPDF And Tabula Mainly. Tabula Is Utilized For Data Tables Embedded In A PDF In Which Is Good To Help With Properly Giving openai A Structured Way Of Knowing What A Table Is Saying. This Has Worked Really Well But Still Seems To Have Some Flaws With Formatting Sometimes So Will Need To Look Into It In The Future. OCR And Blip Were Being Attempted But Never Gave Consistent Results With The Warzone That PDFs Are So Are Currently Deprecated (Still Have Blip Saying Something But Is Really Useless, OCR Needed Gaussian Blur And Never Gave Consistent Results).

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

  #### **- Parallel PDF Processing**
  When The Process Starts, It Will Attempt To Split The Given PDFs Into Their Own Worker Threads. This Allows Us To Process Many PDFs In Parallel Instead Of Sequentially. Let `X1, X2, ..., Xn` Represent The Time It Takes To Process Each PDF. The Time Complexity For Sequential Execution Is:
  
  ```
  O(X1 + X2 + ... + Xn) = O( Σ(Xi), for i=1 to n )
  ```


  
  In Contrast, The Time Complexity For Parallel Execution, Assuming Sufficient Threads, Is:
  ```
  O(max(X1, X2, ..., Xn))
  ```

  
  
  Since The Sum Of The Individual Times Is Always Greater Than Or Equal To The Maximum Time, We Have The Relationship:

  ```
  O(max(X1, X2, ..., Xn)) <= O( Σ(Xi), for i=1 to n )
  ```
 
  
  Therefore, Parallel Execution Can Result In A Significantly Reduced Runtime Compared To Sequential Execution, Particularly When There Is A Large Disparity Among The Individual Processing Times.

  
  All Of These Threads Processing PDFs Will Be Tracked By A Built Class Called `ProgressLogger`--In Which We Will Add As An Argument Into Their Worker Function--Utilizing _rich.progress_ For UI Progress Bars Of How Far Along The PDF Processing Is For A Given Document As Well As The Total Progress. To Keep It In Fixed Position At The Bottom Of The Window It Uses _rich.live_ To Allow It To Asynchronous Updates When Inputs Come From Terminal. I Currently Am Using Mutex Locks To Ensure That Race-Conditions Do Not Happen With Worker Threads In Which Are All Trying To Communicate Their Log Status to Our `ProgressLogger` Instance At Once But May Of Over-Developed Now Realizing _rich.progress_ Has Specific Console Print Commands To Ensure Formatting And Thread-Safety And Don't Need To Keep Logs On Memory If Just Written To File.

  #### **- PDF Content Extraction**
  The Workers Will Go And Run `process_pdf_with_summaries(...)` As Their Thread Function. In This Function It Will Start By Extracting The Contents Of The PDF Using _PyMuPDF_ Through A Page Extraction Function, `extract_content_with_mupdf(...)`. _PyMuPDF_ Will Work Well In Order To Grab Structured Pieces Of Data Compared To Previous Implementations With _PyPDF2_. _PyMuPDF_ Will Start By Grabbing The Raw Text Of The Page, Ensuring Ligatures (Or Combined Text) Is Properly Represented As PDFs Seem To Have Odd Formatting And Can Cause Line Skips. We Then Grab The Images On The Current Page, Ensuring _PyMuPDF_ Gives As Much Metadata On The Image As Possible To Enhance Contexting To Our AI. We Then Load These Images Into Memory And Run A _Blip-2_ Captioning On The Given Image (This Is Also Where OCR Is Commented Out). _Blip-2_ Works Kinda Good At Knowing What A Photo Is Shaped To Be But Has No Sense Of Detail So Probably Will Be Deprecated And Could Use _LLaVa_ Or Something More Local And Proper.

  #### **- Table Extraction**
  After Getting These _Blip-2_ Captions, We Then Scan The Document Page For Any Tables Through _Tabula_. This Will Quickly Format Them Into Strings. After Getting The Caption, Text, And Table Data It Will All Be Added In A List Of Dictionaries In Which Are Formatted As Follows:
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
  This Helps Contextualize Structure Of The Page To The AI Model So It Has More Meaning Of What The Page IS. While It's Not All In-Line For When It Occurs Like table_data I May Have To Go To _PDFMiner_ And _PDFPlumber_ Or Change To Parsing By Dictionaries So I Can Dynamically Parse A Page By Group Sections As Right Now Its Hard To Correlate Figures To Given Images Or Tables. Still Currently Is Really Good Though Currently I've Just Been Holding Back Changes Like These For Until I Get LLaVa Set-Up For Local Multi-Modal Image Processing As Right Now I Have A Stable Foundation And Don't Want To Mess With It Before A Big Lib Deployment.
  
  #### **- Processing Text For Model Fine-Tuning**
After The Raw Content From The PDF Has Been Extracted And Organized, The Text Cleaning Phase Begins. In This Phase, The Raw Text From Each Page Is Preprocessed To Remove Unnecessary Clutter, Such As Excessive Whitespace And Headers Or Footers That Could Hinder The AI Model's Contextual Understanding. Regular Expressions Are Used To Normalize The Text By Eliminating Redundant Spaces And Ensuring Proper Paragraph Formatting.  
Once The Text Is Cleaned, It Is Ready To Be Sent To OpenAI's GPT-Based Models For Further Fine-Tuning. To Avoid Exceeding The Token Limit Imposed By The API, The Function `chunk_text(...)` Splits The Cleaned Text Into Smaller, Manageable Chunks. This Step Is Crucial, Especially For Large Documents, As It Ensures That No Data Is Lost While Maintaining A Seamless Integration With The Language Model.

#### **- Sending Data To OpenAI For Processing**
Each Text Chunk Is Processed Using The `send_to_openai_with_retry(...)` Function. This Function Handles Communication With The OpenAI API, Ensuring That Each Text Chunk Is Passed Along With Any Relevant Context From Previous Pages. If An API Request Fails, A Retry Mechanism Is In Place To Ensure Reliability During Network Instability Or Other Transient Failures. The Model's Response Is Then Cleaned Up To Remove Any Unnecessary Artifacts, Preparing It For Use In Training Datasets. When Providing Out Prompt We Follow The openai Standard For Their .jsonl Files With:
  ```
  
  {{
    "messages": [
        {{"role": "system", "content": "AI's Tone/Expertise"}},
        {{"role": "user", "content": "(Question About Training Data Contents)"}},
        {{"role": "assistant", "content": "(Response To Question, Showing Diagrams, Code, Etc..)"}}
    ]
  }}
  
  
  ```

#### **- Fine-Tuning Contextualization**
After Extracting The Raw Responses From OpenAI, The Program Generates Summaries Using `summarize_text(...)`. These Summaries Provide A Condensed Form Of The Extracted Text, Reducing Token Usage While Retaining Key Contextual Details From The Document. These Summaries Are Stored And Used As Context When Processing Subsequent Pages, Allowing The Model To Maintain A Coherent Understanding Of The Document As A Whole.

#### **- Generating JSON Files For Training**
Once The Model Has Processed The Text Chunks, All Responses Are Stored In A JSON Format. Two Versions Of These JSON Files Are Created: A Raw Output Containing All Responses, And A Cleaned Output That Strips Away Formatting Inconsistencies And Ensures Consistency Across The Dataset (`clean_response(...)`). 

Finally, The Program Ends Once All PDFs Have Been Processed, With Both The Progress Bar And Log Information Indicating Completion With A Solid Green Bar.


<img src="https://github.com/user-attachments/assets/1fc90166-a62b-4d91-a087-5da5a3a7076f" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/1fc90166-a62b-4d91-a087-5da5a3a7076f" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/1fc90166-a62b-4d91-a087-5da5a3a7076f" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/1fc90166-a62b-4d91-a087-5da5a3a7076f" alt="Cornstarch <3" width="65" height="59">

----------------------------------------------

<img src="https://github.com/user-attachments/assets/2c82de29-71b0-433a-bc10-3778026b74a7" alt="Cornstarch <3" width="65" height="59">  <img src="https://github.com/user-attachments/assets/2c82de29-71b0-433a-bc10-3778026b74a7" alt="Cornstarch <3" width="65" height="59">  <img src="https://github.com/user-attachments/assets/2c82de29-71b0-433a-bc10-3778026b74a7" alt="Cornstarch <3" width="65" height="59">  <img src="https://github.com/user-attachments/assets/2c82de29-71b0-433a-bc10-3778026b74a7" alt="Cornstarch <3" width="65" height="59"> 


### **Features:**

  **Script Parsing And Training PDF Data:**
  
  ![2024-09-1300-32-48-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/d6577304-86de-41bd-a678-537d10918f6a)


 **Final Training Data Example:**

 <img width="1768" alt="image" src="https://github.com/user-attachments/assets/61479c16-fa29-4fc9-94c6-fbf885a73a4e">



<img src="https://github.com/user-attachments/assets/b568a6eb-aa1d-4379-80a8-938dd51fa068" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/b568a6eb-aa1d-4379-80a8-938dd51fa068" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/b568a6eb-aa1d-4379-80a8-938dd51fa068" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/b568a6eb-aa1d-4379-80a8-938dd51fa068" alt="Cornstarch <3" width="65" height="59">
