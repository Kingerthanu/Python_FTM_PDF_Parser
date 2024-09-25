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
  Now Implemented Is Two Local LLM Models Provided Through An ollama Session (In Which Is Why You See This As A Dependency In This Project):
  \L <img src="https://github.com/user-attachments/assets/5b7de0ce-5f9b-4922-973c-a702bc92c695" alt="Cornstarch <3" width="25" height="25"><br>
   |
  \L <img src="https://github.com/user-attachments/assets/39175c73-b5cd-4583-ad2d-7809c0be13ba" alt="Cornstarch <3" width="25" height="25">
  
  We Will Install Specifically Two Models; One Is Called The **Large Language and Vision Assistant (LLaVA)** And Is Employed To Give Detailed Explanations Of Images Through Text. This Can Be Employed On Image's Seen In Training Data PDFs To Allow Us To Get A Much More Detailed Analysis Of The Image Than **BLIP-2** Ever Gave Us. Another One Is Called **LLaMA** (Sadly Initials Don't Go To Anything For Some Reason) And Is Made My Meta AI. This Is A Very Performant AI Model In Which Benchmarks Almost As Good As OpenAI's Models In Many Tests. We Use **LLaMA** and **LLaVA** As Local LLMs To Allow Us To Cut-Down On Some Excessive Token Usage. Also Things Like **LLaVA** Have No Real Alternative Other Than Paid Subscriptions Like OpenAI's Models. So While We Are Running Beefy Codebases Under Our Computer To Scan And Reason Images, It Does Allow Us To Get A Much More Lavish Explanation Of Image Data.


  

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
