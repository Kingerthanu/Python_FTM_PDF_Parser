# Python_FTM_PDF_Parser
Python Script In Which Tries To Accelerate The PDF Parsing Stage Of Many openai Fine-Tuning Models. 

This Is From The Multi-Threaded Approach Of Scanning Specific PDF Documents, Allowing Many Worker Threads To Talk To openai About Their Own Document. This Can Allow The Process To Be Scanning Multiple Documents In Paralell In Which Can Greatly Improve Runtime. To Improve Reliability While Keeping Token Usage To A Lower Amount We Employed Chunking Of Pages Contents To Allow Us To Minimize Generalizations Of The Page Contents During Training Data Generation. The openai Request Will Be Provided A Summary Of The Last 2 Chunks/Requests To Help Them Correlate Context Between These Chunks. This Summation Will Be Generated Again By An openai Request To Generalize The Provided Chunk Context. Data Analysis Currently Is Being Done By PyMuPDF And Tabula Mainly. Tabula Is Utilized For Data Tables Embedded In A PDF In Which Is Good To Help With Properly Giving openai A Structured Way Of Knowing What A Table Is Saying. This Has Worked Really Well But Still Seems To Have Some Flaws With Formatting Sometimes So Will Need To Look Into It In The Future. OCR And Blip Were Being Attempted But Never Gave Consistent Results With The Warzone That PDFs Are So Are Currently Deprecated (Still Have Blip Saying Something But Is Really Useless, OCR Needed Gaussian Blur And Never Gave Consistent Results).

----------------------------------------------
<img src="https://github.com/user-attachments/assets/b6920c17-2da4-4c3f-8efc-b94893955e04" alt="Cornstarch <3" width="55" height="49"> <img src="https://github.com/user-attachments/assets/b6920c17-2da4-4c3f-8efc-b94893955e04" alt="Cornstarch <3" width="55" height="49"> <img src="https://github.com/user-attachments/assets/b6920c17-2da4-4c3f-8efc-b94893955e04" alt="Cornstarch <3" width="55" height="49"> <img src="https://github.com/user-attachments/assets/b6920c17-2da4-4c3f-8efc-b94893955e04" alt="Cornstarch <3" width="55" height="49">


**The Breakdown:**
  The Program Starts By Initially Needing 2 Things To Be Done. 
  
  This Is To Initially Garner Some PDF Documents In Which You Want To Use To Use As Primary Source Material To Fine-Tune A Model With. As Well As Establishing An API Key In Which You Are Using To Communicate With _openai_ With.

  After You Fill In These Details Within The Config Located On Line 33 -> Line 43 You Will Be Set.

  When The Process Starts It Will Initially Attempt To Split The Given PDFs Into Their Own Worker Threads. This Allows Us To Process Many PDFs In Paralell Instead Of Sequentially As We Gain:
  ```
  O(max(X)) << O(X1 + X2 + ... + Xn)
  ```
  In Runtime.
  
  All Of These Threads Processing PDFs Will Be Tracked By A Built Class Called ProgressLogger--In Which We Will Add As An Argument Into Their Worker Function--Utilizing _rich.progress_ For UI Progress Bars Of How Far Along The PDF Processing Is For A Given Document As Well As The Total Progress. To Keep It In Fixed Position At The Bottom Of The Window It Uses _rich.live_ To Allow It To Asynchronous Updates When Inputs Come From Terminal. I Currently Am Using Mutex Locks To Ensure That Race-Conditions Do Not Happen With Worker Threads In Which Are All Trying To Communicate Their Log Status to Our ProgressLogger Instance At Once But May Of Over-Developed Now Realizing _rich.progress_ Has Specific Console Print Commands To Ensure Formatting And Thread-Safety And Don't Need To Keep Logs On Memory If Just Written To File.

  The Workers Will Go And Run **process_pdf_with_summaries(...)** As Their Thread Function. In This Function It Will Start By Extracting The Contents Of The PDF Using _PyMuPDF_ Through A Page Extraction Function, **extract_content_with_mupdf(...)**.

  
  
<img src="https://github.com/user-attachments/assets/1fc90166-a62b-4d91-a087-5da5a3a7076f" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/1fc90166-a62b-4d91-a087-5da5a3a7076f" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/1fc90166-a62b-4d91-a087-5da5a3a7076f" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/1fc90166-a62b-4d91-a087-5da5a3a7076f" alt="Cornstarch <3" width="65" height="59">

----------------------------------------------

<img src="https://github.com/user-attachments/assets/2c82de29-71b0-433a-bc10-3778026b74a7" alt="Cornstarch <3" width="65" height="59">  <img src="https://github.com/user-attachments/assets/2c82de29-71b0-433a-bc10-3778026b74a7" alt="Cornstarch <3" width="65" height="59">  <img src="https://github.com/user-attachments/assets/2c82de29-71b0-433a-bc10-3778026b74a7" alt="Cornstarch <3" width="65" height="59">  <img src="https://github.com/user-attachments/assets/2c82de29-71b0-433a-bc10-3778026b74a7" alt="Cornstarch <3" width="65" height="59"> 


**Features:**

  **Result After Parsing:**
  
  ![image](https://github.com/user-attachments/assets/5582883a-eac1-4cae-bae1-d388cef04758)


<img src="https://github.com/user-attachments/assets/b568a6eb-aa1d-4379-80a8-938dd51fa068" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/b568a6eb-aa1d-4379-80a8-938dd51fa068" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/b568a6eb-aa1d-4379-80a8-938dd51fa068" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/b568a6eb-aa1d-4379-80a8-938dd51fa068" alt="Cornstarch <3" width="65" height="59">
