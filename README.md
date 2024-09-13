# Python_FTM_PDF_Parser
Python Script In Which Tries To Accelerate The PDF Parsing Stage Of Many openai Fine-Tuning Models. 

This Is From The Multi-Threaded Approach Of Scanning Specific PDF Documents, Allowing Many Worker Threads To Talk To openai About Their Own Document. This Can Allow The Process To Be Scanning Multiple Documents In Paralell In Which Can Greatly Improve Runtime. To Improve Reliability While Keeping Token Usage To A Lower Amount We Employed Chunking Of Pages Contents To Allow Us To Minimize Generalizations Of The Page Contents During Training Data Generation. The openai Request Will Be Provided A Summary Of The Last 2 Chunks/Requests To Help Them Correlate Context Between These Chunks. This Summation Will Be Generated Again By An openai Request To Generalize The Provided Chunk Context. Data Analysis Currently Is Being Done By PyMuPDF And Tabula Mainly. Tabula Is Utilized For Data Tables Embedded In A PDF In Which Is Good To Help With Properly Giving openai A Structured Way Of Knowing What A Table Is Saying. This Has Worked Really Well But Still Seems To Have Some Flaws With Formatting Sometimes So Will Need To Look Into It In The Future. OCR And Blip Were Being Attempted But Never Gave Consistent Results With The Warzone That PDFs Are So Are Currently Deprecated (Still Have Blip Saying Something But Is Really Useless, OCR Needed Gaussian Blur And Never Gave Consistent Results).

----------------------------------------------
<img src="https://github.com/user-attachments/assets/b6920c17-2da4-4c3f-8efc-b94893955e04" alt="Cornstarch <3" width="55" height="49"> <img src="https://github.com/user-attachments/assets/b6920c17-2da4-4c3f-8efc-b94893955e04" alt="Cornstarch <3" width="55" height="49"> <img src="https://github.com/user-attachments/assets/b6920c17-2da4-4c3f-8efc-b94893955e04" alt="Cornstarch <3" width="55" height="49"> <img src="https://github.com/user-attachments/assets/b6920c17-2da4-4c3f-8efc-b94893955e04" alt="Cornstarch <3" width="55" height="49">


**The Breakdown:**

  Before Running The Program 3 Specific Things Need To Be Done To Ensure Valid Benchmarking These Are: <br>
    &nbsp;1.) Provide fine_tune_id On Line 263. <br>
    &nbsp;2.) Provide An openai.api_key On Line 6. <br>
    &nbsp;3.) Provide A .json File In Which Follows The Style-Guideline (**Shown Below**), Filling In "question" And "expected_answer" For Each Question In Questions <br>

  The Stylization Of The Benchmarks Should Be As Follows, Provided In A .json:

    # You Don't Need "given_answer", "differences" or "similarity" As Will Be Generated During Runtime
    {
      "questions": [
            {
              "question": "YOUR QUESTION",
              "expected_answer": "YOUR EXPECTED ANSWER (WHAT YOU WANT THE FINE-TUNED MODEL TO SAY)",
              "given_answer": "____",
              "similarity": ____,
              "differences": ____
            },
            {
              "question": "YOUR QUESTION",
              "expected_answer": "YOUR EXPECTED ANSWER (WHAT YOU WANT THE FINE-TUNED MODEL TO SAY)",
              "given_answer": "____",
              "similarity": ____,
              "differences": ____
            },
          ]
    }

  After The 3 Preliminary Tasks Are Complete, You Can Run The Script. 

  When Running The Script We Will Start By Initially Finding The Actual Model ID Associated With The Fine-Tuned Model ID We Are Provided By openai When Fine-Tuning (**fine_tune_id**). If We Provided A Valid Fine-Tuned ID, We Will Then Dump The Contents Of Our "questions" (Benchmarks) To Be Done Out Into A Struct. To Then Evaluate Each "question" In Our Pulled From .json We Will Use Multi-Threading. This Is Achieved In **evaluate_benchmark(...)**; We Will Send Off A Worker Thread To Process An Individual Benchmarking Question Entry, This Allows Us To Process Many Benchmarks In Paralell Instead Of Sequentially Processing Questions.

  In The Worker Function (**process_question(...)**), It Will Be Given An Individual Benchmark And In Each One Of These, We Will Have A "question" And "expected_answer". Initially We Will Ask Our Fine-Tuned Model Our Question, Getting It's Response. From This Fine-Tuned Model's Response, We Will Then Compare It To The Solution We Expected To Get Thats Provided in "expected_answer". Using openai Again, We Ask The ChatGPT-4 Model To Compare And Give A Similarity Score Between These Two Answers Based Upon Theoretical And Semantic Relations--This Can Allow Us To Quickly Compare Our Differing Solutions And Recommend Changes To Our Model If Lacking In A Specific Subtopic In The Fine-Tuned Model's Informational Knowledge.

  After These Worker Threads Ask These Questions And Get Their Similarity Scores, We Will Then Add Them All Back Together In A List Struct. Now When Leaving **evaluate_benchmark(...)** We Return This List Struct Of All The Worker Threads' Answers And Can Inject This Back Into The .json Provided, Adding "given_answer" "similarity" And "differences" Now As Entries For Each Benchmark. Before The Process Ends We Will Also Quickly Print The Contents Of Each Benchmark Out, Outlining The Similarity Score And Reasoning For This Score In Terminal.

<img src="https://github.com/user-attachments/assets/1adadc05-1b69-4710-8cd8-632deb67dbb5" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/1adadc05-1b69-4710-8cd8-632deb67dbb5" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/1adadc05-1b69-4710-8cd8-632deb67dbb5" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/1adadc05-1b69-4710-8cd8-632deb67dbb5" alt="Cornstarch <3" width="65" height="59">

----------------------------------------------

<img src="https://github.com/user-attachments/assets/2c82de29-71b0-433a-bc10-3778026b74a7" alt="Cornstarch <3" width="55" height="49"> <img src="https://github.com/user-attachments/assets/2c82de29-71b0-433a-bc10-3778026b74a7" alt="Cornstarch <3" width="55" height="49"> <img src="https://github.com/user-attachments/assets/2c82de29-71b0-433a-bc10-3778026b74a7" alt="Cornstarch <3" width="55" height="49"> <img src="https://github.com/user-attachments/assets/2c82de29-71b0-433a-bc10-3778026b74a7" alt="Cornstarch <3" width="55" height="49">


**Features:**

  **Result After Parsing:**
  
  ![image](https://github.com/user-attachments/assets/5582883a-eac1-4cae-bae1-d388cef04758)


<img src="https://github.com/user-attachments/assets/b568a6eb-aa1d-4379-80a8-938dd51fa068" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/b568a6eb-aa1d-4379-80a8-938dd51fa068" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/b568a6eb-aa1d-4379-80a8-938dd51fa068" alt="Cornstarch <3" width="65" height="59"> <img src="https://github.com/user-attachments/assets/b568a6eb-aa1d-4379-80a8-938dd51fa068" alt="Cornstarch <3" width="65" height="59">
