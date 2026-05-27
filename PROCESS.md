# PROCESS.md — WorkerCompass

---

Log 1

- Project scoping
    - Looked through Annex A and {build} projects, as well as OGP's Build For Good projects
    - 3 main project types: building an internal tool, building a agency-specific/WOG tool, building a tool that benefits the general public
    - A few factors that I considered: personal preference, ease of fit to themes provided, novelty+feasibility+impact of the idea, availability of data and ease of data collection, time constraints

    - Personal preference
        - My personal preference was to have a project focused on vulnerable groups in the community. One of the ideas that I thought of in the past was to proactively identify students with learning disabilities through data collected by the school on the students. (ie. exam grades, attendance, behavioural records). However, this was not very feasible due to the privacy of the data and constraints. It is also more of a traditional ML problem rather than something that can be solved with an AI use case
        - Building on that, a few other demographics that stood out to me were the elderly, disabled, low-income households, or migrant workers.
        - As I have interned under MOM before, I naturally gravitated towards targetting an unaddressed gap in the migrant worker community
    
    - Fitting to provided themes
        - With the scope narrowed down to migrant workers, I tried to think of possible use cases that fit the themes for the project, with AI being central to the solution
        - A main consideration at this point was also the availability of data and the ease of data collection, as that would be the first bottleneck in my development speed
        - Because of all the policy work that MOM handles, I thought of the idea of policies + migrant workers -> something to help migrant workers understand their rights and entitlements, and how to integrate the project themes into that.
        - The first thing that came to mind to me was an agentic chatbot that would help migrant workers understand policies and verify if they are being treated fairly.
        - In this way, the project scope would cover agents & workflows through the chatbot, retrieval and knowledge with RAG over policy documents, and also pulling structured data from the unstructured policy documents across different sources.
        - After some ideating, my initial problem statement was: A multilingual chatbot assistant for low-wage migrant workers in Singapore that answers employment-rights related questions, grounded with RAG on policy documents.

    - Assessing the novelty, technical feasibility and impact of the idea
        - Similar products already built by a government agency, or ideated on in {build} by GovTech/Build For Good by OGP, or built by independent developers/non-profit groups
        - Feasibility: availability of data - publicly available documents/FAQ pages from government websites like MOM/TAFEP. Initial thought is to scrape it or just download
        - Impact: number of migrant workers in Singapore, statistics of workplace violations/salary/discrimination etc. Low English literacy, how this would help

 - Technical Considerations
    - Overall architecture
        - Streamlit for a quick UI layer
    - Agent architecture and framework
        - Agno with built-in features: PII guardrail, high-level Python abstraction
    - Data collection
        - scraping: target which websites? MOM, TAFEP, TADM, SSO Acts
        - downloading of policy documents
        - what data format to save in?
        - embedding into vectordb, first cut choice of vectordb, how to embed, chunk size, etc
    - Query and response translation
        - how and where to translate?
        - how -> which embedding model
        - where -> translate before embedding? or translate at embedding. tradeoffs
        - first cut decided method
    - Retrieval
        - what to use for retrieval and why
    - Safety and guardrails
        - prevent out of topic questions
        - refuse request if answer is not grounded and redirect to proper helplines/ask follow-up questions that might be related
    - Output generation
        - expandable/clickable source citations
        - ask for feedback: useful or not?
    - Current models
        - SEA-HELM
        - Gemma-SEA-LION
    - Evaluations
        - Recall@k, precision@k
        - Claudes top p or something (demystifying agent evals)
        - Citation accuracy
        - Refusal correctness
        - RAGAS? cross-lingual quality

---

