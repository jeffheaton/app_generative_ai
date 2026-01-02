# T81 559: Applications of Generative Artificial Intelligence

[Washington University in St. Louis](http://www.wustl.edu)

Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/)

- Section 1. Spring 2026, Wednesday, 6:00 PM  
  Location: CUPPLES II, Room 00203

# Course Description

This course covers the dynamic world of Generative Artificial Intelligence providing hands-on practical applications of Large Language Models (LLMs) and advanced text-to-image networks. Using Python as the primary tool, students will interact with OpenAI's models for both text and images. The course begins with a solid foundation in generative AI principles, moving swiftly into the utilization of LangChain for model-agnostic access and the management of prompts, indexes, chains, and agents. A significant focus is placed on the integration of the Retrieval-Augmented Generation (RAG) model with graph databases, unlocking new possibilities in AI applications.

As the course progresses, students will delve into sophisticated image generation and augmentation techniques, including LoRA (Low-Rank Adaptation), and learn the art of fine-tuning generative neural networks for specific needs. The final part of the course is dedicated to mastering prompt engineering, a critical skill for optimizing the efficiency and creativity of AI outputs.

**Note:** This course will require the purchase of up to **$100 in OpenAI API credits**.

# Objectives

1. Learn how Generative AI fits into the landscape of deep learning and predictive AI.
2. Create ChatBots, Agents, and other LLM-based automation assistants.
3. Understand how to use image generative AI programmatically.

# Syllabus

This [syllabus](https://data.heatonresearch.com/wustl/syllabus/jheaton-t81-559-fall-2025-syllabus.pdf) presents the expected class schedule, due dates, and reading assignments.

| Module                                                                         | Content                                                                                                                                                                                                                                                                                                                                        |
| ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Module 1](t81_559_class_01_1_overview.ipynb)<br>**Meet on 01/12/2026**        | **Introduction to Generative AI**<br>1.1 Course Overview<br>1.2 Generative AI Overview<br>1.3 Introduction to OpenAI<br>1.4 Introduction to LangChain<br>1.5 Prompt Engineering<br>**We will meet on campus this week (in-class meeting #1)**                                                                                                  |
| [Module 2](t81_559_class_02_1_dev.ipynb)<br>Week of 01/20/2026                 | **Prompt-Based Development**<br>2.1 Prompting for Code Generation<br>2.2 Handling Revision Prompts<br>2.3 Using an LLM to Debug<br>2.4 Tracking Prompts in Software Development<br>2.5 Limits of LLM Code Generation<br>[Module 1 Program](./assignments/assignment_yourname_t81_559_class1.ipynb) due 01/20/2026<br>Icebreaker due 01/20/2026 |
| [Module 3](t81_559_class_03_1_llm.ipynb)<br>**Meet on 01/26/2026**             | **Large Language Models**<br>3.1 Foundation Models<br>3.2 Text Generation<br>3.3 Text Summarization<br>3.4 Text Classification<br>3.5 LLM Writes a Book<br>[Module 2 Program](./assignments/assignment_yourname_t81_559_class2.ipynb) due 01/27/2026<br>**We will meet on campus this week (in-class meeting #2)**                             |
| [Module 4](t81_559_class_04_1_langchain_chat.ipynb)<br>**Meet on 02/02/2026**  | **LangChain: Chat and Memory**<br>4.1 Conversations<br>4.2 Buffer Window Memory<br>4.3 Summary + Fixed Window Chat<br>4.4 Persistence, Rollback, Regeneration<br>4.5 Automated Coder Application<br>[Module 3 Program](./assignments/assignment_yourname_t81_559_class3.ipynb) due 02/03/2026                                                  |
| [Module 5](t81_559_class_05_1_langchain_data.ipynb)<br>Week of 02/09/2026      | **LangChain: Data Extraction**<br>5.1 Structured Output Parser<br>5.2 CSV, JSON, Pandas, Datetime<br>5.3 Pydantic Parser<br>5.4 Custom Parsers<br>5.5 Output-Fixing Parser<br>[Module 4 Program](./assignments/assignment_yourname_t81_559_class4.ipynb) due 02/10/2026                                                                        |
| [Module 6](t81_559_class_06_1_rag.ipynb)<br>**Meet on 02/16/2026**             | **Retrieval-Augmented Generation (RAG)**<br>6.1 Introduction to RAG<br>6.2 ChromaDB<br>6.3 Embeddings<br>6.4 Q&A over Documents<br>6.5 Embedding Databases<br>[Module 5 Program](./assignments/assignment_yourname_t81_559_class5.ipynb) due 02/17/2026<br>**We will meet on campus this week (in-class meeting #3)**                          |
| [Module 7](t81_559_class_07_1_agents.ipynb)<br>Week of 02/23/2026              | **LangChain: Agents**<br>7.1 Introduction to Agents<br>7.2 Agent Tools<br>7.3 Retrieval & Search Tools<br>7.4 Agent Construction<br>7.5 Custom Agents<br>[Module 6 Program](./assignments/assignment_yourname_t81_559_class6.ipynb) due 02/23/2026                                                                                             |
| [Module 8](t81_559_class_08_1_kaggle_intro.ipynb)<br>**Meet on 03/02/2026**    | **Kaggle Assignment**<br>8.1 Introduction to Kaggle<br>8.2 Kaggle Notebooks<br>8.3 Small LLMs<br>8.4 Running LLMs on Kaggle<br>8.5 Semester Kaggle Project<br>[Module 7 Program](./assignments/assignment_yourname_t81_559_class7.ipynb) due 03/02/2026<br>**We will meet on campus this week (in-class meeting #4)**                          |
| [Module 9](t81_559_class_09_1_image_genai.ipynb)<br>Week of 03/16/2026         | **Multimodal & Text-to-Image**<br>9.1 Multimodal Intro<br>9.2 DALL·E Generation<br>9.3 Image Editing<br>9.4 Multimodal Models<br>9.5 Illustrated Book<br>[Module 8 Program](./assignments/assignment_yourname_t81_559_class8.ipynb) due 03/17/2026                                                                                             |
| [Module 10](t81_559_class_10_1_streamlit.ipynb)<br>Week of 03/23/2026          | **Streamlit**<br>10.1 Streamlit in Colab<br>10.2 Intro<br>10.3 State Management<br>10.4 Chat Apps<br>10.5 Advanced Chat Apps<br>[Module 9 Program](./assignments/assignment_yourname_t81_559_class9.ipynb) due 03/24/2026                                                                                                                      |
| [Module 11](t81_559_class_11_1_finetune.ipynb)<br>Week of 03/30/2026           | **Fine Tuning**<br>11.1 When to Fine Tune<br>11.2 Dataset Prep<br>11.3 OpenAI Fine Tuning<br>11.4 Applications<br>11.5 Evaluation<br>[Module 10 Program](./assignments/assignment_yourname_t81_559_class10.ipynb) due 03/31/2026                                                                                                               |
| [Module 12](t81_559_class_12_1_prompt_engineering.ipynb)<br>Week of 04/06/2026 | **Prompt Engineering**<br>12.1 Intro<br>12.2 Few-Shot & Chain-of-Thought<br>12.3 Personas<br>12.4 Refinement & Verification<br>12.5 Structured Prompts                                                                                                                                                                                         |
| [Module 13](t81_559_class_13_1_speech_models.ipynb)<br>Week of 04/13/2026      | **Speech Processing**<br>13.1 Voice ChatBots<br>13.2 Speech Generation<br>13.3 Speech Recognition<br>13.4 Voice ChatBot App<br>13.5 Future Directions<br>Kaggle Assignment due **04/17/2026 (midnight)**                                                                                                                                       |
| Week 14<br>Week of 04/20/2026                                                  | **Future Topics**<br>Final Project due **04/27/2026**                                                                                                                                                                                                                                                                                          |
