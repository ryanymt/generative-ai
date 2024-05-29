import os

import streamlit as st
import vertexai
from vertexai.preview.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)

PROJECT_ID = os.environ.get("GCP_PROJECT")  # Your Google Cloud Project ID
LOCATION = os.environ.get("GCP_REGION")  # Your Google Cloud Project Region
vertexai.init(project=PROJECT_ID, location=LOCATION)


@st.cache_resource
def load_models():
    text_model_pro = GenerativeModel("gemini-1.5-pro-preview-0409")
    #text_model_pro = GenerativeModel("gemini-pro")
    multimodal_model_pro = GenerativeModel("gemini-pro-vision")
    return text_model_pro, multimodal_model_pro


def get_gemini_pro_text_response(
    model: GenerativeModel,
    contents: str,
    generation_config: GenerationConfig,
    stream: bool = True,
):
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    responses = model.generate_content(
        prompt,
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=stream,
    )

    final_response = []
    for response in responses:
        try:
            # st.write(response.text)
            final_response.append(response.text)
        except IndexError:
            # st.write(response)
            final_response.append("")
            continue
    return " ".join(final_response)


def get_gemini_pro_vision_response(
    model, prompt_list, generation_config={}, stream: bool = True
):
    generation_config = {"temperature": 0.1, "max_output_tokens": 2048}
    responses = model.generate_content(
        prompt_list, generation_config=generation_config, stream=stream
    )
    final_response = []
    for response in responses:
        try:
            final_response.append(response.text)
        except IndexError:
            pass
    return "".join(final_response)


st.header("BDR Discovery Questions", divider="rainbow")
text_model_pro, multimodal_model_pro = load_models()

tab1, tab2  = st.tabs(
    ["Basic Questions", "Generated Questions"]
)

with tab1:
    st.write("Using Gemini 1.5  model")
    st.subheader("Basic Questions")

    # Story premise
    company_desc = st.text_input(
        "What is the industry of the Company ? Or explain the business nature of the company ?  \n\n", 
        key="company_desc", 
        value="mobile platform that brings efficient marketing insights to brands and stores"
    )
    it_landscape = st.text_input(
        "How is your IT landscape looks like ?  \n\n", 
        key="it_landscape", 
        value="Currently on AWS and has a little footprint on Google Cloud "
    )
    it_challenges = st.text_input(
        "Do you have any challenges for current IT landscape ?  \n\n",
        key="it_challenges",
        value="Not at all",
    )
    it_initiative = st.text_input(
        "Do you have any initiative or project for this year ?  \n\n",
        key="it_initiative",
        value="Data Analytics and predictions, Cost optimisation",
    )

    temperature = 0.95

    max_output_tokens = 2048

    prompt = f"""You are Business Development Representative from Google Cloud. You are having an initial discovery call with the customer. You have asked a few questions and received answers as below. \n
    Question: What is the industry of the Company ? Or explain the business nature of the company ? \n
    Answer: {company_desc} \n
    Question: How is your IT landscape looks like ? \n
    Answer: {it_landscape} \n
    Question: Do you have any challenges for current IT landscape ? \n
    Answer: {it_challenges} \n
    Question: Do you have any initiative or project for this year ? \n
    Answer: {it_initiative} \n\n
    Base on the information above, generate a few questions to ask more. Generate only the Questions. 3 questions on challenges only if applicable and 3 questions on each initiatives. Questions should also cover their motivation and timeline requirement.
    """
    # config = GenerationConfig(
    #     temperature=temperature,
    #     candidate_count=1,
    #     max_output_tokens=max_output_tokens,
    # )

    config = {
        "temperature": 0.8,
        "max_output_tokens": 2048,
    }

    generate_t2t = st.button("Generate Questions", key="generate_t2t")
    if generate_t2t and prompt:
        with st.spinner("Generating your questions using Gemini..."):
            response = get_gemini_pro_text_response(
                text_model_pro, prompt, generation_config=config
            )

            if response:
                questions = response.split("\n")  # Split on newlines
                with tab2:
                    st.subheader("Generated Questions")
                    for i, question in enumerate(questions):
                        st.text_area(question, key=f"{question}-{i}")
#    if generate_t2t and prompt:
#        # st.write(prompt)
#        with st.spinner("Generating your questions using Gemini..."):
#            first_tab1, first_tab2 = st.tabs(["Story", "Prompt"])
#            with first_tab1:
#                response = get_gemini_pro_text_response(
#                    text_model_pro,
#                    prompt,
#                    generation_config=config,
#                )
#
#                if response:
#                    questions = response.split("\n")  # Split on newlines
#                with tab2:
#                    st.subheader("Generated Questions")
#                    for question in questions:
#                        st.text_area(question, key=question)
#                if response:
#                    st.write("Your questions:")
#                    st.write(response)
#            with first_tab2:
#                st.text(prompt)
