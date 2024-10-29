import logging
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import LoraConfig, get_peft_model

# Set logging to ignore warnings
logging.getLogger().setLevel(logging.CRITICAL)

# Load model and tokenizer with LoRA configuration
def load_model():
    model_name = "klyang/MentaLLaMA-chat-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Apply LoRA configuration
    lora_config = LoraConfig.from_pretrained("Llama-2-7b-chat-finetune")
    model = get_peft_model(model, lora_config)
    return tokenizer, model

# Define chatbot function with a psychiatrist response style
def psychiatrist_chatbot(prompt, model, tokenizer):
    # Initialize text generation pipeline
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
    
    # Preface the prompt with an instruction for psychiatric empathy and understanding
    formatted_prompt = f"<s>[INST] As a compassionate and understanding psychiatrist, please respond thoughtfully to: {prompt} [/INST]"
    
    # Generate response
    result = pipe(formatted_prompt)
    
    # Clean up response text
    response_text = result[0]['generated_text']
    return response_text.replace("<s>[INST]", "").replace("[/INST]", "").strip()

# Streamlit interface
def main():
    # Load model and tokenizer
    tokenizer, model = load_model()
    
    # Set up Streamlit page
    st.set_page_config(page_title="Psychiatrist Chatbot", page_icon="🧠")
    
    # Header and description
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Mental Health Support Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>I'm here to listen and provide support.</h3>", unsafe_allow_html=True)

    # Initialize session state for storing conversation and input
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    # Display conversation with styling
    for message in st.session_state.conversation:
        if message["role"] == "user":
            st.markdown(f"<div style='text-align: right; color: white; background-color: #4CAF50; padding: 10px; border-radius: 10px; margin: 5px 0; max-width: 75%;'>{message['content']}</div>", unsafe_allow_html=True)
        elif message["role"] == "bot":
            st.markdown(f"<div style='text-align: left; color: black; background-color: #D3D3D3; padding: 10px; border-radius: 10px; margin: 5px 0; max-width: 75%;'>{message['content']}</div>", unsafe_allow_html=True)

    # User input area at the bottom
    user_input = st.text_input("You:", value=st.session_state.user_input, key="unique_user_input")

    if user_input:
        # Add user input to conversation
        st.session_state.conversation.append({"role": "user", "content": user_input})
        
        # Generate response with psychiatrist tone
        with st.spinner("Bot is thinking..."):
            bot_response = psychiatrist_chatbot(user_input, model, tokenizer)
        
        # Add bot response to conversation
        st.session_state.conversation.append({"role": "bot", "content": bot_response})

        # Clear user input after processing
        st.session_state.user_input = ""

    # Add a footer with additional information
    st.markdown("<footer style='text-align: center; margin-top: 20px;'>© 2024 Mental Health Support Chatbot</footer>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()



# # Load model and tokenizer
# model_name = "klyang/MentaLLaMA-chat-7B"
# tokenizer_name = model_name

# # Load the model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # Set up LoRA configuration and apply it to the model
# lora_config = LoraConfig.from_pretrained('Llama-2-7b-chat-finetune')
# model = get_peft_model(model, lora_config)

# # Set the model to evaluation mode
# model.eval()

# # Initialize chat history in session state
# if 'history' not in st.session_state:
#     st.session_state.history = []

# # Streamlit app
# st.title("Chatbot with Pretrained LLM")
# st.write("Talk to your model!")

# # Input text box for user queries
# user_input = st.text_input("You: ", "")

# # Button to send the query
# if st.button("Send"):
#     if user_input:
#         # Encode the user input and generate a response
#         inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        
#         # Move tensors to the same device as the model
#         inputs = inputs.to(model.device)

#         with torch.no_grad():
#             outputs = model.generate(inputs, max_length=100, num_return_sequences=1)

#         # Decode and display the response
#         response = tokenizer.decode(outputs[0], skip_special_tokens=True)

#         # Store history
#         st.session_state.history.append(f"You: {user_input}")
#         st.session_state.history.append(f"Bot: {response}")

#         # Display chat history
#         st.text_area("Chat History:", value="\n".join(st.session_state.history), height=300)
