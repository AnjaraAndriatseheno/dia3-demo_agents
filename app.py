import streamlit as st 
from llm_agent import ConversationAgent  


st.set_page_config(page_title="Jarvis")
st.title("Barbe-Quasar")



if "agent" not in st.session_state:
	st.session_state.agent = ConversationAgent()



for message in st.session_state.agent.conversation_history:
	if message["role"] != "system":
		with st.chat_message(message["role"]):
			st.markdown(message["content"])


user_input = st.chat_input(
	"Que veux tu moussaillon ?",
	accept_file=False,
	accept_audio=False
	)

if user_input :
	response = st.session_state.agent.ask_llm(user_input)  
	
	with st.chat_message("user"):
		st.markdown(user_input)

	with st.chat_message("assistant"):  
		st.markdown(response)