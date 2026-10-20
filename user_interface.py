import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os
from llm_agent import ConversationAgent


class UserInterface:
	def __init__(self):
		st.set_page_config(page_title="Jarvis")
		st.title("Barbe-Quasar")
		self.conversation_agent_object = self.get_persistant_conversation_agent()
		self.speech_to_text_object = self.get_persistant_speech_to_text()

	def get_persistant_conversation_agent(self):
		if "conversation_agent_object" not in st.session_state:
			st.session_state.conversation_agent_object = ConversationAgent()

		return st.session_state.conversation_agent_object

	def get_persistant_speech_to_text(self):
		if "speech_to_text_object" not in st.session_state:
			st.session_state.speech_to_text_object = SpeechToText()

		return st.session_state.speech_to_text_object

	def show_discussion_history(self):
		for message in self.conversation_agent_object.get_conversation_history():
			with st.chat_message(message["role"]):
				st.markdown(message["content"])

	def get_text_from_interaction(self, interaction):
		if interaction.audio:
			st.write("Audio reçu :")
			st.audio(interaction.audio)
			try:
				text = self.speech_to_text_object.transcribe(interaction.audio)
				st.write(f"Transcription : '{text}'")
				return text
			except Exception as e:
				st.error(f"Erreur de transcription : {e}")
				return None

		return interaction.text

	def run(self):
		interaction = st.chat_input(
			"Que veux tu moussaillon ?",
			accept_file=False,
			accept_audio=True,
		)

		if interaction:
			text_interaction = self.get_text_from_interaction(interaction)

			if text_interaction:
				self.conversation_agent_object.ask_llm(text_interaction)

		self.show_discussion_history()


class SpeechToText:
	def __init__(self):
		load_dotenv()
		self.groq_client = Groq(api_key=os.environ["GROQ_KEY"])
		self.model = "whisper-large-v3"

	def transcribe(self, audio_file):
		transcription = self.groq_client.audio.transcriptions.create(
			file=("audio.wav", audio_file.getvalue()),
			model=self.model,
			language="fr",
			temperature=0,
		)
		return transcription.text


if __name__ == "__main__":
	user_interface_object = UserInterface()
	user_interface_object.run()