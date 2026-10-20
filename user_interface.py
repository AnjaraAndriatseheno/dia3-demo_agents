import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os
import base64
from llm_agent import ConversationAgent


class UserInterface:
	def __init__(self):
		st.set_page_config(page_title="Jarvis")
		st.title("J.A.R.V.I.S")
		self.conversation_agent_object = self.get_persistant_conversation_agent()
		self.speech_to_text_object = self.get_persistant_speech_to_text()
		self.image_analyzer_object = self.get_persistant_image_analyzer()
		self.image_marker = "\n\n[Image envoyée — description : "

	def get_persistant_conversation_agent(self):
		if "conversation_agent_object" not in st.session_state:
			st.session_state.conversation_agent_object = ConversationAgent()

		return st.session_state.conversation_agent_object

	def get_persistant_speech_to_text(self):
		if "speech_to_text_object" not in st.session_state:
			st.session_state.speech_to_text_object = SpeechToText()

		return st.session_state.speech_to_text_object

	def get_persistant_image_analyzer(self):
		if "image_analyzer_object" not in st.session_state:
			st.session_state.image_analyzer_object = ImageAnalyzer()

		return st.session_state.image_analyzer_object

	def show_discussion_history(self):
		for message in self.conversation_agent_object.get_conversation_history():
			content = message["content"]

			if self.image_marker in content:
				content = content.split(self.image_marker)[0] + "\n\n📷 *Image jointe*"

			with st.chat_message(message["role"]):
				st.markdown(content)

	def get_text_from_interaction(self, interaction):
		if interaction.audio:
			try:
				text = self.speech_to_text_object.transcribe(interaction.audio)
			except Exception:
				st.error("Je n'ai pas compris l'audio, réessaie.")
				return None
		else:
			text = interaction.text

		if interaction.files:
			try:
				with st.spinner("Analyse de l'image..."):
					description = self.image_analyzer_object.describe(interaction.files[0])
			except Exception:
				st.error("Je n'ai pas réussi à analyser l'image.")
				return None

			if not text:
				text = "Que vois-tu sur cette image ?"

			text = f"{text}{self.image_marker}{description}]"

		return text

	def run(self):
		interaction = st.chat_input(
			"Que veux tu moussaillon ?",
			accept_file=True,
			file_type=["png", "jpg", "jpeg", "webp"],
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


class ImageAnalyzer:
	def __init__(self):
		load_dotenv()
		self.groq_client = Groq(api_key=os.environ["GROQ_KEY"])
		self.model = "qwen/qwen3.8-27b"

	def encode_image(self, image_file):
		return base64.b64encode(image_file.getvalue()).decode("utf-8")

	def describe(self, image_file):
		base64_image = self.encode_image(image_file)

		response = self.groq_client.chat.completions.create(
			model=self.model,
			messages=[
				{
					"role": "user",
					"content": [
						{
							"type": "text",
							"text": "Décris cette image en détail, en français. Si elle contient du texte, retranscris-le.",
						},
						{
							"type": "image_url",
							"image_url": {
								"url": f"data:{image_file.type};base64,{base64_image}",
							},
						},
					],
				}
			],
		)
		return response.choices[0].message.content


if __name__ == "__main__":
	user_interface_object = UserInterface()
	user_interface_object.run()