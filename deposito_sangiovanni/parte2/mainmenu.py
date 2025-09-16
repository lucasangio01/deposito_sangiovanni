import streamlit as st
from streamlit_option_menu import option_menu
from openai import AzureOpenAI
import time

class Sidebar:
    @staticmethod
    def render():
        with st.sidebar:
            selected = option_menu(
                "Menu",
                ["Chat", "Impostazioni"],
                menu_icon="rocket-fill",
                icons=["chat-left-text-fill", "gear"]
            )
        return selected


class SettingsPage:
    @staticmethod
    def render():
        st.title("Inserisci le tue informazioni")
        st.divider()
        st.session_state["api_key"] = st.text_input(
            "Inserisci la tua key",
            value=st.session_state.get("api_key", "")
        )
        st.session_state["endpoint"] = st.text_input(
            "Inserisci il tuo endpoint",
            value=st.session_state.get("endpoint", "")
        )
        st.session_state["api_version"] = st.text_input(
            "Inserisci la tua versione API",
            value=st.session_state.get("api_version", "")
        )

class ChatPage:
    @staticmethod
    def render():
        st.title("ChatGPT-like clone (Azure Streaming)")

        # Recupera i dati salvati dall'utente
        api_key = st.session_state.get("api_key")
        endpoint = st.session_state.get("endpoint")
        api_version = st.session_state.get("api_version")

        # Se mancano credenziali -> avvisa l’utente
        if not api_key or not endpoint or not api_version:
            st.warning("⚠️ Inserisci le credenziali nella pagina 'Impostazioni' prima di usare la chat.")
            return
        
        start = time.time()

        # Inizializza client
        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version
        )

        elapsed = time.time() - start
        st.write(f"Connesso dopo {round(elapsed, 2)} secondi")

        DEPLOYMENT_NAME = "gpt-4o"

        st.divider()

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Mostra messaggi precedenti
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        # Input utente
        if prompt := st.chat_input("Scrivi qualcosa..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Stream della risposta
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""

                stream = client.chat.completions.create(
                    model=DEPLOYMENT_NAME,
                    messages=st.session_state.messages,
                    stream=True,
                )

                for chunk in stream:
                    if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        placeholder.markdown(full_response + "▌")  # cursore

                placeholder.markdown(full_response)  # rimuove il cursore finale

            st.session_state.messages.append({"role": "assistant", "content": full_response})


 

st.set_page_config(page_title="Menu", layout="wide", initial_sidebar_state="expanded")

# Render sidebar
selected_page = Sidebar.render()

# Render selected page
if selected_page == "Chat":
    ChatPage.render()
elif selected_page == "Impostazioni":
    SettingsPage.render()
