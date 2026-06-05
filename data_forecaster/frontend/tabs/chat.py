import streamlit as st
import plotly.graph_objects as go
from api_service import ForecastingAPI

def render_chat_tab(info):
    st.subheader("💬 Data Explorer")
    st.info("Ask questions about your data or the analysis results stored in my memory.")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("viz"):
                viz = message["viz"]
                if viz["type"] == "pie":
                    fig = go.Figure(data=[go.Pie(labels=viz["data"]["labels"], values=viz["data"]["values"])])
                    st.plotly_chart(fig, use_container_width=True)

    if prompt := st.chat_input("How many datapoints are missing?"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    resp = ForecastingAPI.send_chat(info["file_id"], prompt)
                    if resp.status_code == 200:
                        data = resp.json()
                        answer = data["answer"]
                        st.markdown(answer)
                        
                        viz_payload = None
                        if data.get("visualization_type") == "pie":
                            viz_data = data["visualization_data"]
                            fig = go.Figure(data=[go.Pie(labels=viz_data["labels"], values=viz_data["values"])])
                            st.plotly_chart(fig, use_container_width=True)
                            viz_payload = {"type": "pie", "data": viz_data}

                        st.session_state.chat_history.append({"role": "assistant", "content": answer, "viz": viz_payload})
                    else:
                        st.error(f"Error: {resp.json().get('detail', 'Failed to get response.')}")
            except Exception as e:
                st.error(f"Connection error: {e}")