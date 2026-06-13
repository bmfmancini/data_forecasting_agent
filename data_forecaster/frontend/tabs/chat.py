import streamlit as st
import plotly.graph_objects as go
from api_service import ForecastingAPI
from utils.visualization_utils import DynamicVisualizer, parse_llm_visualization_response

def render_chat_tab(info):
    st.subheader("💬 Data Explorer")
    st.info("Ask questions about time series data analysis or general forecasting concepts.")
    
    # Initialize chat history if not exists
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Initialize visualizer
    visualizer = DynamicVisualizer()

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("viz"):
                viz = message["viz"]
                if viz["type"] == "pie":
                    fig = go.Figure(data=[go.Pie(labels=viz["data"]["labels"], values=viz["data"]["values"])])
                    st.plotly_chart(fig, use_container_width=True)
                elif viz["type"] == "dynamic":
                    visualizer.render_from_config(viz["data"], key=f"chat_viz_{len(st.session_state.chat_history)}")

    if prompt := st.chat_input("Ask about time series forecasting..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    # Send chat with file_id if available, otherwise send without
                    file_id = info.get("file_id") if info else None
                    resp = ForecastingAPI.send_chat(file_id, prompt)
                    if resp.status_code == 200:
                        data = resp.json()
                        answer = data["answer"]
                        st.markdown(answer)
                        
                        viz_payload = None
                        viz_rendered = False
                        
                        # Handle predefined visualization types
                        if data.get("visualization_type") == "pie":
                            viz_data = data["visualization_data"]
                            fig = go.Figure(data=[go.Pie(labels=viz_data["labels"], values=viz_data["values"])])
                            st.plotly_chart(fig, use_container_width=True)
                            viz_payload = {"type": "pie", "data": viz_data}
                            viz_rendered = True
                        
                        # Handle dynamic visualizations from LLM
                        elif data.get("visualization_data"):
                            viz_data = data["visualization_data"]
                            if visualizer.render_from_config(viz_data, key=f"chat_viz_new_{len(st.session_state.chat_history)}"):
                                viz_payload = {"type": "dynamic", "data": viz_data}
                                viz_rendered = True
                        
                        # Try to parse visualization from the answer text as a fallback
                        elif answer and not viz_rendered:
                            parsed_viz = parse_llm_visualization_response(answer)
                            if parsed_viz:
                                if visualizer.render_from_config(parsed_viz, key=f"chat_viz_parsed_{len(st.session_state.chat_history)}"):
                                    viz_payload = {"type": "dynamic", "data": parsed_viz}
                                    # Remove the JSON part from the answer text for cleaner display
                                    # This is a simple approach - you might want more sophisticated text cleaning
                                    clean_answer = answer.replace(json.dumps(parsed_viz), "").strip()
                                    if clean_answer != answer:
                                        # Update the displayed answer
                                        st.markdown(clean_answer)

                        st.session_state.chat_history.append({"role": "assistant", "content": answer, "viz": viz_payload})
                    else:
                        st.error(f"Error: {resp.json().get('detail', 'Failed to get response.')}")
            except Exception as e:
                st.error(f"Connection error: {e}")