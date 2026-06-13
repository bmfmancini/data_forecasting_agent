"""
UI utility functions for the Time Series Data Forecaster Agent.
Provides helper functions for rendering UI components and handling user interactions.
"""

import streamlit as st
from typing import Any, Dict, List


def render_reasoning(steps: List[Dict[str, Any]]) -> None:
    """
    Helper to render agent reasoning traces in an expander.

    Args:
        steps: List of reasoning steps to display
    """
    if not steps:
        st.info("No detailed reasoning trace captured for this step.")
        return

    for i, step in enumerate(steps):
        with st.container():
            st.markdown(f"**Step {i+1}**")
            thought = (step.get("thought") or step.get("log") or "").strip()

            if thought.lower().startswith("thought:"):
                thought = thought[8:].strip()

            if thought:
                st.caption(thought)

            if step.get("observation"):
                st.info(f"Observation: {step['observation']}")


def preflight_defaults(preflight: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract default values from preflight decisions.

    Args:
        preflight: Preflight dictionary containing decisions

    Returns:
        Dictionary mapping decision keys to default values
    """
    return {
        decision["key"]: decision["default"]
        for decision in preflight.get("decisions", [])
    }


def render_preflight_contents(
    preflight: Dict[str, Any], disabled: bool = False
) -> Dict[str, Any]:
    """
    Render preflight content and return user selections.

    Args:
        preflight: Preflight dictionary containing decisions
        disabled: Whether to disable UI elements

    Returns:
        Dictionary of user selections
    """
    if preflight.get("detected_frequency"):
        st.caption(f"Selected-series frequency: **{preflight['detected_frequency']}**")

    for message in preflight.get("issues", []):
        st.info(message)
    for message in preflight.get("warnings", []):
        st.warning(message)

    choices = dict(
        st.session_state.get("_preflight_options_current")
        or preflight_defaults(preflight)
    )
    for decision in preflight.get("decisions", []):
        key = decision["key"]
        options = decision["options"]
        default = choices.get(key, decision["default"])
        default_index = options.index(default) if default in options else 0
        choices[key] = st.selectbox(
            decision["label"],
            options=options,
            index=default_index,
            help=decision["message"],
            disabled=disabled,
            key=f"preflight_choice_{key}",
        )
    return choices


def render_preflight_dialog_content(
    preflight: dict[str, Any], disabled: bool = False
) -> bool:
    choices = render_preflight_contents(preflight, disabled=disabled)
    if st.button(
        "Apply Preflight Choices",
        disabled=disabled,
        use_container_width=True,
        key="preflight_apply_btn",
    ):
        st.session_state._preflight_options_current = choices
        st.rerun()
    return False
