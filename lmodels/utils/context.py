from lmodels.types import AnnotatedConversation


def merge_system_messages(conversation: AnnotatedConversation) -> AnnotatedConversation:
    """
    Merge system messages into the user/assistant messages in the conversation.

    ### Parameters
    --------------
    - `conversation`: the conversation in which to merge the system messages.

    ### Returns
    --------------
    The conversation with the system messages merged into the user messages.
    """

    if len(conversation) < 2:
        return conversation

    for i in range(len(conversation) - 1):
        if "role" not in conversation[i]:
            continue

        if conversation[i]["role"] == "system":
            conversation[i + 1]["content"] = (
                conversation[i]["content"] + "\n" + conversation[i + 1]["content"]
            )
    return [message for message in conversation if message["role"] != "system"]
