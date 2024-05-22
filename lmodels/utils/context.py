from lmodels.types import AnnotatedConversation


def merge_system_messages(
    conversations: list[AnnotatedConversation],
) -> list[AnnotatedConversation]:
    """
    Merge system messages into the user/assistant messages in the conversation.

    ### Parameters
    --------------
    - `conversation`: the conversation in which to merge the system messages.

    ### Returns
    --------------
    The conversation with the system messages merged into the user messages.
    """

    merged = []
    for conversation in conversations:
        if len(conversation) < 2:
            merged.append(conversation)
            continue

        for i in range(len(conversation) - 1):
            if "role" in conversation[i] and conversation[i]["role"] == "system":
                conversation[i + 1]["content"] = (
                    conversation[i]["content"] + "\n" + conversation[i + 1]["content"]
                )

        merged.append(
            [message for message in conversation if message["role"] != "system"]
        )

    return merged
